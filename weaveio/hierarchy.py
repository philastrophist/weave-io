import inspect
from typing import Tuple, Dict, Type, Union, List
from warnings import warn

import networkx as nx
from graphviz import Source

from .config_tables import progtemp_config
from .graph import Graph
from .writequery import CypherQuery, Unwind
from .context import ContextError
from .utilities import Varname


def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def graph2pdf(graph, ftitle):
    dot = nx.nx_pydot.to_pydot(graph)
    dot.set_strict(False)
    # dot.obj_dict['attributes']['splines'] = 'ortho'
    dot.obj_dict['attributes']['nodesep'] = '0.5'
    dot.obj_dict['attributes']['ranksep'] = '0.75'
    dot.obj_dict['attributes']['overlap'] = False
    dot.obj_dict['attributes']['penwidth'] = 18
    dot.obj_dict['attributes']['concentrate'] = False
    Source(dot).render(ftitle, cleanup=True, format='pdf')


lightblue = '#69A3C3'
lightgreen = '#71C2BF'
red = '#D08D90'
orange = '#DFC6A1'
purple = '#a45fed'
pink = '#d50af5'

hierarchy_attrs = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
abstract_hierarchy_attrs = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
factor_attrs = {'type': 'factor', 'style': 'filled', 'fillcolor': orange, 'shape': 'box', 'edgecolor': orange}
identity_attrs = {'type': 'id', 'style': 'filled', 'fillcolor': purple, 'shape': 'box', 'edgecolor': purple}
product_attrs = {'type': 'factor', 'style': 'filled', 'fillcolor': pink, 'shape': 'box', 'edgecolor': pink}
l1file_attrs = {'type': 'file', 'style': 'filled', 'fillcolor': lightblue, 'shape': 'box', 'edgecolor': lightblue}
l2file_attrs = {'type': 'file', 'style': 'filled', 'fillcolor': lightgreen, 'shape': 'box', 'edgecolor': lightgreen}
rawfile_attrs = l1file_attrs

FORBIDDEN_LABELS = []
FORBIDDEN_PROPERTY_NAMES = []
FORBIDDEN_LABEL_PREFIXES = ['_']
FORBIDDEN_PROPERTY_PREFIXES = ['_']
FORBIDDEN_IDNAMES = ['idname']


class RuleBreakingException(Exception):
    pass


class Multiple:
    def __init__(self, node, minnumber=1, maxnumber=None, constrain=None):
        self.node = node
        self.minnumber = minnumber
        self.maxnumber = maxnumber
        self.name = node.plural_name
        self.singular_name = node.singular_name
        self.plural_name = node.plural_name
        self.idname = self.node.idname
        try:
            self.factors =  self.node.factors
        except AttributeError:
            self.factors = []
        try:
            self.parents = self.node.parents
        except AttributeError:
            self.parents = []
        self.constrain = [] if constrain is None else constrain

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}])>"


class GraphableMeta(type):
    def __new__(meta, name: str, bases, _dct):
        dct = {'is_template': False}
        dct.update(_dct)
        dct['aliases'] = dct.get('aliases', [])
        dct['aliases'] += [a for base in bases for a in base.aliases]
        if dct.get('plural_name', None) is None:
            dct['plural_name'] = name.lower() + 's'
        dct['singular_name'] = name.lower()
        dct['plural_name'] = dct['plural_name'].lower()
        dct['singular_name'] = dct['singular_name'].lower()
        idname = dct.get('idname', None)
        if idname in FORBIDDEN_IDNAMES:
            raise RuleBreakingException(f"You may not name an id as one of {FORBIDDEN_IDNAMES}")
        if not (isinstance(idname, str) or idname is None):
            raise RuleBreakingException(f"{name}.idname ({idname}) must be a string or None")
        if name[0] != name.capitalize()[0] or '_' in name:
            raise RuleBreakingException(f"{name} must have `CamelCaseName` style name")
        for factor in dct.get('factors', []) + ['idname'] + [dct['singular_name'], dct['plural_name']]:
            if factor != factor.lower():
                raise RuleBreakingException(f"{name}.{factor} must have `lower_snake_case` style name")
            if factor in FORBIDDEN_PROPERTY_NAMES:
                raise RuleBreakingException(f"The name {factor} is not allowed for class {name}")
            if any(factor.startswith(p) for p in FORBIDDEN_PROPERTY_PREFIXES):
                raise RuleBreakingException(f"The name {factor} may not start with any of {FORBIDDEN_PROPERTY_PREFIXES} for {name}")
        r = super(GraphableMeta, meta).__new__(meta, name, bases, dct)
        return r


class Graphable(metaclass=GraphableMeta):
    idname = None
    name = None
    identifier = None
    indexer = None
    type_graph_attrs = {}
    plural_name = None
    singular_name = None
    parents = []
    uses_tables = False
    factors = []
    data = None
    query = None
    is_template = False
    products = []

    @classmethod
    def requirement_names(cls):
        l = []
        for p in cls.parents:
            if isinstance(p, type):
                if issubclass(p, Graphable):
                    l.append(p.singular_name)
            else:
                if isinstance(p, Multiple):
                    l.append(p.plural_name)
                else:
                    raise RuleBreakingException(f"The parent list of a Hierarchy must contain "
                                                f"only other Hierarchies or Multiple(Hierarchy)")
        return l

    def add_parent_data(self, data):
        self.data = data

    def add_parent_query(self, query):
        self.query = query

    def __getattr__(self, item):
        if self.query is not None:
            warn('Lazily loading a hierarchy attribute can be costly. Consider using a more flexible query.')
            attribute = getattr(self.query, item)()
            setattr(self, item, attribute)
            return attribute
        raise AttributeError(f"Query not added to {self}, cannot search for {self}.{item}")

    @property
    def neotypes(self):
        clses = [i.__name__ for i in inspect.getmro(self.__class__)]
        clses = clses[:clses.index('Graphable')]
        return clses[::-1]

    @property
    def neoproperties(self):
        d = {}
        if self.identifier is None and self.idname is not None:
            raise ValueError(f"{self} must have an identifier")
        if self.idname is None and self.identifier is not None:
            raise ValueError(f"{self} must have an idname to be given an identifier")
        else:
            d[self.idname] = self.identifier
            d['id'] = self.identifier
        for f in self.factors:
            value = getattr(self, f.lower())
            if value is not None:
                d[f.lower()] = value
        if self.parents:
            d['parents'] = self.parents
        return d

    def __init__(self, _protect=None, **predecessors):
        if _protect is None:
            _protect = []
        self.predecessors = predecessors
        self.data = None
        try:
            graph = Graph.get_context().query  # type: CypherQuery
            relationships = []
            for k, parent_list in predecessors.items():
                if self.indexer is None:
                    type = 'is_required_by'
                elif k in self.indexer.lower():
                    type = 'indexes'
                else:
                    type = 'is_required_by'
                if isinstance(parent_list, (list, tuple)):
                    for iparent, parent in enumerate(parent_list):
                        relationships.append(Relationship(parent.node, child, type, order=iparent))
                else:
                    order = 0 if k in _protect else None
                    parent = parent_list
                    relationships.append(Relationship(parent.node, child, type, order=order))

            self.node = child
            if self.identifier is not None:
                graph.merge_node(child)
                graph.merge_relationships(relationships)
            else:
                graph.merge_node_and_parent_relationships(relationships)
        except ContextError:
            pass


class Hierarchy(Graphable):
    parents = []
    factors = []
    indexer = None
    identifier_builder = None

    def generate_identifier(self):
        """
        if `idname` is set, then return the identifier set at instantiation
        otherwise, make an identifier by stitching together identifiers of its input
        """
        if self.identifier is not None:
            return self.identifier
        elif len(self.identifier_builder):
            strings = []
            identifiers = []
            for i in self.identifier_builder:
                obj = getattr(self, i)  # type: Union[List[Union[Hierarchy, str]], Hierarchy, str]
                if not isinstance(obj, (list, tuple)):
                    obj = [obj]
                for o in obj:
                    if isinstance(o, Hierarchy):
                        identifiers.append(o.identifier)
                    else:
                        identifiers.append(o)
            for i in identifiers:
                if isinstance(i, (Unwind, Varname)):
                    s = str(i)
                    strings += ['+', s, '+']
                else:
                    s = str(i)
                    strings.append(f"+'{s}'+")
            final = ''.join(strings).strip('+').replace('++', '+').replace("''", "")
            return Varname(final)
        else:
            return None

    def __repr__(self):
        return self.name

    def make_specification(self) -> Tuple[Dict[str, Type[Graphable]], Dict[str, str]]:
        """
        Make a dictionary of {name: HierarchyClass} and a similar dictionary of factors
        """
        parents = {p.__name__.lower() if isinstance(p, type) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        return specification, factors

    def __init__(self, tables=None, **kwargs):
        _protect = kwargs.pop('_protect', [])
        self.uses_tables = False
        if tables is None:
            for value in kwargs.values():
                if isinstance(value, Unwind):
                    self.uses_tables = True
                elif isinstance(value, Hierarchy):
                    self.uses_tables = value.uses_tables
        else:
            self.uses_tables = True
        self.identifier = kwargs.pop(self.idname, None)
        self.specification, factors = self.make_specification()
        # add any data held in a neo4j unwind table
        for k, v in self.specification.items():
            if k not in kwargs:
                if tables is not None:
                    if k in tables:
                        kwargs[k] = tables[k]
        self._kwargs = kwargs.copy()
        # Make predecessors a dict of {name: [instances of required Factor/Hierarchy]}
        predecessors = {}
        for name, nodetype in self.specification.items():
            value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple):
                if not isinstance(value, (tuple, list)):
                    if isinstance(value, Graphable):
                        if not getattr(value, 'uses_tables', False):
                            raise TypeError(f"{name} expects multiple elements")
            else:
                value = [value]
            if name not in factors:
                predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")
        self.predecessors = predecessors
        if self.identifier_builder is not None:
            if self.identifier is not None:
                raise RuleBreakingException(f"{self} must not take an identifier if it has an identifier_builder")
            self.identifier = self.generate_identifier()
        setattr(self, self.idname, self.identifier)
        self.name = f"{self.__class__.__name__}({self.idname}={self.identifier})"
        super(Hierarchy, self).__init__(**predecessors, _protect=_protect)


class CASU(Hierarchy):
    idname = 'version'


class APS(Hierarchy):
    idname = 'version'


class Simulator(Hierarchy):
    idname = 'version'
    factors = ['simvdate', 'simver', 'simmode']


class System(Hierarchy):
    idname = 'version'


class ArmConfig(Hierarchy):
    factors = ['resolution', 'vph', 'camera', 'colour']

    def __init__(self, tables=None, **kwargs):
        if kwargs['vph'] == 3 and kwargs['camera'] == 'blue':
            kwargs['colour'] = 'green'
        else:
            kwargs['colour'] = kwargs['camera']
        super().__init__(tables, **kwargs)

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        config = progtemp_config.loc[progtemp_code[0]]
        red = cls(resolution=str(config.resolution), vph=int(config.red_vph), camera='red')
        blue = cls(resolution=str(config.resolution), vph=int(config.blue_vph), camera='blue')
        return red, blue


class ObsTemp(Hierarchy):
    factors = ['maxseeing', 'mintrans', 'minelev', 'minmoon', 'maxsky']

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors]
        obstemp_code = list(header['OBSTEMP'])
        return cls(**{n: v for v, n in zip(obstemp_code, names)})


class Survey(Hierarchy):
    idname = 'surveyname'


class WeaveTarget(Hierarchy):
    idname = 'cname'


class Fibre(Hierarchy):
    idname = 'fibreid'


class SubProgramme(Hierarchy):
    parents = [Multiple(Survey)]
    factors = ['targprog']  # unique only to survey groups


class SurveyCatalogue(Hierarchy):
    parents = [SubProgramme]
    factors = ['targcat']  # unique only to subprogrammes


class SurveyTarget(Hierarchy):
    parents = [SurveyCatalogue, WeaveTarget]
    # assume the worst case, that targids are not unique
    factors = ['targid', 'targname', 'targra', 'targdec', 'targepoch',
               'targpmra', 'targpmdec', 'targparal', 'mag_g', 'emag_g', 'mag_r', 'emag_r', 'mag_i', 'emag_i', 'mag_gg', 'emag_gg',
               'mag_bp', 'emag_bp', 'mag_rp', 'emag_rp']


class FibreTarget(Hierarchy):
    factors = ['fibrera', 'fibredec', 'status', 'xposition', 'yposition',
               'orientat',  'retries', 'targx', 'targy', 'targuse', 'targprio']
    parents = [Fibre, SurveyTarget]


class FibreSet(Hierarchy):
    # defined entirely by constituent fibretargets (unique only in this respect)
    parents = [Multiple(FibreTarget)]


class CFG(Hierarchy):
    parents = [ArmConfig]
    factors = ['mode', 'binning']


class InstrumentConfiguration(Hierarchy):
    factors = ['mode', 'binning']
    parents = [Multiple(ArmConfig, 2, 2)]


class ProgTemp(Hierarchy):
    parents = [InstrumentConfiguration]
    factors = ['length', 'exposure_code']

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        progtemp_code = progtemp_code.split('.')[0]
        progtemp_code_list = list(map(int, progtemp_code))
        configs = ArmConfig.from_progtemp_code(progtemp_code_list)
        mode = progtemp_config.loc[progtemp_code_list[0]]['mode']
        binning = progtemp_code_list[3]
        config = InstrumentConfiguration(armconfigs=configs, mode=mode, binning=binning)
        exposure_code = progtemp_code[2:4]
        length = progtemp_code_list[1]
        return cls(progtemp_code=progtemp_code, length=length, exposure_code=exposure_code,
                   instrumentconfiguration=config)

class OBSpec(Hierarchy):
    idname = 'xml'  # this is CAT-NAME in the header not CATNAME, annoyingly no hyphens allowed
    factors = ['obtitle']
    parents = [ObsTemp, FibreSet, ProgTemp]


class OB(Hierarchy):
    idname = 'obid'  # This is globally unique by obid
    factors = ['obstartmjd']
    parents = [OBSpec]


class Exposure(Hierarchy):
    idname = 'expmjd'  # globally unique
    parents = [OB]


class Run(Hierarchy):
    idname = 'runid'
    parents = [ArmConfig, Exposure]
    indexer = 'armconfig'


class Spectrum(Hierarchy):
    is_template = True
    products = ['flux', 'ivar', 'noss_flux', 'noss_ivar']


class RawSpectrum(Spectrum):
    parents = [Run, CASU, Simulator, System]
    products = Spectrum.products + ['guideinfo', 'metinfo']
    version_on = ['run']
    # any duplicates under a run will be versioned based on their appearance in the database
    # only one raw per run essentially


class L1SpectrumRow(Spectrum):
    is_template = True


class L1SingleSpectrum(L1SpectrumRow):
    parents = [RawSpectrum, FibreTarget, CASU]
    version_on = ['rawspectrum', 'fibretarget']
    factors = [
        'nspec', 'rms_arc1', 'rms_arc2', 'resol', 'helio_cor',
        'wave_cor1', 'wave_corrms1', 'wave_cor2', 'wave_corrms2',
        'skyline_off1', 'skyline_rms1', 'skyline_off2', 'skyline_rms2',
        'sky_shift', 'sky_scale', 'exptime', 'snr',
        'meanflux_g', 'meanflux_r', 'meanflux_i',
        'meanflux_gg', 'meanflux_bp', 'meanflux_rp'
               ]


class L1StackSpectrum(L1SpectrumRow):
    parents = [Multiple(L1SingleSpectrum, 2, constrain=[OB, ArmConfig, FibreTarget]), CASU]
    version_on = ['l1singlespectra']
    factors = ['exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']


class L1SuperStackSpectrum(L1SpectrumRow):
    parents = [Multiple(L1SingleSpectrum, 2, constrain=[OBSpec, ArmConfig, FibreTarget]), CASU]
    version_on = ['l1singlespectra']
    factors = ['exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']


class L1SuperTargetSpectrum(L1SpectrumRow):
    parents = [Multiple(L1SingleSpectrum, 2, constrain=[CFG, WeaveTarget]), CASU]
    version_on = ['l1singlespectra']
    factors = ['exptime', 'snr', 'meanflux_g', 'meanflux_r', 'meanflux_i',
               'meanflux_gg', 'meanflux_bp', 'meanflux_rp']


class L2(Hierarchy):
    is_template = True


class L2RowHDU(L2):
    is_template = True
    parents = [Multiple(L1SpectrumRow, 2, 3), APS]
    products = []
    version_on = ['l1spectrumrows']

#
# class Classifications(L2RowHDU):
#     pass
#
#
# class Star(L2RowHDU):
#     pass
#
#
# class Galaxy(L2RowHDU):
#     pass
#
#
# class L2Spectrum(L2RowHDU):
#     is_template = True
#
#
# class ClassificationModelSpectrum(L2Spectrum):
#     pass
#
#
# class StellarModelSpectrum(L2Spectrum):
#     pass
