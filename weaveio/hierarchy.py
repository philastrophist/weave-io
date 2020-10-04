import inspect
from typing import Tuple, Dict, Type

import networkx as nx
import xxhash
from graphviz import Source
from tqdm import tqdm

from .config_tables import progtemp_config
from .graph import Graph, Node, Relationship, ContextError, Unwind
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


class RuleBreakingException(Exception):
    pass


class Multiple:
    def __init__(self, node, minnumber=1, maxnumber=None):
        self.node = node
        self.minnumber = minnumber
        self.maxnumber = maxnumber
        self.name = node.plural_name
        self.singular_name = node.singular_name
        self.plural_name = node.plural_name
        try:
            self.factors =  self.node.factors
        except AttributeError:
            self.factors = []
        try:
            self.parents = self.node.parents
        except AttributeError:
            self.parents = []

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}])>"


class GraphableMeta(type):
    def __new__(meta, name: str, bases, dct):
        if dct.get('plural_name', None) is None:
            dct['plural_name'] = name.lower() + 's'
        dct['singular_name'] = name.lower()
        dct['plural_name'] = dct['plural_name'].lower()
        dct['singular_name'] = dct['singular_name'].lower()
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
    indexers = []
    type_graph_attrs = {}
    plural_name = None
    singular_name = None
    parents = []
    uses_tables = False
    factors = []

    def add_parent_data(self, data):
        self.data = data

    def __getattr__(self, item):
        if self.data is not None:
            if self.data.is_plural_idname(item):
                is_plural = True
                name = self.data.singular_idnames[item[:-1]].singular_name
                final = [item[:-1]]
            elif self.data.is_singular_idname(item):
                is_plural = False
                name = self.data.singular_idnames[item].singular_name
                final = [item]
            elif self.data.is_plural_name(item):
                is_plural = True
                name = self.data.singular_name(item)
                final = []
            elif self.data.is_singular_name(item):
                is_plural = False
                name = item
                final = []
            else:
                raise AttributeError(f"{self} has no attribute {item}")
            should_be_plural, _ = self.data.node_implies_plurality_of(self.singular_name, name)
            plural_name = self.data.plural_name(name)
            if should_be_plural and not is_plural:
                raise AttributeError(f"{self} has no attribute '{item}' but it does have the plural '{plural_name}'")
            elif not should_be_plural and is_plural:
                raise AttributeError(f"{self} has no plural attribute '{item}' but it does have the singular '{name}'")
            path = self.data.traversal_path(self.singular_name, name) + final
            current = self
            try:
                for attribute in path:
                    if isinstance(current, (list, tuple)):
                        current = [getattr(c, attribute) for c in current]
                    else:
                        current = getattr(current, attribute)
                return current
            except AttributeError as e:
                raise AttributeError(f"{item} cannot be found in {self} or its parent structure.") from e
        raise AttributeError(f"Data not added to {self}, cannot search for {self}.{item}")

    @property
    def neotypes(self):
        clses = [i.__name__ for i in inspect.getmro(self.__class__)]
        clses = clses[:clses.index('Graphable')]
        return clses[::-1]

    @property
    def neoproperties(self):
        if self.identifier is None:
            d = {self.idname: 'None'}
            d['id'] = 'None'
        else:
            d = {self.idname: self.identifier}
            d['id'] = self.identifier
        for f in self.factors:
                d[f.lower()] = getattr(self, f.lower())
        return d

    def __init__(self, **predecessors):
        self.predecessors = predecessors
        self.data = None
        try:
            graph = Graph.get_context()  # type: Graph
            self.node = graph.add_node(*self.neotypes, **self.neoproperties)
            for k, node_list in predecessors.items():
                if k in [i.lower() for i in self.indexers]:
                    type = 'indexes'
                else:
                    type = 'is_required_by'
                if isinstance(node_list, (list, tuple)):
                    for inode, node in enumerate(node_list):
                        graph.add_relationship(node.node, self.node, type, order=inode)
                else:
                    order = None
                    for v in node_list.neoproperties.values():
                        if isinstance(v, Unwind):
                            order = f"{v.name}._input_index"
                    graph.add_relationship(node_list.node, self.node, type,
                                           order=Varname(order))
        except ContextError:
            pass


class Hierarchy(Graphable):
    idname = None
    identifier_builder = []
    parents = []
    factors = []
    indexers = []
    type_graph_attrs = hierarchy_attrs

    def __repr__(self):
        return self.name

    def generate_identifier(self):
        """
        if `idname` is set, then return the identifier set at instantiation
        otherwise, make an identifier by stitching together identifiers of its input
        """
        if self.identifier is not None:
            return self.identifier
        elif len(self.identifier_builder):
            strings = []
            for i in self.identifier_builder:
                obj = getattr(self, i)
                if isinstance(obj, Hierarchy):
                    obj = obj.identifier
                if isinstance(obj, Unwind):
                    s = str(obj)
                    strings += ['+', s, '+']
                else:
                    s = str(obj)
                    strings.append(f"'{s}'")
            final =  ''.join(strings).strip('+').replace('++', '+').replace("''", "")
            return Varname(final)
        else:
            return None

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
        if tables is None:
            for value in kwargs.values():
                if getattr(value, 'uses_tables', False) or isinstance(value, Unwind):
                    self.uses_tables = True
        else:
            self.uses_tables = True
        if self.idname not in kwargs:
            self.identifier = None
        else:
            self.identifier = kwargs.pop(self.idname)
        specification, factors = self.make_specification()
        # add any data held in a neo4j unwind table
        for k, v in specification.items():
            if k not in kwargs:
                if tables is not None:
                    if k in tables:
                        kwargs[k] = tables[k]
        self._kwargs = kwargs.copy()
        # Make predecessors a dict of {name: [instances of required Factor/Hierarchy]}
        predecessors = {}
        for name, nodetype in specification.items():
            value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple):
                if not getattr(value, 'uses_tables', False) and not isinstance(value, (tuple, list)):
                    raise TypeError(f"{name} expects multiple elements")
            else:
                value = [value]
            if name not in factors:
                predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")
        self.predecessors = predecessors
        if self.identifier is None:
            self.identifier = self.generate_identifier()
        if self.idname is None:
            self.idname = 'id'
        setattr(self, self.idname, self.identifier)
        self.name = f"{self.__class__.__name__}({self.idname}={self.identifier})"
        super(Hierarchy, self).__init__(**predecessors)


class ArmConfig(Hierarchy):
    factors = ['resolution', 'vph', 'camera']
    idname = 'armcode'
    identifier_builder = ['resolution', 'vph', 'camera']

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        config = progtemp_config.loc[progtemp_code[0]]
        red = cls(resolution=str(config.resolution), vph=int(config.red_vph), camera='red')
        blue = cls(resolution=str(config.resolution), vph=int(config.blue_vph), camera='blue')
        return red, blue


class ObsTemp(Hierarchy):
    factors = ['maxseeing', 'mintrans', 'minelev', 'minmoon', 'maxsky']
    idname = 'obstemp_code'

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors]
        obstemp_code = list(header['OBSTEMP'])
        return cls(obstemp_code=''.join(obstemp_code), **{n: v for v, n in zip(obstemp_code, names)})


class Survey(Hierarchy):
    idname = 'surveyname'


class Catalogue(Hierarchy):
    pass


class WeaveInputCatalogue(Hierarchy):
    pass


class Target(Hierarchy):
    idname = 'cname'
    factors = ['targid', 'targname', 'targid', 'targra', 'targdec', 'targepoch', 'targcat',
               'targpmra', 'targpmdec', 'targparal', 'targuse', 'targprog',  'targprog',
               'targprio', 'mag_g', 'emag_g', 'mag_r', 'emag_r', 'mag_i', 'emag_i', 'mag_gg',
               'emag_gg', 'mag_bp', 'emag_bp', 'mag_rp', 'emag_rp', 'fpvignet', 'simvel']
    parents = [Multiple(Survey)]


class Fibre(Hierarchy):
    idname = 'fibreid'


class FibreAssignment(Hierarchy):
    factors = ['fibrera', 'fibredec', 'status', 'xposition', 'yposition', 'orientat', 'retries',]
    parents = [Fibre, Target]
    identifier_builder = ['fibre', 'target', 'xposition', 'yposition']


class FibreSet(Hierarchy):
    parents = [Multiple(FibreAssignment)]
    idname = 'id'


class ProgTemp(Hierarchy):
    factors = ['mode', 'binning']
    parents = [Multiple(ArmConfig, 2, 2)]
    idname = 'progtemp_code'

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        progtemp_code = progtemp_code.split('.')[0]
        progtemp_code_list = list(map(int, progtemp_code))
        configs = ArmConfig.from_progtemp_code(progtemp_code_list)
        mode = progtemp_config.loc[progtemp_code_list[0]]['mode']
        binning = progtemp_code_list[3]
        return cls(progtemp_code=progtemp_code, mode=mode, binning=binning, armconfigs=configs)


class OBSpec(Hierarchy):
    idname = 'catname'
    factors = ['obtitle']
    parents = [ObsTemp, FibreSet, ProgTemp]


class OBRealisation(Hierarchy):
    idname = 'obid'
    factors = ['obstartmjd']
    parents = [OBSpec]


class Exposure(Hierarchy):
    parents = [OBRealisation]
    factors = ['expmjd']
    identifier_builder = ['expmjd']


class Run(Hierarchy):
    idname = 'runid'
    parents = [Exposure]
    factors = ['camera']
    indexers = ['camera']
