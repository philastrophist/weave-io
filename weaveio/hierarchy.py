import inspect

import networkx as nx
import xxhash
from graphviz import Source
from tqdm import tqdm

from .config_tables import progtemp_config
from .graph import Graph, Node, Relationship, ContextError


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


class PluralityMeta(type):
    def __new__(meta, name, bases, dct):
        if dct.get('plural_name', None) is None:
            dct['plural_name'] = name.lower() + 's'
        dct['singular_name'] = name.lower()
        dct['plural_name'] = dct['plural_name'].lower()
        dct['singular_name'] = dct['singular_name'].lower()
        r = super(PluralityMeta, meta).__new__(meta, name, bases, dct)
        return r


class Graphable(metaclass=PluralityMeta):
    idname = None
    name = None
    identifier = None
    indexers = []
    type_graph_attrs = {}
    plural_name = None
    singular_name = None
    parents = []

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
        raise NotImplementedError("Data not added to hierarchy object")

    @property
    def neotypes(self):
        clses = [i.__name__ for i in inspect.getmro(self.__class__)]
        clses = clses[:clses.index('Graphable')]
        return clses

    @property
    def neoproperties(self):
        if self.idname is not None:
            d = {self.idname: self.identifier}
            d['id'] = self.identifier
            return d
        else:
            return {'dummy': 1}   # just to stop py2neo complaining, shouldnt actually be encountered

    def __init__(self, **predecessors):
        self.data = None
        try:
            graph = Graph.get_context()
            graph.statement
            graph.tx.evaluate(self.statement, rows=params)


            self.node = Node(*self.neotypes, **self.neoproperties)
            try:
                key = list(self.neoproperties.keys())[0]
            except IndexError:
                key = None
            primary = {'primary_label': self.neotypes[-1], 'primary_key': key}
            tx.merge(self.node, **primary)
            for k, node_list in predecessors.items():
                for inode, node in enumerate(node_list):
                    if k in [i.lower() for i in self.indexers]:
                        type = 'indexes'
                    else:
                        type = 'is_required_by'
                    tx.merge(Relationship(node.node, type, self.node, order=inode))
        except ContextError:
            pass
        self.predecessors = predecessors


class Factor(Graphable):
    type_graph_attrs = factor_attrs

    @property
    def neotypes(self):
        return ['Factor', self.idname]

    @property
    def neoproperties(self):
        return {'value': self.identifier, 'id': self.identifier}

    def __init__(self, name, value, plural_name=None):
        self.idname = name
        self.identifier = value
        self.name = f"{self.idname}({self.identifier})"
        super(Factor, self).__init__()

    def __repr__(self):
        return f"<Factor({self.idname}={self.identifier})>"


class Hierarchy(Graphable):
    idname = None
    parents = []
    factors = []
    indexers = []
    type_graph_attrs = hierarchy_attrs

    def __repr__(self):
        return self.name

    def generate_identifier(self):
        raise NotImplementedError

    def __init__(self, **kwargs):
        if self.idname is None:
            self.idname = 'id'
        if self.idname not in kwargs:
            self.identifier = None
        else:
            self.identifier = kwargs.pop(self.idname)

        parents = {p.__name__.lower() if isinstance(p, type) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        self._kwargs = kwargs.copy()

        predecessors = {}
        for name, nodetype in specification.items():
            value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple):
                if not isinstance(value, (tuple, list)):
                    raise TypeError(f"{name} expects multiple elements")
                if name in factors:
                    value = [Factor(name, val) for val in value]
            elif name in factors:
                value = [Factor(name, value)]
            else:
                value = [value]
            predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")
        self.predecessors = predecessors
        if self.identifier is None:
            self.identifier = self.generate_identifier()
        setattr(self, self.idname, self.identifier)
        self.name = f"{self.__class__.__name__}({self.idname}={self.identifier})"
        super(Hierarchy, self).__init__(**predecessors)


class ArmConfig(Hierarchy):
    factors = ['Resolution', 'VPH', 'Camera']
    idname = 'armcode'

    def generate_identifier(self):
        return f'{self.resolution}{self.vph}{self.camera}'

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        config = progtemp_config.loc[progtemp_code[0]]
        red = cls(resolution=str(config.resolution), vph=int(config.red_vph), camera='red')
        blue = cls(resolution=str(config.resolution), vph=int(config.blue_vph), camera='blue')
        return red, blue


class ObsTemp(Hierarchy):
    factors = ['MaxSeeing', 'MinTrans', 'MinElev', 'MinMoon', 'MaxSky']
    idname = 'obstemp'

    def generate_identifier(self):
        return f'{self.maxseeing}{self.mintrans}{self.minelev}{self.minmoon}{self.maxsky}'

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors]
        return cls(**{n: v for v, n in zip(list(header['OBSTEMP']), names)})


class Target(Hierarchy):
    idname = 'cname'

    def generate_identifier(self):
        return f"{self.cname}"

    @classmethod
    def from_fibinfo_row(cls, row):
        return Target(cname=row['CNAME'])


class TargetSet(Hierarchy):
    parents = [Multiple(Target)]

    def generate_identifier(self):
        h = xxhash.xxh64()
        for i, t in enumerate(self.targets):
            h.update(f"{i}{t.cname}")
        return '#' + h.hexdigest()

    @classmethod
    def from_fibinfo(cls, fibinfo):
        targets = [Target.from_fibinfo_row(row) for row in tqdm(fibinfo)]
        return cls(targets=targets)


class ProgTemp(Hierarchy):
    factors = ['Mode', 'Binning']
    parents = [Multiple(ArmConfig, 2, 2)]

    def generate_identifier(self):
        return f'{self.mode}{self.binning}{"".join(a.identifier for a in self.armconfigs)}'

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        progtemp_code = progtemp_code.split('.')[0]
        progtemp_code = list(map(int, progtemp_code))
        configs = ArmConfig.from_progtemp_code(progtemp_code)
        mode = progtemp_config.loc[progtemp_code[0]]['mode']
        binning = progtemp_code[3]
        return cls(mode=mode, binning=binning, armconfigs=configs)


class OBSpec(Hierarchy):
    factors = ['OBTitle']
    parents = [ObsTemp, TargetSet, ProgTemp]

    def generate_identifier(self):
        return f"{self.obstemp.identifier}-{self.progtemp.identifier}-{self.targetset.identifier}"


class OBRealisation(Hierarchy):
    idname = 'obid'
    factors = ['OBStartMJD']
    parents = [OBSpec]


class Exposure(Hierarchy):
    parents = [OBRealisation]
    factors = ['ExpMJD']

    def generate_identifier(self):
        return f"{self.obrealisation.identifier}-{self.expmjd}"


class Run(Hierarchy):
    idname = 'runid'
    parents = [Exposure]
    factors = ['Camera']
    indexers = ['Camera']
