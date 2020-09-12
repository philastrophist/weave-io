from pathlib import Path
from typing import Union, Any

import networkx as nx
from astropy.io import fits
from astropy.table import Table
from graphviz import Source

from .config_tables import progtemp_config
from .graph import Graph


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
        self.name = node.__name__.lower()+'s'
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


class Graphable:
    idname = None
    name = None
    identifier = None
    indexers = []
    type_graph_attrs = {}

    def __init__(self, **nodes):
        graph = Graph.get_context()  # type: Graph
        graph.add_node(self.__class__.__name__, peripheries=2, **self.type_graph_attrs)
        idname = self.idname
        if self.idname is not None:
            idname = self.idname.lower()
        graph.add_node(self.name, idname=idname, identifier=self.identifier, **self.type_graph_attrs)
        color = graph.nodes[self.name]['edgecolor']
        graph.add_edge(self.name, self.__class__.__name__, color=color, type='is_a', label='is_a')
        for k, node_list in nodes.items():
            for node in node_list:
                if k in [i.lower() for i in self.indexers]:
                    type = 'indexes'
                    label = type
                else:
                    type = 'is_required_by'
                    label = ''
                graph.add_edge(node.name, self.name, type=type, label=label,
                               color=graph.nodes[self.name]['edgecolor'])


class Factor(Graphable):
    type_graph_attrs = factor_attrs

    def __init__(self, name, value):
        self.idname = name
        self.identifier = value
        self.name = f"{self.idname}({self.identifier})"
        super(Factor, self).__init__()
        graph = Graph.get_context()  # type: Graph
        graph.add_node(self.idname, peripheries=2, **self.type_graph_attrs)
        graph.add_edge(self.idname, self.name)


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

    def __init__(self, **kwargs):
        parents = {p.__name__.lower() if isinstance(p, type) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        if self.idname is not None:
            self.identifier = kwargs.pop(self.idname.lower())
            setattr(self, self.idname, self.identifier)

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
        if self.idname is not None:
            self.name = f"{self.__class__.__name__}({self.idname.lower()}={self.identifier})"
        else:
            name = ''
            for predecessor_list in predecessors.values():
                for predecessor in predecessor_list:
                    name += predecessor.name
            name = '#' + str(hash(name))  # TODO: urgently needs replacing with a reliable hash
            self.identifier = name
            self.name = f"{self.__class__.__name__}({name})"
        super(Hierarchy, self).__init__(**predecessors)


class File(Graphable):
    idname = 'fname'
    constructed_from = []
    type_graph_attrs = l1file_attrs

    def __init__(self, fname: Union[Path, str]):
        self.fname = Path(fname)
        self.identifier = str(self.fname)
        self.name = f'{self.__class__.__name__}({self.fname})'
        self.predecessors = self.read()
        super(File, self).__init__(**self.predecessors)

    def match(self, directory: Path):
        raise NotImplementedError


    @property
    def graph_name(self):
        return f"File({self.fname})"

    def read(self):
        raise NotImplementedError


class ArmConfig(Hierarchy):
    factors = ['Resolution', 'VPH', 'Camera']

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        config = progtemp_config.loc[progtemp_code[0]]
        red = cls(resolution=config.resolution, vph=config.red_vph, camera='red')
        blue = cls(resolution=config.resolution, vph=config.blue_vph, camera='blue')
        return red, blue


class ObsTemp(Hierarchy):
    factors = ['MaxSeeing', 'MinTrans', 'MinElev', 'MinMoon', 'MaxSky']

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors]
        return cls(**{n: v for v, n in zip(list(header['OBSTEMP']), names)})


class Target(Hierarchy):
    idname = 'CNAME'
    factors = ['RA', 'DEC']

    @classmethod
    def from_fibinfo_row(cls, row):
        return Target(cname=row['CNAME'], ra=row['FIBRERA'], dec=row['FIBREDEC'])


class TargetSet(Hierarchy):
    parents = [Multiple(Target)]

    @classmethod
    def from_fibinfo(cls, fibinfo):
        targets = [Target.from_fibinfo_row(row) for row in fibinfo]
        return cls(targets=targets)


class ProgTemp(Hierarchy):
    factors = ['Mode', 'Binning']
    parents = [Multiple(ArmConfig, 2, 2)]

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        progtemp_code = list(map(int, progtemp_code))
        configs = ArmConfig.from_progtemp_code(progtemp_code)
        mode = progtemp_config.loc[progtemp_code[0]]['mode']
        binning = progtemp_code[3]
        return cls(mode=mode, binning=binning, armconfigs=configs)


class OBSpec(Hierarchy):
    factors = ['OBTitle']
    parents = [ObsTemp, TargetSet, ProgTemp]


class OBRealisation(Hierarchy):
    idname = 'OBID'
    factors = ['OBStartMJD']
    parents = [OBSpec]


class Exposure(Hierarchy):
    parents = [OBRealisation]
    factors = ['ExpMJD']


class Run(Hierarchy):
    idname = 'RunID'
    parents = [Exposure]
    factors = ['Camera']
    indexers = ['Camera']


class HeaderFibinfoFile(File):
    fibinfo_i = -1

    def read(self):
        header = fits.open(self.fname)[0].header
        runid = header['RUN']
        camera = header['CAMERA'].lower()[len('WEAVE'):]
        expmjd = header['MJD-OBS']
        res = header['VPH']
        obstart = header['OBSTART']
        obtitle = header['OBTITLE']
        obid = header['OBID']

        fibinfo = Table(fits.open(self.fname)[self.fibinfo_i].data)
        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = progtemp_config[(progtemp_config['mode'] == progtemp.mode)
                              & (progtemp_config['resolution'] == res)][f'{camera}_vph'].iloc[0]
        armconfig = ArmConfig(vph=vph, resolution=res, camera=camera)  # must instantiate even if not used
        obstemp = ObsTemp.from_header(header)
        targetset = TargetSet.from_fibinfo(fibinfo)
        obspec = OBSpec(targetset=targetset, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        obrealisation = OBRealisation(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, camera=camera, exposure=exposure)
        return {'run': [run]}


class Raw(HeaderFibinfoFile):
    parents = [Run]
    fibinfo_i = 3

    @classmethod
    def match(cls, directory: Path):
        return directory.glob('r*.fit')


class L1Single(HeaderFibinfoFile):
    parents = [Run]
    constructed_from = [Raw]

    @classmethod
    def match(cls, directory):
        return directory.glob('single_*.fit')


class L1Stack(HeaderFibinfoFile):
    parents = [OBRealisation]
    factors = ['VPH']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('stacked_*.fit')


class L1SuperStack(File):
    parents = [OBSpec]
    factors = ['VPH']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('superstacked_*.fit')


class L1SuperTarget(File):
    parents = [ArmConfig, Target]
    factors = ['Binning', 'Mode']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*.fit')


class L2Single(File):
    parents = [Exposure]
    constructed_from = [Multiple(L1Single, 2, 2)]

    @classmethod
    def match(cls, directory):
        return directory.glob('single_*_aps.fit')


class L2Stack(File):
    parents = [Multiple(ArmConfig, 1, 3), TargetSet]
    factors = ['Binning', 'Mode']
    constructed_from = [Multiple(L1Stack, 0, 3), Multiple(L1SuperStack, 0, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('(super)?stacked_*_aps.fit')


class L2SuperTarget(File):
    parents = [Multiple(ArmConfig, 1, 3), Target]
    factors = ['Mode', 'Binning']
    constructed_from = [Multiple(L1SuperTarget, 2, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*_aps.fit')
