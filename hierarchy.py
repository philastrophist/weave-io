from pathlib import Path
from typing import Union

from astropy.io import fits
from astropy.table import Table

from hierarchy_network import graph2pdf
from config_tables import progtemp_config


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

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}])>"


class Factor:
    type_graph_attrs = factor_attrs

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"<Factor({self.name}={self.value})>"

    @property
    def graph_name(self):
        return f"{self.name}({self.value})"


class Hierarchy:
    idname = None
    parents = []
    factors = []
    type_graph_attrs = hierarchy_attrs

    @property
    def graph_name(self):
        if self.idname is not None:
            return f"{self.__class__.__name__}({self.idname}={self.identifier})"
        else:
            d = {}
            for thing, _ in self.traverse_edges():
                if isinstance(thing, Factor):
                    d[thing.name] = thing.value
                elif thing.idname is not None:
                    d[thing.name+thing.idname] = thing.identifier
            return f'{self.__class__.__name__}('+'-'.join([f'{k}_{v}' for k,v in d.items()]) + ')'

    def __repr__(self):
        try:
            s = f'({self.idname}={self.identifier})'
        except AttributeError:
            s = ''
        return f"<{self.__class__.__name__}{s}>"

    def traverse_edges(self):
        for name, thing_list in self.predecessors.items():
            for thing in thing_list:
                if isinstance(thing, Factor):
                    yield (thing, self)
                else:
                    for edge in thing.traverse_edges():
                        yield edge
                    yield (thing, self)

    def __init__(self, **kwargs):
        self.predecessors = {}
        self.name = self.__class__.__name__
        parents = {p.__name__.lower() if isinstance(p, type) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        if self.idname is not None:
            self.identifier = kwargs.pop(self.idname.lower())
            setattr(self, self.idname, self.identifier)
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
            self.predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")


class File(Hierarchy):
    idname = 'fname'
    constructed_from = []
    type_graph_attrs = l1file_attrs

    def __init__(self, fname: Union[Path, str]):
        self.name = self.__class__.__name__
        self.predecessors = {}
        self.fname = Path(fname)

    @property
    def graph_name(self):
        return f"File({self.fname})"

    def read(self):
        raise NotImplementedError

    def read_hierarchy(self):
        hierarchies = self.read()
        for hierarchy in hierarchies:
            assert isinstance(hierarchy, Hierarchy)
            self.predecessors[hierarchy.name] = [hierarchy]
        return self


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
    factors = ['VPH']


class Raw(File):
    parents = [Run]

    def read(self):
        header = fits.open(self.fname)[0].header
        runid = header['RUN']
        camera = header['CAMERA'].lower()[len('WEAVE'):]
        expmjd = header['MJD-OBS']
        res = header['VPH']
        obstart = header['OBSTART']
        obtitle = header['OBTITLE']
        obid = header['OBID']

        fibinfo = Table(fits.open(self.fname)[3].data)
        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = progtemp_config[(progtemp_config['mode'] == progtemp.mode)
                              & (progtemp_config['resolution'] == res)][f'{camera}_vph'].iloc[0]
        armconfig = ArmConfig(vph=vph, resolution=res, camera=camera)
        obstemp = ObsTemp.from_header(header)
        targetset = TargetSet.from_fibinfo(fibinfo)
        obspec = OBSpec(targetset=targetset, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        obrealisation = OBRealisation(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, vph=vph, exposure=exposure)
        return armconfig, exposure, run


class L1Single(File):
    parents = [Run]
    constructed_from = [Raw]


class L1Stack(File):
    parents = [OBRealisation]
    factors = ['VPH']
    constructed_from = [L1Single]


class L1SuperStack(File):
    parents = [OBSpec]
    factors = ['VPH']
    constructed_from = [L1Single]


class L1SuperTarget(File):
    parents = [ArmConfig, Target]
    factors = ['Binning', 'Mode']
    constructed_from = [L1Single]


class L2Single(File):
    parents = [Exposure]
    constructed_from = [Multiple(L1Single, 2, 2)]


class L2Stack(File):
    parents = [Multiple(ArmConfig, 1, 3), TargetSet]
    factors = ['Binning', 'Mode']
    constructed_from = [Multiple(L1Stack, 0, 3), Multiple(L1SuperStack, 0, 3)]


class L2SuperTarget(File):
    parents = [Multiple(ArmConfig, 1, 3), Target]
    factors = ['Mode', 'Binning']
    constructed_from = [Multiple(L1SuperTarget, 2, 3)]


def add_hierarchies(graph, hierarchies: Hierarchy):
    for from_thing, to_thing in hierarchies.traverse_edges():
        graph.add_node(to_thing.__class__.__name__, **to_thing.type_graph_attrs)
        graph.add_node(from_thing.graph_name, **from_thing.type_graph_attrs)
        graph.add_node(to_thing.graph_name, **to_thing.type_graph_attrs)
        graph.add_node(from_thing.__class__.__name__, **from_thing.type_graph_attrs)
        graph.add_edge(from_thing.__class__.__name__, from_thing.graph_name, color=graph.nodes[from_thing.graph_name]['edgecolor'])
        graph.add_edge(to_thing.__class__.__name__, to_thing.graph_name, color=graph.nodes[to_thing.graph_name]['edgecolor'])
        graph.add_edge(from_thing.graph_name, to_thing.graph_name, color=graph.nodes[to_thing.graph_name]['edgecolor'])

if __name__ == '__main__':
    import networkx as nx
    instance_graph = nx.DiGraph()
    raw = Raw('r1002813.fit').read_hierarchy()
    add_hierarchies(instance_graph, raw.predecessors['ArmConfig'][0])
    add_hierarchies(instance_graph, raw.predecessors['Exposure'][0])
    invisible = ['Target'] + [f.lower() for f in Target.factors]
    view = nx.subgraph_view(instance_graph, lambda n: not any(i in n for i in invisible))
    graph2pdf(view, 'instance_graph')
