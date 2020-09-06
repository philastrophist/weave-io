from pathlib import Path
from typing import Union

from astropy.io import fits
from astropy.table import Table

from hierarchy_network import graph2pdf


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


class Hierarchy:
    idname = None
    parents = []
    factors = []
    predecessors = {}
    type_graph_attrs = hierarchy_attrs

    def traverse_edges(self):
        for name, thing in self.predecessors.items():
            if not isinstance(thing, Hierarchy):
                yield (thing, self)
            else:
                for edge in thing.traverse_edges():
                    yield edge
                yield (thing, self)

    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        parents = {p.__name__.lower() if isinstance(p, type) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        inputs = parents.copy()
        inputs.update(factors)
        if self.idname is not None:
            self.identifier = kwargs.pop(self.idname.lower())
            setattr(self, self.idname, self.identifier)
        for k, v in inputs.items():
            if isinstance(v, Multiple):
                if not isinstance(kwargs[k], list):
                    raise TypeError(f"{k} expects multiple elements")
            try:
                value = kwargs.pop(k)
                setattr(self, k, value)
                self.predecessors[k] = value
            except KeyError:
                pass
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")


class File(Hierarchy):
    idname = 'fname'
    constructed_from = []
    type_graph_attrs = l1file_attrs

    def __init__(self, fname: Union[Path, str]):
        self.fname = Path(fname)

    def read(self):
        raise NotImplementedError

    def read_hierarchy(self):
        things = self.read()
        for thing in things:
            if isinstance(thing, dict):
                for k, v in thing.items():
                    self.predecessors[k] = v
            else:
                self.predecessors[thing.name] = thing
        return self


class ArmConfig(Hierarchy):
    factors = ['VPH', 'Camera']


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
    parents = [Multiple(ArmConfig, 1, 2)]


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
        fibinfo = Table(fits.open(self.fname)[3].data)
        runid = header['RUN']
        vph = header['VPH']
        camera = header['CAMERA']
        armconfig = ArmConfig(vph=vph, camera=camera)
        expmjd = header['MJD-OBS']
        vph = header['VPH']
        obstart = header['OBSTART']
        obtitle = header['OBTITLE']
        obstemp = ObsTemp.from_header(header)
        obid = header['OBID']
        targetset = TargetSet.from_fibinfo(fibinfo)
        obspec = OBSpec(targetset=targetset, obtitle=obtitle, obstemp=obstemp)
        obrealisation = OBRealisation(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, vph=vph)
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


if __name__ == '__main__':
    import networkx as nx
    instance_graph = nx.DiGraph()

    def add_hierarchies(graph, hierarchies: Hierarchy):
        for from_thing, to_thing in hierarchies.traverse_edges():
            if not isinstance(from_thing, Hierarchy):
                from_thing_graph_attrs = factor_attrs
                from_thing_name = from_thing.__name__.lower()
            else:
                from_thing_graph_attrs = from_thing.type_graph_attrs
                from_thing_name = from_thing.name.lower()
            if not isinstance(to_thing, Hierarchy):
                to_thing_name = to_thing.__name__.lower()
            else:
                to_thing_name = to_thing.name.lower()
            graph.add_node(from_thing.__class__.__name__, **from_thing_graph_attrs)
            graph.add_node(to_thing.__class__.__name__, **to_thing.type_graph_attrs)
            graph.add_node(from_thing.name, **from_thing_graph_attrs)
            graph.add_node(to_thing.name, **to_thing.type_graph_attrs)
            graph.add_edge(from_thing.__class__.__name__, from_thing, color=graph.nodes[from_thing]['edgecolor'])
            graph.add_edge(to_thing.__class__.__name__, to_thing, color=graph.nodes[to_thing]['edgecolor'])
            graph.add_edge(from_thing, to_thing, color=graph.nodes[to_thing]['edgecolor'])

    raw = Raw('r1002813.fit').read_hierarchy()
    add_hierarchies(instance_graph, raw)
    graph2pdf(instance_graph, 'instance_graph')
