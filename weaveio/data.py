from functools import reduce
from pathlib import Path
from typing import Union, Any, Dict, Set, List

import networkx as nx

from weaveio.graph import Graph
from weaveio.hierarchy import File, Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget, Hierarchy


class Address(dict):
    pass


class Data:
    filetypes = []

    def __init__(self, rootdir: Union[Path, str]):
        super(Data, self).__init__()
        self.graph = Graph()
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.read_in_directory()
        self.address = Address()
        self.hierarchies = {}
        self.factors = []

    def read_in_directory(self):
        for filetype in self.filetypes:
            self.filelists[filetype] = filetype.match(self.rootdir)
        with self.graph:
            for filetype, files in self.filelists.items():
                for file in files:
                    filetype(file).read()
            self.graph.remove_node('Factor')
        self.hierarchies = {h.__class__.__name__: h for ft in self.filetypes for h in ft.parents}
        self.factors = [f for ft in self.filetypes for h in ft.parents for f in h.factors]


    def _match_nodes(self, factor_type_name: str, factor_value: Any, nodetype: str) -> Set[str]:
        """
        Match files by building a list of shortest paths that have the subpath:
            (factor_name)->(factor_value)->...->(file)->(filetype)
        return [(file) from each valid path]
        """
        paths, lengths = zip(*[(p, len(p)) for p in nx.all_shortest_paths(self.graph, factor_type_name, nodetype)])
        node = f'{factor_type_name}({factor_value})'
        paths = [p for p in paths if len(p) == min(lengths) and node in p]
        files = {p[-2] for p in paths}
        return files

    def _query_nodes(self, nodetype: str, factors: Dict[str, Any]) -> Set[str]:
        """
        Match all valid files
        Do this for each given factor and take then intersection of the file set as the result
        """
        return reduce(lambda a, b: a & b, [self._match_nodes(k, v, nodetype) for k, v in factors.items()])

    def graph_view(self, include: Union[str, List[str]] = None,
                   exclude: Union[str, List[str]] = None) -> nx.Graph:
        """
        Returns a view (not a copy) into the data graph but hiding any nodes in `exclude` and any
        nodes that don't have a path to any node in `include`.
        :param include: Nodes that all resulting graph nodes should have a path to
        :param exclude: Nodes that should be included
        :return: networkx.Graph
        """
        if isinstance(include, str):
            include = [include]
        elif include is None:
            include = []
        if isinstance(exclude, str):
            exclude = [exclude]
        elif exclude is None:
            exclude = []
        view = self.graph
        if include:
            view = nx.subgraph_view(view, lambda n: any(nx.has_path(self.graph, n, i) for i in include))
        if exclude:
            view = nx.subgraph_view(view, lambda n: not any(n.startswith(i) for i in exclude))
        isolates = nx.isolates(view)
        view = nx.subgraph_view(view, lambda n: n not in isolates)
        return view

    def node_implies_plurality_of(self, start_node, implication_node):
        if implication_node in self.relation_graph.descendants(start_node):
            return True
        else:
            edges = nx.shortest_path(self.relation_graph.reverse(), start_node, implication_node)
            return any(self.relation_graph.edges[(n1, n2)]['multiplicity'] > 1 for n1, n2 in zip(edges[:-1], edges[1:]))

    def __getitem__(self, address):
        return HeterogeneousHierarchy(self, address)

    def __getattr__(self, item):
        return HeterogeneousHierarchy(self, Address()).__getattr__(item)


class Indexable:
    def __init__(self, data: Data, address: Address):
        self.data = data
        self.address = address

    def index_by_hierarchy_name(self, hierarchy_name):
        raise NotImplementedError

    def index_by_address(self, address):
        return HeterogeneousHierarchy(self.data, self.address & address)

    def plural_name(self, singular_name):
        if singular_name in self.data.factors:
            return singular_name+'s'
        elif singular_name in self.data.hierarchies:
            return self.data.hierarchies[singular_name].plural_name
        else:
            raise KeyError(f"{singular_name} is not a valid singular name")

    def singular_name(self, plural_name):
        if plural_name in self.data.factors:
            return plural_name[:-1]  # remove the 's'
        elif plural_name in self.data.hierarchies:
            return self.data.hierarchies[plural_name].__class__.__name__.lower()
        else:
            raise KeyError(f"{plural_name} is not a valid plural name")

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        try:
            return self.plural_name(name) == name
        except KeyError:
            return self.singular_name(name) == name

    def node_expected_plural(self, name):
        """
        Returns True if the current hierarchy object expects name to be plural by looking at the
        relation graph
        """
        raise NotImplementedError


class HeterogeneousHierarchy(Indexable):
    """
	.<other> - data[Address(vph='green')].vph
		- if <other> is in the address, returns other
		- else raise IndexError
	.<other>s - data.OBs
		- returns HomogeneousStore
	[Address()] - data[Address(vph='green')]
		- Returns HeterogeneousStore filtered by the combined address
	[key]
		- Not implemented
    """
    def __getitem__(self, address):
        if isinstance(address, Address):
            return self.index_by_address(address)
        else:
            raise NotImplementedError("Cannot index by an id over multiple heterogeneous hierarchies")

    def node_expected_plural(self, name):
        return True

    def index_by_hierarchy_name(self, hierarchy_name):
        try:
            self.address[hierarchy_name]
        except KeyError:
            if not self.is_plural_name(hierarchy_name):
                raise NotImplementedError(f"Can only index plural hierarchies in a heterogeneous address")
            return HomogeneousHierarchy(self.data, self.address, hierarchy_name)

    def __getattr__(self, item):
        return self.index_by_hierarchy_name(item)


class HomogeneousHierarchy(Indexable):
    """
	.<other> - OBs.OBspec
		- returns Hierarchy/factor/id/file
	.<other>s
		- return HomogeneousStore (e.g. ob.l1.singles)
	[Address()]
		- return Hierarchy if address describes a unique object
		- return HomogeneousStore if address contains some missing factors
		- return HomogeneousStore  if not
		- raise IndexError if address is incompatible
	[key] - OBs[obid]
		- if indexable by key, return Hierarchy
    """
    def __init__(self, data: Data, address: Address, nodetype: str):
        super().__init__(data, address)
        self.nodetype = nodetype

    def node_expected_plural(self, name):
        return self.data.node_implies_plurality_of(self.nodetype, name)

    def index_by_id(self, id_value):
        nodename = f'f{self.nodetype.capitalize()}({id_value})'
        return  self.nodetype(id_value)

    def __getitem__(self, item):
        if isinstance(item, Address):
            return self.index_by_address(item)
        else:
            return self.index_by_id(item)



class OurData(Data):
    filetypes = [Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget]
