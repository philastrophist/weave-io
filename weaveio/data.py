from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Union, Any, Dict, Set, List

import networkx as nx

from weaveio.graph import Graph
from weaveio.hierarchy import File, Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget, Hierarchy


class Address(dict):
    pass


def quote(x):
    """
    Return a quoted string if x is a string otherwise return x
    """
    if isinstance(x, str):
        return f"'{x}'"
    return x


class Data:
    filetypes = []

    def __init__(self, rootdir: Union[Path, str], host: str = 'host.docker.internal'):
        self.graph = Graph(host=host)
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.read_in_directory()
        self.address = Address()
        self.hierarchies = {}
        self.factors = []

    def read_in_directory(self):
        for filetype in self.filetypes:
            self.filelists[filetype] = list(filetype.match(self.rootdir))
        with self.graph:
            for filetype, files in self.filelists.items():
                for file in files:
                    tx = self.graph.begin()
                    filetype(file)
                    tx.commit()
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


class BasicQuery:
    def __init__(self, matches: List = None, wheres: List = None, current_varname: str = None):
        self.matches = []
        self.wheres = []
        self.matches += [] if matches is None else matches
        self.wheres += [] if wheres is None else wheres
        self.current_varname = None
        self.current_label = None
        self.counter = defaultdict(int)

    def make(self, branch=False):
        if self.matches:
            match = 'MATCH {}'.format('\nMATCH '.join(self.matches))
        else:
            match = ""
        if self.wheres:
            where = 'WHERE {}'.format('\nAND'.join(self.wheres))
        else:
            where = ""
        if branch:
            returns = '\n'.join([f"MATCH p1=({self.current_varname})<-[*]-(n1)",
                                 f"MATCH p2=({self.current_varname})-[*]->(n2)",
                                 f"WITH collect(p2) as p2s, collect(p1) as p1s",
                                 f"CALL apoc.convert.toTree(p1s+p2s) yield value",
                                 f"RETURN value"])
            returns = r'//Add Hierarchy Branch'+'\n' + returns
        else:
            returns = f"RETURN DISTINCT {self.current_varname}.id as id"
        return f"{match}\n{where}\n{returns}"

    def index_by_address(self, address):
        if self.current_varname is None:
            name = 'first'
        else:
            name = self.current_varname
        self.matches += [f"(:{k} {{value: {quote(v)}}})-[*]->({name})" for k, v in address.items()]
        return self

    def index_by_hierarchy_name(self, hierarchy_name, direction=None):
        name = '{}{}'.format(hierarchy_name.lower(), self.counter[hierarchy_name])
        if self.current_varname is None:
            first_encountered = False
            for i, m in enumerate(self.matches):
                if '(first)' in m and not first_encountered:
                    self.matches[i] = m.replace('(first)', f'({name}:{hierarchy_name})')
                    first_encountered = True
                else:
                    self.matches[i] = m.replace('(first)', f'({name})')
            self.current_varname = name
        else:
            arrows = '<-[*]-' if direction == 'above' else '-[*]->'
            self.matches += [f"({self.current_varname}){arrows}({name}:{hierarchy_name})"]
            self.current_varname = name
        self.current_label = hierarchy_name
        self.counter[hierarchy_name] += 1
        return self

    def index_by_id(self, id_value):
        self.wheres += [f"{self.current_varname}.id = {quote(id_value)}"]
        return self


class Indexable:
    def __init__(self, query, address: Address):
        self.query = query
        self.address = address

    def __call__(self):
        raise NotImplementedError

    def index_by_single_hierarchy_name(self, hierarchy_name):
        raise NotImplementedError

    def index_by_address(self, address):
        matches = [f"(:{k} {{value: {quote(v)}}})-[*]->(h)" for k, v in self.address.items()]
        query = HierarchyQuery(self.query, matches)
        return HeterogeneousHierarchy(query, self.address & address)

    def index_by_plural_hierarchy_name(self, hierarchy_name):
        raise NotImplementedError

    def index_by_key(self, key):
        raise NotImplementedError


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

    def __init__(self, data: Data, address: Address):
        super().__init__(data, address)
        self.matches += [f"(:{k} {{value: {quote(v)}}})-[*]->(h)" for k, v in self.address.items()]

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
    def __init__(self, data: Data, address: Address, nodetype: type):
        super().__init__(data, address)
        self.nodetype = nodetype
        matches, wheres, self.neo_varname = nodetype.neo_query('h')
        self.matches += matches
        self.wheres += wheres


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
