from collections import defaultdict
from copy import deepcopy
from functools import reduce
from pathlib import Path
from textwrap import dedent
from typing import Union, Any, Dict, Set, List, Type, TypeVar

import networkx as nx

from weaveio.graph import Graph
from weaveio.hierarchy import File, Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget, Hierarchy, Multiple, Graphable


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

    def __init__(self, rootdir: Union[Path, str], host: str = 'host.docker.internal', port=11002):
        self.graph = Graph(host=host, port=port)
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.address = Address()
        self.hierarchies = {h.node if isinstance(h, Multiple) else h for ft in self.filetypes for h in ft.parents}
        self.hierarchies |= set(self.filetypes)
        self.singular_hierarchies = {h.singular_name: h for h in self.hierarchies}
        self.plural_hierarchies = {h.plural_name: h for h in self.hierarchies if h.plural_name != 'graphables'}
        self.factors = {f for ft in self.filetypes for h in ft.parents for f in h.factors}
        self.plural_factors =  {f.lower() + 's': f for f in self.factors}
        self.singular_factors = {f.lower() : f for f in self.factors}
        self.make_relation_graph()

    def make_relation_graph(self):
        self.relation_graph = nx.DiGraph()
        d = list(self.singular_hierarchies.values())
        while len(d):
            h = d.pop()
            if isinstance(h, Multiple):
                multiplicity = True
            else:
                multiplicity = False
            self.relation_graph.add_node(h.singular_name)
            for child in h.parents:
                self.relation_graph.add_node(child.singular_name)
                self.relation_graph.add_edge(child.singular_name, h.singular_name, multiplicity=multiplicity)
                d.append(child)
            try:
                for f in h.factors:
                    self.relation_graph.add_node(f.lower())
                    self.relation_graph.add_edge(f.lower(), h.singular_name, multiplicity=False)
            except AttributeError:
                pass

    def directory_to_neo4j(self):
        for filetype in self.filetypes:
            self.filelists[filetype] = list(filetype.match(self.rootdir))
        with self.graph:
            for filetype, files in self.filelists.items():
                for file in files:
                    tx = self.graph.begin()
                    filetype(file)
                    tx.commit()

    def node_implies_plurality_of(self, start_node, implication_node):
        if nx.has_path(self.relation_graph, start_node, implication_node):
            return True, 'below'
        else:
            edges = nx.shortest_path(self.relation_graph, implication_node, start_node)
            return any(self.relation_graph.edges[(n1, n2)]['multiplicity'] > 1
                       for n1, n2 in zip(edges[:-1], edges[1:])), 'above'

    def __getitem__(self, address):
        return HeterogeneousHierarchy(self, BasicQuery()).__getitem__(address)

    def __getattr__(self, item):
        return HeterogeneousHierarchy(self, BasicQuery()).__getattr__(item)


class BasicQuery:
    def __init__(self, blocks: List = None, current_varname: str = None,
                 current_label: str = None, counter: defaultdict = None):
        self.blocks = [] if blocks is None else blocks
        self.current_varname = current_varname
        self.current_label = current_label
        self.counter = defaultdict(int) if counter is None else counter

    def spawn(self, blocks, current_varname, current_label) -> 'BasicQuery':
        return BasicQuery(blocks, current_varname, current_label, self.counter)

    def make(self, branch=False):
        if self.blocks:
            match = '\n\n'.join(['\n'.join(blocks) for blocks in self.blocks])
        else:
            raise ValueError(f"Cannot build a query with no MATCH statements")
        if branch:
            returns = '\n'.join([f"MATCH p1=({self.current_varname})<-[*]-(n1)",
                                 f"MATCH p2=({self.current_varname})-[*]->(n2)",
                                 f"WITH collect(p2) as p2s, collect(p1) as p1s",
                                 f"CALL apoc.convert.toTree(p1s+p2s) yield value",
                                 f"RETURN value"])
            returns = r'//Add Hierarchy Branch'+'\n' + returns
        else:
            returns = f"\nRETURN DISTINCT {self.current_varname}.id as {self.current_label}"
        return f"{match}\n{returns}"

    def index_by_address(self, address):
        if self.current_varname is None:
            name = '<first>'
        else:
            name = self.current_varname
        blocks = []
        for k, v in address.items():
            k = k.lower()
            count = self.counter[k]
            path_match = f"MATCH ({k}{count}: {k} {{value: {quote(v)}}})-[*]->({name})"
            possible_index_match = f"OPTIONAL MATCH ({name})<-[:indexes]-({k}{count+1}: {k})"
            with_segment = f"WITH {k}{count}, {k}{count+1}, {name}"
            where = f"WHERE {k}{count}={k}{count+1} OR {k}{count+1} IS NULL"
            block = [path_match, possible_index_match, with_segment, where]
            self.counter[k] += 2
            blocks.append(block)
        return self.spawn(self.blocks + blocks, self.current_varname, self.current_label)

    def index_by_hierarchy_name(self, hierarchy_name, direction):
        name = '{}{}'.format(hierarchy_name.lower(), self.counter[hierarchy_name])
        if self.current_varname is None:
            first_encountered = False
            blocks = deepcopy(self.blocks)
            for i, block in enumerate(blocks):
                for j, line in enumerate(block):
                    if '<first>' in line and not first_encountered:
                        blocks[i][j] = line.replace('<first>', f'{name}:{hierarchy_name}')
                        first_encountered = True
                    else:
                        blocks[i][j] = line.replace('<first>', f'{name}')
            current_varname = name
        else:
            if direction == 'above':
                arrows = '<-[*]-'
            elif direction == 'below':
                arrows = '-[*]->'
            else:
                raise ValueError(f"Direction must be above or below")
            blocks = self.blocks + [[f"MATCH ({self.current_varname}){arrows}({name}:{hierarchy_name})"]]
            current_varname = name
        self.counter[hierarchy_name] += 1
        return self.spawn(blocks, current_varname, hierarchy_name)

    def index_by_id(self, id_value):
        blocks = self.blocks + [[f"WITH {self.current_varname}",
                                 f"WHERE {self.current_varname}.id = {quote(id_value)}"]]
        return self.spawn(blocks, self.current_varname, self.current_label)


class Indexable:
    def __init__(self, data, query: BasicQuery):
        self.data = data
        self.query = query

    def index_by_single_hierarchy(self, hierarchy_name, direction):
        hierarchy = self.data.singular_hierarchies[hierarchy_name]
        name = hierarchy.__name__
        query = self.query.index_by_hierarchy_name(name, direction)
        return SingleHierarchy(self.data, query, hierarchy)

    def index_by_address(self, address):
        query = self.query.index_by_address(address)
        return HeterogeneousHierarchy(self.data, query)

    def index_by_plural_hierarchy(self, hierarchy_name, direction):
        hierarchy = self.data.singular_hierarchies[self.singular_name(hierarchy_name)]
        query = self.query.index_by_hierarchy_name(hierarchy.__name__, direction)
        return HomogeneousHierarchy(self.data, query, hierarchy)

    def plural_name(self, singular_name):
        try:
            return self.data.plural_factors[singular_name] + 's'
        except KeyError:
            return self.data.singular_hierarchies[singular_name].plural_name

    def singular_name(self, plural_name):
        try:
            return self.data.singular_factors[plural_name[:-1]]
        except KeyError:
            return self.data.plural_hierarchies[plural_name].singular_name

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        return name in self.data.plural_hierarchies or name in self.data.plural_factors

    def is_singular_name(self, name):
        return name in self.data.singular_hierarchies or name in self.data.singular_factors

    def implied_plurality_direction_of_node(self, name):
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

    def implied_plurality_direction_of_node(self, name):
        return True, 'below'

    def index_by_hierarchy_name(self, hierarchy_name):
        if not self.is_plural_name(hierarchy_name):
            raise NotImplementedError(f"Can only index plural hierarchies in a heterogeneous address")
        return self.index_by_plural_hierarchy(hierarchy_name, 'below')

    def __getattr__(self, item):
        return self.index_by_hierarchy_name(item)


class ExecutableHierarchy(Indexable):
    def __init__(self, data, query, nodetype):
        assert issubclass(nodetype, Graphable)
        super().__init__(data, query)
        self.nodetype = nodetype

    def __call__(self):
        id_list = self.data.graph.neograph.run(self.query.make(False)).to_ndarray().T[0]
        return id_list
        objects = neo4jjson2hierarchies(self.query.make(True))
        d = {i.identifier: i for i in objects if isinstance(i, self.nodetype)}
        return [d[i] for i in id_list]


class SingleHierarchy(ExecutableHierarchy):
    def __init__(self, data, query, nodetype, idvalue=None):
        assert issubclass(nodetype, Graphable)
        super().__init__(data, query, nodetype)
        self.idvalue = idvalue

    def index_by_address(self, address):
        raise NotImplementedError("Cannot index a single hierarchy by an address")

    def index_by_hierarchy_name(self, hierarchy_name):
        if self.is_plural_name(hierarchy_name):
            multiplicity, direction = self.implied_plurality_direction_of_node(hierarchy_name)
            return self.index_by_plural_hierarchy(hierarchy_name, direction)
        elif self.is_singular_name(hierarchy_name):
            multiplicity, direction = self.implied_plurality_direction_of_node(hierarchy_name)
            if multiplicity:
                plural = self.plural_name(hierarchy_name)
                raise KeyError(f"{self} has several possible {plural}. Please use `.{plural}` instead")
            return self.index_by_single_hierarchy(hierarchy_name, direction)
        else:
            raise KeyError(f"{hierarchy_name} is an unknown factor/hierarchy")

    def implied_plurality_direction_of_node(self, name):
        if self.is_plural_name(name):
            name = self.singular_name(name)
        return self.data.node_implies_plurality_of(self.nodetype.singular_name, name)

    def __getattr__(self, item):
        return self.index_by_hierarchy_name(item)


class HomogeneousHierarchy(ExecutableHierarchy):
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
    def index_by_id(self, idvalue):
        query = self.query.index_by_id(idvalue)
        return SingleHierarchy(self.data, query, self.nodetype, idvalue)

    def implied_plurality_direction_of_node(self, name):
        return self.data.node_implies_plurality_of(self.nodetype.singular_name, name)

    def index_by_address(self, address):
        query = self.query.index_by_address(address)
        return HomogeneousHierarchy(self.data, query, self.nodetype)

    def __getitem__(self, item):
        if isinstance(item, Address):
            return self.index_by_address(item)
        else:
            return self.index_by_id(item)


class OurData(Data):
    filetypes = [Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget]
