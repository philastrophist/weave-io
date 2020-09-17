from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Union, Any, Dict, Set, List, Type, TypeVar

import networkx as nx

from weaveio.graph import Graph
from weaveio.hierarchy import File, Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget, Hierarchy, Multiple


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
        if implication_node in self.relation_graph.successors(start_node):
            return True, 'below'
        else:
            edges = nx.shortest_path(self.relation_graph, implication_node, start_node)
            return any(self.relation_graph.edges[(n1, n2)]['multiplicity'] > 1
                       for n1, n2 in zip(edges[:-1], edges[1:])), 'above'

    def __getitem__(self, address):
        return HeterogeneousHierarchy(self, BasicQuery())

    def __getattr__(self, item):
        return HeterogeneousHierarchy(self, BasicQuery()).__getattr__(item)


class BasicQuery:
    def __init__(self, matches: List = None, wheres: List = None, current_varname: str = None,
                 current_label: str = None, counter: defaultdict = None):
        self.matches = [] if matches is None else matches.copy()
        self.wheres = [] if wheres is None else wheres.copy()
        self.current_varname = current_varname
        self.current_label = current_label
        self.counter = defaultdict(int) if counter is None else counter

    def spawn(self, matches, wheres, current_varname, current_label) -> 'BasicQuery':
        return BasicQuery(matches, wheres, current_varname, current_label, self.counter)

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
        matches = self.matches + [f"(:{k} {{value: {quote(v)}}})-[*]->({name})" for k, v in address.items()]
        return self.spawn(matches, self.wheres, self.current_varname, self.current_label)

    def index_by_hierarchy_name(self, hierarchy_name, direction=None):
        name = '{}{}'.format(hierarchy_name.lower(), self.counter[hierarchy_name])
        if self.current_varname is None:
            first_encountered = False
            matches = self.matches.copy()
            for i, m in enumerate(matches):
                if '(first)' in m and not first_encountered:
                    matches[i] = m.replace('(first)', f'({name}:{hierarchy_name})')
                    first_encountered = True
                else:
                    matches[i] = m.replace('(first)', f'({name})')
            current_varname = name
        else:
            arrows = '<-[*]-' if direction == 'above' else '-[*]->'
            matches = self.matches + [f"({self.current_varname}){arrows}({name}:{hierarchy_name})"]
            current_varname = name
        self.counter[hierarchy_name] += 1
        return self.spawn(matches, self.wheres, current_varname, hierarchy_name)

    def index_by_id(self, id_value):
        wheres = self.wheres + [f"{self.current_varname}.id = {quote(id_value)}"]
        return self.spawn(self.matches, wheres, self.current_varname, self.current_label)


class Indexable:
    def __init__(self, data, query: BasicQuery):
        self.data = data
        self.query = query

    def __call__(self):
        raise NotImplementedError

    def index_by_single_hierarchy(self, hierarchy):
        query = self.query.index_by_hierarchy_name(hierarchy)
        return SingleHierarchy(self.data, query, self.data.singular_hierarchies[hierarchy])

    def index_by_address(self, address):
        query = self.query.index_by_address(address)
        return HeterogeneousHierarchy(self.data, query)

    def index_by_plural_hierarchy(self, hierarchy):
        query = self.query.index_by_hierarchy_name(hierarchy)
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
        return True

    def index_by_hierarchy_name(self, hierarchy_name):
        if not self.is_plural_name(hierarchy_name):
            raise NotImplementedError(f"Can only index plural hierarchies in a heterogeneous address")
        query = self.query.index_by_hierarchy_name(hierarchy_name)
        return HomogeneousHierarchy(self.data, query, self.singular_name(hierarchy_name))

    def __getattr__(self, item):
        return self.index_by_hierarchy_name(item)


class SingleHierarchy(Indexable):
    def __init__(self, data, query, nodetype, idvalue=None):
        super().__init__(data, query)
        self.nodetype = nodetype
        self.idvalue = idvalue

    def index_by_address(self, address):
        raise NotImplementedError("Cannot index a single hierarchy by an address")

    def index_by_hierarchy_name(self, hierarchy_name):
        if self.is_plural_name(hierarchy_name):
            return self.index_by_plural_hierarchy(hierarchy_name)
        elif self.is_singular_name(hierarchy_name):
            if self.implied_plurality_direction_of_node(hierarchy_name)[0]:
                plural = self.plural_name(hierarchy_name)
                raise KeyError(f"{self} has several possible {plural}. Please use `.{plural}` instead")
            return self.index_by_single_hierarchy(hierarchy_name)
        else:
            raise KeyError(f"{hierarchy_name} is an unknown factor/hierarchy")

    def implied_plurality_direction_of_node(self, name):
        return self.data.node_implies_plurality_of(self.nodetype, name)

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
    def __init__(self, data: Data, query: BasicQuery, nodetype):
        super().__init__(data, query)
        self.nodetype = nodetype

    def __call__(self):
        id_list = self.query.make(False)
        objects = neo4jjson2hierarchies(self.query.make(True))
        d = {i.identifier: i for i in objects if isinstance(i, self.nodetype)}
        return [d[i] for i in id_list]

    def index_by_id(self, idvalue):
        query = self.query.index_by_id(idvalue)
        return SingleHierarchy(self.data, query, self.nodetype, idvalue)

    def implied_plurality_direction_of_node(self, name):
        return self.data.node_implies_plurality_of(self.nodetype, name)

    def __getitem__(self, item):
        if isinstance(item, Address):
            return self.index_by_address(item)
        else:
            return self.index_by_id(item)


class OurData(Data):
    filetypes = [Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget]
