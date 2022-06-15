import math
from itertools import zip_longest
from typing import Type, Union, Set, List
import networkx as nx
from networkx.classes.filters import no_filter

from weaveio.data import get_all_class_bases, plot_graph
from weaveio.hierarchy import Multiple, OneOf, Hierarchy


def normalise_relation(h):
    if not isinstance(h, Multiple):
        h = OneOf(h)
    h.instantate_node()
    relation, node = h, h.node
    return relation, node

def get_all_subclasses(cls: Type) -> List[Type]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def hierarchies_from_hierarchy(hier: Type[Hierarchy], done=None, templates=False) -> Set[Type[Hierarchy]]:
    if done is None:
        done = []
    hierarchies = set()
    todo = {h.node if isinstance(h, Multiple) else h for h in hier.parents + hier.children + hier.produces}
    if not templates:
        todo = {h for h in todo if not h.is_template}
    else:
        todo.update({h for h in todo for hh in get_all_class_bases(h) if issubclass(hh, Hierarchy)})
    for new in todo:
        if isinstance(new, Multiple):
            new.instantate_node()
            h = new.node
        else:
            h = new
        if h not in done and h is not hier:
            hierarchies.update(hierarchies_from_hierarchy(h, done, templates))
            done.append(h)
    hierarchies.add(hier)
    return hierarchies


def expand_template_relation(relation):
    """
    Returns a list of relations that relate to each non-template class
    e.g.
    >>> expand_template_relation(Multiple(L1StackedSpectrum))
    [Multiple(L1SingleSpectrum), Multiple(L1StackSpectrum), Multiple(L1SuperstackSpectrum)]
    """
    if not relation.node.is_template:
        return [relation]
    subclasses = [cls for cls in get_all_subclasses(relation.node) if not cls.is_template and issubclass(cls, Hierarchy)]
    return [Multiple(subclass, 0, relation.maxnumber, relation.constrain, relation.relation_idname, relation.one2one) for subclass in subclasses]


def shortest_simple_paths(G, source, target, weight):
    try:
        return nx.shortest_simple_paths(G, source, target, weight)
    except nx.NetworkXNoPath:
        return None

def find_forking_path(graph, top, bottom, weight=None):
    done = set()
    parents = nx.subgraph_view(graph, filter_edge=lambda *e: graph.edges[e]['type'] != 'is_a')
    gen = nx.shortest_simple_paths(graph, bottom, top, weight)
    try:
        todo = [next(gen)]
    except nx.NetworkXNoPath:
        return []
    while todo:
        _path = todo.pop()
        edges = list(zip(_path[:-1], _path[1:]))
        types = [graph.edges[edge]['type'] for edge in edges]
        path = [_path[0]]
        for i, (edge, typ) in enumerate(zip(edges, types)):
            if typ == 'is_a' and edge[1] != top:
                # try to find a path using the superclass
                subpaths = find_forking_path(parents, top, edge[0], weight)
                if subpaths:
                    for subpath in subpaths:
                        done.add((*path[:-1], *subpath))
                else:  # otherwise expand all classes
                    subclasses = [i for i in get_all_subclasses(edge[0]) if not i.is_template]
                    if top in subclasses:
                        done.add((*path, top))
                    else:
                        for subclass in subclasses:
                            for subpath in find_forking_path(graph, top, subclass, weight):
                                done.add((*path, *subpath))
                break
            else:
                path.append(edge[1])
        else:  # only append here, if we've cycled through all edges without breaking
            done.add(tuple(path))
    return done


class HierarchyGraph(nx.DiGraph):
    def initialise(self):
        hiers = get_all_subclasses(Hierarchy)
        for h in hiers:
            self._add_hierarchy(h)
        self.remove_node(Hierarchy)
        self._assign_edge_weights()

    def surrounding_nodes(self, n):
        return list(self.parents.successors(n)) + list(self.parents.predecessors(n)) + [n]

    def _add_parent(self, child: Type[Hierarchy], parent: Union[Type[Hierarchy], Multiple]):
        relation, parent = normalise_relation(parent)
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        self.add_edge(child, parent, singular=relation.maxnumber == 1, one2one=relation.one2one,
                       optional=relation.minnumber == 0, style=relstyle, actual_number=relation.maxnumber, type='is_child_of')
        if relation.one2one:
            self.add_edge(parent, child, singular=True, optional=relation.minnumber == 0, style='solid',
                          type='is_parent_of', one2one=relation.one2one,
                          relation=relation, actual_number=relation.maxnumber)
        else:
            self.add_edge(parent, child, singular=False, optional=relation.minnumber == 0, style='dashed',
                          type='is_parent_of', one2one=relation.one2one,
                          relation=relation, actual_number=relation.maxnumber)

    def _add_child(self, parent: Type[Hierarchy], child: Union[Type[Hierarchy], Multiple]):
        relation, child = normalise_relation(child)
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        self.add_edge(parent, child, singular=relation.maxnumber == 1, one2one=relation.one2one,
                       optional=relation.minnumber == 0, type='is_parent_of',
                       relation=relation, style=relstyle, actual_number=relation.maxnumber)
        self.add_edge(child, parent, singular=True, optional=relation.minnumber == 0,  one2one=relation.one2one,
                      style='solid', type='is_child_of', actual_number=relation.maxnumber)

    def _add_self_reference(self, relation):
        relation, h = normalise_relation(relation)
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        self.add_edge(h, h, singular=relation.maxnumber == 1, optional=relation.minnumber == 0,
                      one2one=relation.one2one, style=relstyle, actual_number=relation.maxnumber, type='is_parent_of')


    def _add_inheritance(self, hierarchy, base):
        self.add_edge(base, hierarchy, type='subclassed_by', style='dotted', optional=False, one2one=False)
        self.add_edge(hierarchy, base, type='is_a', style='dotted', optional=False, one2one=False)

    def _assign_edge_weights(self):
        for u, v, d in self.edges(data=True):
            if d['type'] == 'is_a':
                weight = 0
            elif d['type'] == 'subclassed_by':
                weight = math.inf
            elif d['singular']:
                weight = 0
            else:
                weight = d['actual_number']
            self.edges[u, v]['weight'] = weight

    def _add_hierarchy(self, hierarchy: Type[Hierarchy]):
        """
        For a given hierarchy, traverse all its required inputs (parents and children)
        """
        self.add_node(hierarchy)
        for parent in hierarchy.parents:
            if hierarchy is parent:
                self._add_self_reference(parent)
            else:
                self._add_parent(hierarchy, parent)
        for child in hierarchy.children:
            if hierarchy is child:
                self._add_self_reference(child)
            else:
                self._add_child(hierarchy, child)
        for inherited in hierarchy.__bases__:
            if issubclass(inherited, Hierarchy):
                self._add_inheritance(hierarchy, inherited)

    def ancestor_subgraph(self, source):
        nodes = nx.ancestors(self.parents_and_subclassed_by, source)
        nodes.add(source)
        return nx.subgraph(self, nodes).copy()

    def subgraph_view(self, filter_node=no_filter, filter_edge=no_filter) -> 'HierarchyGraph':
        return nx.subgraph_view(self, filter_node, filter_edge)  # type: HierarchyGraph

    @property
    def inheritance(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_a').copy()

    @property
    def parents_and_subclassed_by(self):
        allowed = ['is_parent_of', 'subclassed_by']
        return self.subgraph_view(filter_edge=lambda *e: any(i == self.edges[e]['type'] for i in allowed)).copy()

    @property
    def parents(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_parent_of').copy()

    @property
    def children(self):
        return self.subgraph_view(filter_edge=lambda *e: graph.edges[e]['type'] == 'is_child_of').copy()

    @property
    def parents_and_inheritance(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_parent_of' or 'is_a' == self.edges[e]['type']).copy()

    @property
    def traversal(self):
        def func(u, v):
            edge = self.edges[u, v]
            typ = edge['type']
            return (typ == 'is_parent_of') or (typ == 'is_a') or edge['one2one']
        return self.subgraph_view(filter_edge=func)

    @property
    def children_and_inheritance(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_child_of' or 'is_a' == self.edges[e]['type']).copy()

    def shortest_unidirectional_paths(self, a, b, weight=None):
        """Returns a generator of unidirectional paths between a and b where a and b can both be source or target"""
        ab = shortest_simple_paths(self.parents_and_inheritance, a, b, weight)
        ba = shortest_simple_paths(self.children_and_inheritance, a, b, weight)
        for x, y in zip_longest(ab, ba):
            if x is not None:
                yield x
            if y is not None:
                yield y
            if x is None and y is None:
                raise nx.NetworkXNoPath(f"No unidirectional path between {a} and {b}")

    def sort_deepest(self, a, b):
        """Returns a or b, whichever is the deepest in the graph"""
        if a not in self:
            raise nx.NodeNotFound(f"Node {a} not found in graph")
        if b not in self:
            raise nx.NodeNotFound(f"Node {b} not found in graph")
        if b in nx.ancestors(self.parents_and_inheritance, a):
            return b, a
        elif a in nx.ancestors(self.parents_and_inheritance, b):
            return a, b
        else:
            raise nx.NetworkXNoPath(f"There is no path between {a} and {b}")

    def find_paths(self, a, b):
        G = self.parents_and_inheritance.reverse()
        sorted_nodes = self.sort_deepest(*nodes)
        _paths = find_forking_path(G, *sorted_nodes, 'weight')
        # put in requested order (a-->b)
        if sorted_nodes != (a, b):
            _paths = [path[::-1] for path in _paths]
        # now remove chains of is_a
        # so x-l1spectrum-l1stacked-l1stack becomes x-l1stack
        reduced_paths = set()
        for ip, path in enumerate(_paths):
            reduced_path = []
            for ic, current in enumerate(path[1:-1], 1):
                right = path[ic + 1]
                if not (issubclass(current, right) or issubclass(right, current)):
                    reduced_path.append(current)
            reduced_paths.add((path[0], *reduced_path, path[-1]))
        return reduced_paths


if __name__ == '__main__':
    graph = HierarchyGraph()
    graph.initialise()
    from weaveio.opr3.hierarchy import *
    from weaveio.opr3.l1files import *
    from weaveio.opr3.l2files import *


    # G = graph.ancestor_subgraph(L2StackFile).copy().parents_and_inheritance
    nodes = Redrock, Survey
    paths = graph.find_paths(*nodes)
    for path in paths:
        print('-'.join(n.__name__ for n in path))