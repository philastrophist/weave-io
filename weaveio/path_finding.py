import math
from itertools import zip_longest
from typing import Type, Union, Set, List, Tuple
import networkx as nx
from networkx.classes.filters import no_filter

from weaveio.hierarchy import Multiple, Hierarchy, OneOf, Graphable


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

def collapse_classes_to_superclasses(graph, classes):
    """
    Given a list of classes, return the shortest list of classes possible by merging classes together.
    A merge is only possible if they share a base class
    class B(A) and class C(A) can be merged -> class A
    In graph terms, this is equivalent to finding islands (connected components) and returning
     the shallowest node that has a path to all leaves (the head(s))
    """
    G = nx.subgraph_view(graph, filter_node=lambda n: n != Hierarchy and n in classes,
                         filter_edge=lambda *e: graph.edges[e]['type'] == 'subclassed_by')
    for ns in nx.connected_components(G.to_undirected()):
        subgraph = nx.subgraph(G, ns)
        for n in subgraph.nodes:
            if not subgraph.in_degree(n):
                yield n

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

def pick_shortest_multiedge(G, a, b, weight=None):
    edges = set(G.out_edges(a, keys=True)) & set(G.in_edges(b, keys=True))
    if not edges:
        raise nx.NetworkXNoPath("No path found between %s and %s" % (a, b))
    if weight is not None:
        return min(edges, key=lambda e: G.edges[e][weight])
    return edges.pop()

def edges_from_path(G, path, weight=None):
    return list(map(lambda e: pick_shortest_multiedge(G, *e, weight=weight), zip(path[:-1], path[1:])))

def find_forking_path(graph, top, bottom, weight=None):
    done = set()
    parents = nx.subgraph_view(graph, filter_edge=lambda *e: graph.edges[e]['type'] != 'is_a')
    is_a_graph = nx.subgraph_view(graph, filter_edge=lambda *e: graph.edges[e]['type'] == 'is_a')
    gen = nx.all_shortest_paths(graph, bottom, top, weight)
    try:
        todo = [next(gen)]
    except nx.NetworkXNoPath:
        return []
    while todo:
        _path = todo.pop()
        edges = edges_from_path(graph, _path, weight)
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
                        inheritance_path = nx.shortest_path(is_a_graph, edge[0], top)
                        done.add((*path, *inheritance_path[1:]))
                    else:
                        for subclass in subclasses:
                            inheritance_path = nx.shortest_path(is_a_graph, edge[0], subclass)
                            for subpath in find_forking_path(graph, top, subclass, weight):
                                done.add((*path, *inheritance_path[1:-1], *subpath))
                break
            else:
                path.append(edge[1])
        else:  # only append here, if we've cycled through all edges without breaking
            done.add(tuple(path))
    return done


def find_paths(graph, a, b, weight='weight') -> Set[Tuple[Type[Hierarchy]]]:
    """
    Returns a set of paths from a to b
    Not all paths are guaranteed to be valid, for example:
        l2superstack.ob yields paths of which `L2superstack-...-L1Single-...-ob` is one.
        This path is invalid, but it is included anyway because it is one of a set
        (through single,stack,superstack,supertarget).
        When queried, invalid paths will yield 0 results so it is not a problem.
    """
    *sorted_nodes, G, use_children = graph.sort_deepest(a, b)
    G = G.reverse()
    _paths = find_forking_path(G, *sorted_nodes, weight)
    _paths = {(p[0], p[0]) if len(p) == 1 else p for p in _paths}
    # put in requested order (a-->b)
    # now remove chains of is_a
    # so x-l1spectrum-l1stacked-l1stack becomes x-l1stack
    reduced_paths = set()
    for ip, path in enumerate(_paths):
        edges = edges_from_path(G, path, weight)
        reduced_path = [edge[0] for edge in edges if G.edges[edge]['type'] not in ['is_a', 'subclassed_by']]
        reduced_path = (*reduced_path, path[-1])
        if not use_children:
            reduced_path = reduced_path[::-1]  # reverse it because the actual database has only parent relations
            edges = (graph.short_edge(e[1], e[0], weight) for e in edges[::-1])
        reduced_paths.add((reduced_path, tuple(edges)))
    return reduced_paths


class HierarchyGraph(nx.MultiDiGraph):
    """
    A multidigraph where edges have keys
    """
    def initialise(self):
        hiers = get_all_subclasses(Hierarchy)
        for h in hiers:
            self._add_hierarchy(h)
        self.remove_node(Hierarchy)
        self._assign_edge_weights()

    def to_neo4j(self, graph):
        node_type = 'AbstractNode'
        for node in self.nodes:
            graph.execute(f"""MERGE (n:{node.__name__}{node_type}:{node_type}) ON CREATE 
            SET n.factors = $factors, n.identifier_builder = $identifier_builder, n.singular_name=$singular_name,
             n.plural_name = $plural_name, n.idname = $idname, n.products = $products""",
                          factors=node.factors, singular_name=node.singular_name, plural_name=node.plural_name,
                          idname=node.idname, identifier_builder=node.identifier_builder, products=node.products)
        for edge in self.edges:
            rel = {k: v for k, v in self.edges[edge].items() if not isinstance(v, Multiple)}
            try:
                rel.update(self.edges[edge]['relation'].to_reldict())
            except KeyError:
                rel['singular'] = False
            if 'constrain' in rel:
                rel['constrain'] = [i.__name__ for i in rel['constrain']]
            graph.execute(f"""MATCH (a:{edge[0].__name__}{node_type}) 
            MATCH (b:{edge[1].__name__}{node_type})
            MERGE (a)-[r:{rel.pop('type')}]->(b)
            ON CREATE SET r = $rel
            """, rel=rel)

    def add_edge(self, u_for_edge, v_for_edge, idname=None, **attr):
        """
        Create an edge if the edge (uniquely identified by attributes) doesn't exist
        the edge key will be an unused integer or the idname of the relation
        if idname is given, it will be used and attr is not checked for uniqueness (only ever 1 edge)
        if idname is not given, attr is checked for uniqueness and key is the number of the same edges
        """
        if not attr:
            return super(HierarchyGraph, self).add_edge(u_for_edge, v_for_edge, idname)
        hsh = hash(tuple(attr.items()))
        if idname is not None:
            # use idname as unique key
            key = super(HierarchyGraph, self).add_edge(u_for_edge, v_for_edge, idname, hsh=hsh, **attr)
            # and make a "general" link as well
            if (u_for_edge, v_for_edge, '*') in self.edges:
                self.edges[u_for_edge, v_for_edge, '*']['singular'] = False
                self.edges[u_for_edge, v_for_edge, '*']['actual_number'] += attr['actual_number']
            else:
                super(HierarchyGraph, self).add_edge(u_for_edge, v_for_edge, '*', hsh=hsh, **attr)
            return key
        if u_for_edge in self and v_for_edge in self:
            edges = set(self.out_edges(u_for_edge, keys=True)) & set(self.in_edges(v_for_edge, keys=True))
        else:
            edges = set()
        if hsh not in (self.edges[e]['hsh'] for e in edges):
            # use an incremented integer as a unique key only if the edge is unique in attr and (u,v)
            return super(HierarchyGraph, self).add_edge(u_for_edge, v_for_edge, hsh=hsh, **attr)

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
        self.add_edge(base, hierarchy, type='subclassed_by', actual_number=math.inf, style='dotted', optional=False,
                      one2one=False)
        self.add_edge(hierarchy, base, type='is_a', actual_number=0, style='dotted', optional=False, one2one=False)

    def _assign_edge_weights(self):
        for u, v, k, d in self.edges(data=True, keys=True):
            if d['type'] == 'is_a':
                weight = 0
            elif d['type'] == 'subclassed_by':
                weight = math.inf
            elif d['singular']:
                weight = 0
            else:
                weight = d['actual_number']
            self.edges[u, v, k]['weight'] = weight

    def _add_hierarchy(self, hierarchy: Type[Hierarchy]):
        """
        For a given hierarchy, traverse all its required inputs (parents and children)
        """
        self.add_node(hierarchy)
        for parent in hierarchy.parents:
            if normalise_relation(hierarchy)[1] is parent:
                self._add_self_reference(parent)
            else:
                self._add_parent(hierarchy, parent)
        for child in hierarchy.children:
            if normalise_relation(hierarchy)[1] is child:
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
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_child_of').copy()

    @property
    def parents_and_inheritance(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_parent_of' or 'is_a' == self.edges[e]['type']).copy()

    @property
    def children_and_inheritance(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['type'] == 'is_child_of' or 'is_a' == self.edges[e]['type']).copy()

    @property
    def traversal(self):
        def func(u, v):
            edge = self.edges[u, v]
            typ = edge['type']
            return (typ == 'is_parent_of') or (typ == 'is_a') or edge['one2one']
        return self.subgraph_view(filter_edge=func)

    @property
    def nonoptional(self):
        return self.subgraph_view(filter_edge=lambda *e: not self.edges[e]['optional']).copy()

    def sort_deepest(self, a, b):
        """Returns [a, b] or [b, a], in order of increasing depth in the graph"""
        if a is b:
            return a, b, self.parents_and_inheritance, False
        if a not in self:
            raise nx.NodeNotFound(f"Node {a} not found in graph {self}")
        if b not in self:
            raise nx.NodeNotFound(f"Node {b} not found in graph {self}")
        if b in nx.ancestors(self.parents_and_inheritance, a):
            return b, a, self.parents_and_inheritance, False
        elif a in nx.ancestors(self.parents_and_inheritance, b):
            return a, b, self.parents_and_inheritance, False
        elif a in nx.ancestors(self.children_and_inheritance, b):
            return a, b, self.children_and_inheritance, True
        elif b in nx.ancestors(self.children_and_inheritance, a):
            return b, a, self.children_and_inheritance, True
        else:
            raise nx.NetworkXNoPath(f"There is no path between {a} and {b}")

    def short_edge(self, a, b, weight=None):
        """return the shortest edge in the multidigraph for a given digraph edge """
        return pick_shortest_multiedge(self, a, b, weight)

    @property
    def singular(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['weight'] == 0)

    @property
    def plural(self):
        return self.subgraph_view(filter_edge=lambda *e: self.edges[e]['weight'] > 0)

    def find_paths(self, a, b, singular):
        """
        Returns a set of paths from a to b
        Not all paths are guaranteed to be valid, for example:
            l2superstack.ob yields paths of which `L2superstack-...-L1Single-...-ob` is one.
            This path is invalid, but it is included anyway because it is one of a set
            (through single,stack,superstack,supertarget).
            When queried, invalid paths will silently yield 0 results so it is not a problem.
        """
        try:
            return find_paths(self.singular.nonoptional, a, b)
        except nx.NetworkXNoPath:
            try:
                return find_paths(self.singular, a, b)
            except nx.NetworkXNoPath as e:
                if singular:
                    raise e
                try:
                    return find_paths(self.nonoptional, a, b)
                except nx.NetworkXNoPath:
                    return find_paths(self, a, b)

    def edge_weights(self, path) -> List[int]:
        return [self.edges[self.short_edge(*e, 'weight')]['weight'] for e in nx.utils.pairwise(path)]

    def edge_path_is_singular(self, edges):
        return not any(self.edges[e]['weight'] > 0 for e in edges)

    def path_is_singular(self, path) -> bool:
        return not any(e > 0 for e in self.edge_weights(path))


def get_all_class_bases(cls: Type[Graphable]) -> Set[Type[Graphable]]:
    new = set()
    for b in cls.__bases__:
        if b is Graphable or not issubclass(b, Graphable):
            continue
        new.add(b)
        new.update(get_all_class_bases(b))
    return new
