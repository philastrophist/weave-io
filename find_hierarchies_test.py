from typing import Type, Union, Set, List
import networkx as nx

from weaveio.data import get_all_class_bases
from weaveio.hierarchy import Multiple, OneOf, Hierarchy


def normalise_relation(h):
    if not isinstance(h, Multiple):
        h = OneOf(h)
    h.instantate_node()
    relation, parent = h, h.node
    return relation, h

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


class HierarchyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        for h in get_all_subclasses(Hierarchy):
            self._add_hierarchy(h)

    def _add_parent(self, child: Type[Hierarchy], parent: Union[Type[Hierarchy], Multiple]):
        """
        Given a hierarchy and a parent of that hierarchy, create the required relationship in the graph
        Template hierarchies are expanded
        """
        relation, parent = normalise_relation(parent)
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        parent = relation.node  # reset from new relations
        self.graph.add_edge(child, parent, singular=relation.maxnumber == 1,
                       optional=relation.minnumber == 0, style=relstyle, actual_number=1, type='relates')
        if relation.one2one:
            self.graph.add_edge(parent, child, singular=True, optional=True, style='solid', type='relates',
                           relation=relation, actual_number=1)
        else:
            self.graph.add_edge(parent, child, singular=False, optional=True, style='dotted', type='relates',
                           relation=relation, actual_number=1)

    def _add_child(self, parent: Type[Hierarchy], child: Union[Type[Hierarchy], Multiple]):
        relation, child = normalise_relation(child)
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        child = relation.node  # reset from new relations
        self.graph.add_edge(parent, child, singular=relation.maxnumber == 1,
                       optional=relation.minnumber == 0, type='relates',
                       relation=relation, style=relstyle, actual_number=1)
        self.graph.add_edge(child, parent, singular=True, optional=True, style='solid', type='relates',
                            actual_number=1)

    def _add_self_reference(self, relation):
        relation, h = normalise_relation(relation)
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        self.graph.add_edge(h, h, singular=relation.maxnumber == 1, optional=relation.minnumber == 0,
                            style=relstyle, actual_number=1)


    def _add_inheritance(self, hierarchy, base):
        self.graph.add_edge(base, hierarchy, type='subclasses')


    def _add_hierarchy(self, hierarchy: Type[Hierarchy]):
        """
        For a given hierarchy, traverse all its required inputs (parents and children)
        """
        self.graph.add_node(hierarchy)
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




if __name__ == '__main__':
    graph = HierarchyGraph()
    graph