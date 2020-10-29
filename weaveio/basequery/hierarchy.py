from copy import copy
from typing import List, Tuple, Union, Any

from .query import AmbiguousPathError, FullQuery, Node, Path, Condition
from ..address import Address
from .common import NotYetImplementedError, FrozenQuery
from ..hierarchy import Hierarchy
from ..utilities import quote


class HierarchyFrozenQuery(FrozenQuery):
    def __getitem__(self, item):
        raise NotImplementedError

    def __getattr__(self, item):
        raise NotImplementedError

    def _prepare_query(self, query):
        raise NotImplementedError

    def _execute_query(self):
        query = self._prepare_query(copy(self.query))
        cypher = query.to_neo4j()
        return self._post_process(self.data.graph.execute(cypher))

    def _post_process(self, result):
        raise NotImplementedError

    def __call__(self):
        result = self._execute_query()
        return self._post_process(result)


class HeterogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    def __getattr__(self, item):
        if item in self.data.factors:
            return self._get_set_of_factors()
        elif item in self.data.singular_hierarchies:
            raise IndexError(f"Cannot return a singular hierarchy without filtering first")
        else:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)

    def _get_plural_hierarchy(self, hierarchy_name):
        hier = self.data.singular_hierarchies[hierarchy_name]
        label = hier.__name__
        start = self.handler.generator.node(label)
        root = [Path(start)]
        return HomogeneousHierarchyFrozenQuery(self.handler, FullQuery(root), hier, self)


class DefiniteHierarchyFrozenQuery(HierarchyFrozenQuery):
    def _prepare_query(self, query):
        query.returns.append(self.query.current_node)
        return query

    def __getattr__(self, item):
        if item in self.data.singular_factors:
            return self._get_set_of_factors()
        elif item in self.data.plural_factors:
            name = self.data.singular_name(item)
            return self._get_set_of_factors(name)
        elif item in self.data.singular_hierarchies:
            return self._get_singular_hierarchy(item)
        elif item in self.data.plural_hierarchies:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)
        else:
            raise AttributeError(f"{self} has no attribute {item}")

    def node_implies_plurality_of(self, end):
        start = self._hierarchy.singular_name.lower()
        multiplicity, direction, path = self.data.node_implies_plurality_of(start, end)
        path = [self.data.singular_hierarchies[n].__name__ for n in path]
        if direction == 'above':
            arrow = '<--'
        elif direction == 'below':
            arrow = '-->'
        else:
            raise ValueError(f"direction {direction} not known")
        nodes = self.handler.generator.nodes(*path[1:])
        node_path = [self.query.current_node]
        for node in nodes:
            node_path += [arrow, node]
        path = Path(*node_path)
        return multiplicity, path

    def _get_plural_hierarchy(self, name):
        query = copy(self.query)
        multiplicity, path = self.node_implies_plurality_of(name)
        # dont check for multiplicity here, since plural is requested anyway
        query.matches.append(path)
        h = self.handler.data.singular_hierarchies[name]
        return HomogeneousHierarchyFrozenQuery(self.handler, query, h, self)


class SingleHierarchyFrozenQuery(DefiniteHierarchyFrozenQuery):
    def __init__(self, handler, query: FullQuery, hierarchy: Hierarchy, identifier: Any, parent: 'FrozenQuery'):
        super().__init__(handler, query, parent)
        self._hierarchy = hierarchy
        self._identifier = identifier


    def _get_singular_hierarchy(self, name):
        query = copy(self.query)
        multiplicity, path = self.node_implies_plurality_of(name)
        if multiplicity:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"You have requested a single {name} but {self} has multiple {plural}. Use .{plural}")
        query.matches.append(path)
        h = self.handler.data.singular_hierarchies[name]
        return SingleHierarchyFrozenQuery(self.handler, query, h, None, self)


class HomogeneousHierarchyFrozenQuery(DefiniteHierarchyFrozenQuery):
    def __init__(self, handler, query: FullQuery, hierarchy: Hierarchy, parent: 'FrozenQuery'):
        super().__init__(handler, query, parent)
        self._hierarchy = hierarchy

    def __getitem__(self, item):
        return self._filter_by_identifier(item)

    def _filter_by_identifier(self, identifier: Union[str,int,float]) -> SingleHierarchyFrozenQuery:
        query = copy(self.query)
        condition = Condition(query.current_node.id, '=', identifier)
        if query.conditions is not None:
            query.conditions = query.conditions & condition
        else:
            query.conditions = condition
        return SingleHierarchyFrozenQuery(self.handler, query, self._hierarchy, identifier, self)
