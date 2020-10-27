from copy import copy
from typing import List, Tuple, Union, Any

from .query import AmbiguousPathError, FullQuery, Node, Path
from ..address import Address
from .common import NotYetImplementedError, FrozenQuery
from ..hierarchy import Hierarchy
from ..utilities import quote


class HierarchyFrozenQuery(FrozenQuery):
    def __getitem__(self, item):
        raise NotImplementedError

    def __getattr__(self, item):
        raise NotImplementedError


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
        start = Node(label)
        root = Path(start)
        return HomogeneousHierarchyFrozenQuery(self.handler, FullQuery(root), hier, self)


class SingleHierarchyFrozenQuery(HierarchyFrozenQuery):
    def __init__(self, handler, query: FullQuery, hierarchy: Hierarchy, identifier: Any, parent: 'FrozenQuery'):
        super().__init__(handler, query, parent)
        self._hierarchy = hierarchy
        self._identifier = identifier


class HomogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    def __init__(self, handler, query: FullQuery, hierarchy: Hierarchy, parent: 'FrozenQuery'):
        super().__init__(handler, query, parent)
        self._hierarchy = hierarchy

    def __getitem__(self, item):
        return self._filter_by_identifier(item)

    def __getattr__(self, item):
        if item in self.data.factors:
            return self._get_set_of_factors()
        elif item in self.data.singular_hierarchies:
            return self._get_singular_hierarchy(item)
        else:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)

    def _filter_by_identifier(self, identifier: Union[str,int,float]) -> SingleHierarchyFrozenQuery:
        query = copy(self.query)
        query.conditions.append(f"{query.root[-1]} = {quote(identifier)}")
        return SingleHierarchyFrozenQuery(self.handler, query, self._hierarchy, identifier, self)
