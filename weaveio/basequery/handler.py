from copy import deepcopy as copy
from typing import Union, Any

from .hierarchy import *
from .factor import *
from .query import Path, AmbiguousPathError, FullQuery, Generator
from ..utilities import quote


class Handler:
    def __init__(self, data):
        self.data = data
        self.generator = Generator()

    def begin_with_heterogeneous(self):
        return HeterogeneousHierarchyFrozenQuery(self, FullQuery())

    def hierarchy_of_factor(self, factor_name: str) -> str:
        factor_name = self.data.plural_factors.get(factor_name, factor_name)  # singular_name
        hierarchy_names = self.data.factor_hierarchies[factor_name]
        if len(hierarchy_names) > 1:
            raise AmbiguousPathError(f"The factor {factor_name} is ambiguous when starting from {self}. "
                                     f"{factor_name} has {len(hierarchy_names)} parents: {hierarchy_names}."
                                     f"Be explicit and choose one of them. "
                                     f"E.g. {self}.{hierarchy_names[0]}.{factor_name}")
        else:
            return hierarchy_names[0].singular_name.lower()

    def path(self, start, end) -> Path:
        raise NotImplementedError

    def _filter_by_boolean(self, parent, boolean):
        raise NotImplementedError

    def _get_single_factor(self, parent: HierarchyFrozenQuery, factor_name: str) -> SingleFactorFrozenQuery:
        query = copy(parent.query)
        h = self.hierarchy_of_factor(factor_name)
        if isinstance(parent, HeterogeneousHierarchyFrozenQuery):
            raise AmbiguousPathError(f"{query.matches[-1].name} has multiple {factor_name}s. Use .{factor_name}s")
        if isinstance(parent, HomogeneousHierarchyFrozenQuery):
            if self.guaranteed_not_sharing_parent(query.matches[-1].name, h):
                raise AmbiguousPathError(f"{query.matches[-1].name}.{factor_name} can only work if"
                                         f"they share the same parent (this is not locally decidable)."
                                         f" Use .{factor_name}s")
        if h not in query.matches:
            long_path = self.path(query.matches.nodes[0], h)
            if self.path_plurality(long_path):
                raise AmbiguousPathError(f"{query.matches} has multiple {factor_name}s. Use .{factor_name}s")
            short_path = query.matches.merge(long_path)
            query.exist_branches.append(short_path)
        query.return_properties.append((query.matches[h], factor_name))  # append pair of node, prop
        return SingleFactorFrozenQuery(self, query, parent)

    def _get_plural_factor(self, parent: HierarchyFrozenQuery, factor_name: str):
        query = copy(parent.query)
        h = self.hierarchy_of_factor(factor_name)
        if h not in query.matches:
            long_path = self.path(query.matches.nodes[0], h)
            short_path = query.matches.merge(long_path)
            query.exist_branches.append(short_path)
        query.return_properties.append((query.matches[h], factor_name))  # append pair of node, prop
        cls = TableFactorFrozenQuery
        return cls(self, query, parent)

    def _get_different_factors(self, parent, factor_names: Union[List, Tuple]):
        query = copy(parent.query)
        for factor_name in factor_names:
            h = self.hierarchy_of_factor(factor_name)
            if h not in query.matches:
                long_path = self.path(query.matches.nodes[0], h)
                short_path = query.matches.merge(long_path)
                query.exist_branches.append(short_path)
            query.return_properties.append((query.matches[h], factor_name))  # append pair of node, prop
        if isinstance(parent, HomogeneousHierarchyFrozenQuery):
            cls = TableFactorFrozenQuery
        elif isinstance(parent, SingleHierarchyFrozenQuery):
            cls = RowFactorFrozenQuery
        else:
            cls = TableFactorFrozenQuery
        return cls(self, query, isinstance(factor_names, list), parent)

    def _get_plural_hierarchy(self, parent, hierarchy_name):
        raise NotImplementedError

    def _equality(self, parent, other, negate=False):
        raise NotImplementedError

    def _compare(self, parent, other, operation):
        raise NotImplementedError

    def _combine(self, parent, other, operation):
        raise NotImplementedError