from copy import deepcopy as copy
from typing import Union, Any, Tuple

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

    def _equality(self, parent, other, negate=False):
        raise NotImplementedError

    def _compare(self, parent, other, operation):
        raise NotImplementedError

    def _combine(self, parent, other, operation):
        raise NotImplementedError