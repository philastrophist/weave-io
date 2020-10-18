from typing import List, Tuple

from ..address import Address
from .common import NotYetImplementedError, FrozenQuery


class HierarchyFrozenQuery(FrozenQuery):
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.handler._get_single_factor(self, item)
        elif isinstance(item, (tuple, list)):
            return self.handler._get_table_factor(self, item)
        elif isinstance(item, Address):
            return self.handler._filter_by_address(self, item)

    def __getattr__(self, item):
        raise NotImplementedError

class HeterogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    pass


class SingleHierarchyFrozenQuery(HierarchyFrozenQuery):
    pass


class HomogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    pass