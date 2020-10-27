from copy import copy
from typing import List, Tuple, Union

from ..address import Address
from .common import NotYetImplementedError, FrozenQuery
from ..utilities import quote


class HierarchyFrozenQuery(FrozenQuery):
    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.handler.factors:
                return self.handler._get_single_factor(self, item)
            else:
                return self.handler._filter_by_identifier(item)
        elif isinstance(item, (tuple, list)):
            if item[0] in self.handler.factors:
                return self.handler._get_table_factor(self, item)
            else:
                return self.handler._filter_by_identifiers(item)
        elif isinstance(item, Address):
            return self.handler._filter_by_address(self, item)

    def __getattr__(self, item):
        raise NotImplementedError

class HeterogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    pass


class SingleHierarchyFrozenQuery(HierarchyFrozenQuery):
    pass


class HomogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    def _filter_by_identifier(self, identifier: Union[str,int,float]) -> SingleHierarchyFrozenQuery:
        query = copy(self.query)
        query.conditions.append(f"{query.root[-1]} = {quote(identifier)}")
        return SingleHierarchyFrozenQuery(self.handler, query, self)
