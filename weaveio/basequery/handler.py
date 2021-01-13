from copy import deepcopy as copy
from typing import Union, Any, Tuple

import networkx as nx

from .hierarchy import *
from .factor import *
from .query import AmbiguousPathError, FullQuery
from .query_objects import Path, Generator
from .tree import BranchHandler
from ..utilities import quote


class Handler:
    def __init__(self, data):
        self.data = data
        self.generator = Generator()

    def begin_with_heterogeneous(self):
        query_handler = BranchHandler()
        return HeterogeneousHierarchyFrozenQuery(self, query_handler)

    def hierarchy_of_factor(self, factor_name: str, start: Type[Hierarchy] = None) -> Tuple[str, str, str]:
        namelist = factor_name.split('.')
        hierarchy_names = None
        singular_name = None
        if len(namelist) > 1:
            hierarchy_name_list, factor_name = namelist[:-1], namelist[-1]
            singular_name = self.data.singular_name(factor_name)
            if not all(h in self.data.singular_hierarchies for h in hierarchy_name_list):
                raise KeyError(f"Not all of {hierarchy_name_list} are valid factor names")
            if not nx.has_path(self.data.relation_graph, hierarchy_name_list[0], hierarchy_name_list[-1]):
                raise KeyError(f"The path {'-'.join(hierarchy_name_list)} does not exist in the schema")
            hierarchy_names = []
            for n in self.data.factor_hierarchies[singular_name]:
                if hierarchy_name_list[-1] == n or nx.has_path(self.data.relation_graph, hierarchy_name_list[-1], n.singular_name):
                    hierarchy_names.append(n)
            if len(hierarchy_names) == 0:
                raise KeyError(f"`{hierarchy_name_list[-1]}` does not have a factor `{singular_name}`")
        elif start is not None:
            if factor_name in start.factors:
                hierarchy_names = [start]
        if singular_name is None:
            singular_name = self.data.singular_name(factor_name)  # singular_name
        if hierarchy_names is None:
            hierarchy_names = self.data.factor_hierarchies[singular_name]
        if len(hierarchy_names) > 1:
            raise AmbiguousPathError(f"The factor {singular_name} is ambiguous. "
                                     f"{singular_name} has {len(hierarchy_names)} parents: {hierarchy_names}."
                                     f"Be explicit and choose one of them. "
                                     f"E.g. {hierarchy_names[0]}.{factor_name}")
        else:
            return hierarchy_names[0].singular_name.lower(), factor_name, singular_name

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