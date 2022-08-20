from functools import reduce
from typing import Union

from .objects import BaseQuery, TableQuery
import networkx as nx

from .wrappers import QueryWrapper


class AlignedQuery(QueryWrapper):
    pass_to_init = ['wrt']

    def __init__(self, *unnamed_queries, wrt, **named_queries):
        super().__init__(*unnamed_queries, **named_queries)
        self.wrt = wrt

    def _precompile(self):
        if any(isinstance(v, TableQuery) for v in self.queries.values()):
            if not all(isinstance(v, TableQuery) for v in self.queries.values()):
                raise ValueError('All queries must be tables if any are tables')
            return self.wrt[{f"{k}_": q for k, q in self.queries.items()}]
        return QueryWrapper(**self.queries)

    def __call__(self, skip=0, limit=1000, distinct=False, **kwargs):
        q = self._precompile()
        return q(skip, limit, distinct, **kwargs)

    def __iter__(self):
        q = self._precompile()
        return q.__iter__()


def align(*unnamed_queries: BaseQuery, wrt=None, **named_queries: BaseQuery):
    queries = unnamed_queries + tuple(named_queries.values())
    if not all(isinstance(q, BaseQuery) for q in queries):
        raise TypeError('All queries must be Queries objects')
    data = queries[0]._data
    if not all(q._data is data for q in queries):
        raise ValueError('All queries must be from the same parent Data object')
    if wrt is None:
        d = {}
        previous_nodes = []
        for q in queries:
            d[q._node] = q
            _previous_nodes = []
            while True:
                _previous_nodes.append(q._node)
                q = q._previous
                if q is None:
                    break
            previous_nodes.append(_previous_nodes)
        wrt = d[max(reduce(set.intersection, map(set, previous_nodes)))]
    else:
        descendants = nx.descendants(queries[0]._G.G, wrt._node)
        if not all(q._node in descendants for q in queries):
            raise ValueError(f'wrt {wrt} must be a common ancestor of all queries')
    return AlignedQuery(*unnamed_queries, wrt=wrt, **named_queries)

