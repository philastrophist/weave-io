from functools import reduce

from .wrappers import QueryWrapper


class AlignedQuery(QueryWrapper):

    @property
    def aligned(self):
        return self.queries

    def _get_wrt(self):
        d = {}
        previous_nodes = []
        for q in self.queries.values():
            _previous_nodes = []
            while True:
                d[q._node] = q
                _previous_nodes.append(q._node)
                q = q._previous
                if q is None:
                    break
            previous_nodes.append(_previous_nodes)
        return d[max(reduce(set.intersection, map(set, previous_nodes)))]

    def _precompile(self):
        from .objects import TableQuery
        prefix, suffix = '', ''
        if any(isinstance(v, TableQuery) for v in self.queries.values()):
            if not all(isinstance(v, TableQuery) for v in self.queries.values()):
                raise ValueError('All queries must be tables if any are tables')
            suffix = '_'
        wrt = self._get_wrt()
        return wrt[{f"{prefix}{k}{suffix}": q for k, q in self.queries.items()}]


def align(*unnamed_queries, **named_queries):
    from .objects import BaseQuery
    queries = unnamed_queries + tuple(named_queries.values())
    if not all(isinstance(q, BaseQuery) for q in queries):
        raise TypeError('All queries must be Queries objects')
    data = queries[0]._data
    if not all(q._data is data for q in queries):
        raise ValueError('All queries must be from the same parent Data object')
    return AlignedQuery(*unnamed_queries, **named_queries)

