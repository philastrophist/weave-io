import operator

from .base import QueryFunctionBase


class QueryWrapper(QueryFunctionBase):
    pass_to_init = []

    def __init__(self, *unnamed_queries, **named_queries):
        self.unnamed_queries = unnamed_queries
        self.named_queries = named_queries
        self.queries = {**named_queries}
        self.queries.update({i: q for i, q in enumerate(unnamed_queries)})

    def __repr__(self):
        return f'{self.__class__.__name__}({self.queries})'

    def apply(self, function):
        # apply function to all queries and return a new QueryWrapper with the same named an unnamed queries
        return self.__class__(*map(function, self.unnamed_queries),
                              **{k: getattr(self, k) for k in self.pass_to_init},
                              **{k: function(v) for k, v in self.named_queries.items()})

    def __getitem__(self, item):
        func = operator.itemgetter(item)
        return self.apply(func)

    def __getattr__(self, item):
        func = operator.attrgetter(item)
        return self.apply(func)

    def __call__(self, skip=0, limit=1000, distinct=False, **kwargs):
        return {k: v(skip=skip, limit=limit, distinct=distinct, **kwargs) for k, v in self.queries.items()}

    def __iter__(self):
        queries = {k: q._iterate() for k, q in self.queries.items()}
        while True:
            row = {k: next(v, None) for k, v in queries.items()}
            if all(v is None for v in row.values()):
                break
            yield row






