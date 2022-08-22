import operator
from typing import Any, overload

from typing_extensions import SupportsIndex, Literal

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

    def apply_zip(self, function, *other_wrappers):
        return self.__class__(*[function(*x) for x in zip(self.unnamed_queries, *[o.unamed_queries for o in other_wrappers])],
                              **{k: getattr(self, k) for k in self.pass_to_init},
                              **{k: function(v, *[o[k] for o in other_wrappers]) for k, v in self.named_queries.items()})

    def __getitem__(self, item):
        func = operator.itemgetter(item)
        return self.apply(func)

    def __getattr__(self, item):
        func = operator.attrgetter(item)
        return self.apply(func)

    def __and__(self, other):
        return self.apply(lambda x: x.__and__(other))

    def __rand__(self, other):
        return self.apply(lambda x: x.__rand__(other))

    def __or__(self, other):
        return self.apply(lambda x: x.__or__(other))

    def __ror__(self, other):
        return self.apply(lambda x: x.__ror__(other))

    def __xor__(self, other):
        return self.apply(lambda x: x.__xor__(other))

    def __rxor__(self, other):
        return self.apply(lambda x: x.__rxor__(other))

    def __invert__(self):
        return self.apply(lambda x: x.__invert__())

    def __add__(self, other):
        return self.apply(lambda x: x.__add__(other))

    def __radd__(self, other):
        return self.apply(lambda x: x.__radd__(other))

    def __mul__(self, other):
        return self.apply(lambda x: x.__mul__(other))

    def __rmul__(self, other):
        return self.apply(lambda x: x.__rmul__(other))

    def __sub__(self, other):
        return self.apply(lambda x: x.__sub__(other))

    def __rsub__(self, other):
        return self.apply(lambda x: x.__rsub__(other))

    def __truediv__(self, other):
        return self.apply(lambda x: x.__truediv__(other))

    def __rtruediv__(self, other):
        return self.apply(lambda x: x.__rtruediv__(other))

    def __eq__(self, other):
        return self.apply(lambda x: x.__eq__(other))

    def __ne__(self, other):
        return self.apply(lambda x: x.__ne__(other))

    def __lt__(self, other):
        return self.apply(lambda x: x.__lt__(other))

    def __le__(self, other):
        return self.apply(lambda x: x.__le__(other))

    def __gt__(self, other):
        return self.apply(lambda x: x.__gt__(other))

    def __ge__(self, other):
        return self.apply(lambda x: x.__ge__(other))

    def __ceil__(self):
        return self.apply(lambda x: x.__ceil__())

    def __floor__(self):
        return self.apply(lambda x: x.__floor__())

    def __round__(self, ndigits: int):
        return self.apply(lambda x: x.__round__())

    def __neg__(self):
        return self.apply(lambda x: x.__neg__())

    def __abs__(self):
        return self.apply(lambda x: x.__abs__())

    def __iadd__(self, other):
        return self.apply(lambda x: x.__iadd__())

    def _perform_arithmetic(self, op_string, op_name, other=None, expected_dtype=None, returns_dtype=None, parameters=None):
        return

    def __call__(self, skip=0, limit=1000, distinct=False, **kwargs):
        return {k: v(skip=skip, limit=limit, distinct=distinct, **kwargs) for k, v in self.queries.items()}

    def __iter__(self):
        queries = {k: q._iterate() for k, q in self.queries.items()}
        while True:
            row = {k: next(v, None) for k, v in queries.items()}
            if all(v is None for v in row.values()):
                break
            yield row






