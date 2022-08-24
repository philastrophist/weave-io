import operator
from collections.abc import Iterable, Callable
from typing import Any, overload

from typing_extensions import SupportsIndex, Literal

from .base import QueryFunctionBase
from math import floor, ceil



class QueryWrapper(QueryFunctionBase):

    def __init__(self, *unnamed_queries, **named_queries):
        self.unnamed_queries = unnamed_queries
        self.named_queries = named_queries
        self.queries = {**named_queries}
        self.queries.update({i: q for i, q in enumerate(unnamed_queries)})
        self.first_query = next(iter(self.queries.values()))

    def _precompile(self):
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}({next(iter(self.queries.values()))} [{self.queries.keys()}])'

    def apply(self, function, *args, **kwargs):
        return apply(function, self, *args, **kwargs)

    def __getitem__(self, item):
        return self.apply(operator.__getitem__, item)

    def __getattr__(self, item):
        return self.apply(getattr, item)

    def __and__(self, other):
        return self.apply(operator.__and__, other)

    def __or__(self, other):
        return self.apply(operator.__or__, other)

    def __xor__(self, other):
        return self.apply(operator.__xor__, other)

    def __invert__(self):
        return self.apply(operator.__invert__)

    def __add__(self, other):
        return self.apply(operator.__add__, other)

    def __mul__(self, other):
        return self.apply(operator.__mul__, other)

    def __sub__(self, other):
        return self.apply(operator.__sub__, other)

    def __truediv__(self, other):
        return self.apply(operator.__truediv__, other)

    def __eq__(self, other):
        return self.apply(operator.__eq__, other)

    def __ne__(self, other):
        return self.apply(operator.__ne__, other)

    def __lt__(self, other):
        return self.apply(operator.__lt__, other)

    def __le__(self, other):
        return self.apply(operator.__le__, other)

    def __gt__(self, other):
        return self.apply(operator.__gt__, other)

    def __ge__(self, other):
        return self.apply(operator.__ge__, other)

    def __ceil__(self):
        return self.apply(ceil)

    def __floor__(self):
        return self.apply(floor)

    def __round__(self, ndigits: int):
        return self.apply(round, ndigits=ndigits)

    def __neg__(self):
        return self.apply(operator.__neg__)

    def __abs__(self):
        return self.apply(operator.__abs__)

    def __iadd__(self, other):
        return self.apply(operator.__iadd__, other)

    def __call__(self, *args, **kwargs):
        if isinstance(self.first_query, QueryFunctionBase):
            q = self._precompile()
            if q is not self:
                return q(*args, **kwargs)
            return {k: v(*args, **kwargs) for k, v in self.queries.items()}
        return self.apply(lambda x, *a, **k: x(*a, **k), *args, **kwargs)

    def __iter__(self):
        q = self._precompile()
        if isinstance(q, QueryWrapper):
            queries = {k: q._iterate() for k, q in self.queries.items()}
            while True:
                row = {k: next(v, None) for k, v in queries.items()}
                if all(v is None for v in row.values()):
                    break
                yield row
        else:
            yield from iter(q)

    def _debug_output(self, *args, **kwargs):
        return self._precompile()._debug_output(*args, **kwargs)

def crawl_copy_replace(obj, args_i=None, kwargs_k=None):
    """
    Return a shallow copy of the given object but with any QueryWrapper contained in it replaced with one of its subqueries denoted
    by args_i or kwargs_k
    The crawling only happens on the first level of the object and only for iterables
    """
    if isinstance(obj, QueryWrapper):
        if args_i is not None and kwargs_k is not None:
            raise ValueError('cannot specify both args_i and kwargs_k')
        elif args_i is not None:
            return obj.unnamed_queries[args_i]
        elif kwargs_k is not None:
            return obj.named_queries[kwargs_k]
        else:
            raise ValueError('Either args_i or kwargs_k must be specified')
    elif isinstance(obj, dict):
        return {crawl_copy_replace(k, args_i, kwargs_k): crawl_copy_replace(v, args_i, kwargs_k) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        return obj.__class__([crawl_copy_replace(v, args_i, kwargs_k) for v in obj])
    else:
        return obj

def find_query_wrapper(obj):
    """
    Return the first QueryWrapper found in the given object
    """
    if isinstance(obj, QueryWrapper):
        return obj
    elif isinstance(obj, dict):
        for v in obj.values():
            q = find_query_wrapper(v)
            if q is not None:
                return q
    elif isinstance(obj, Iterable):
        for v in obj:
            q = find_query_wrapper(v)
            if q is not None:
                return q
    return None


def apply(function, *args, **kwargs):
    """
    Apply a function to all queries contained within any arg or kwarg which is a QueryWrapper.
    All unnamed and named queries will be matched and fed to the function.
    The effect (with only one unnamed_query and one named query) is the same as:
    >>> function(*[a.unnamed_queries[0] for a in args], **{k: v.unnamed_queries[0] for k, v in kwargs.items()})
    But this function is more flexible and can be used to apply a function to all queries in a QueryWrapper
    and any number of args or kwargs can be QueryWrappers or any other type of object.
    """
    q = find_query_wrapper({'args': args, 'kwargs': kwargs})
    if q is None:
        return function(*args, **kwargs)
    unnamed_queries = [function(*crawl_copy_replace(args, args_i=i), **crawl_copy_replace(kwargs, args_i=i)) for i in range(len(q.unnamed_queries))]
    named_queries = {k: function(*crawl_copy_replace(args, kwargs_k=k), **crawl_copy_replace(kwargs, kwargs_k=k)) for k, v in q.named_queries.items()}

    # wrappers = [a for a in args if isinstance(a, QueryWrapper)] + [v for k, v in kwargs.items() if isinstance(v, QueryWrapper)]
    # n_unnamed_queries = len(wrappers[0].unnamed_queries) if wrappers else 0
    # n_named_queries = len(wrappers[0].named_queries) if wrappers else 0
    # if not all(len(w.unnamed_queries) == n_unnamed_queries for w in wrappers) and \
    #         not all(w.named_queries.keys() == wrappers[0].named_queries.keys() for w in wrappers):
    #     raise ValueError('All unnamed_queries and named_queries must be the same length. These QueryWrappers are not compatible.')
    # if not (n_unnamed_queries or n_named_queries):
    #     return function(*args, **kwargs)
    # unnamed_queries = [function(*[a.unnamed_queries[i] if isinstance(a, QueryWrapper) else a for a in args],
    #                             **{k: v.unnamed_queries[i] if isinstance(v, QueryWrapper) else v for k, v in kwargs.items()})
    #                    for i in range(n_unnamed_queries)]
    # named_queries = {k: function(*[a.named_queries[k] if isinstance(a, QueryWrapper) else a for a in args],
    #                                 **{k: v.named_queries[k] if isinstance(v, QueryWrapper) else v for k, v in kwargs.items()})
    #                     for k in wrappers[0].named_queries}
    return q.__class__(*unnamed_queries, **named_queries)