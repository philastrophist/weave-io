from typing import Callable

from weaveio.basequery.common import FrozenQuery
from weaveio.basequery.dissociated import Dissociated
from weaveio.data import Data


def _template_aggregator(string_function, name, normal: Callable, item: Dissociated, wrt: FrozenQuery = None):
    if not isinstance(item, FrozenQuery):
        if wrt is not None:
            return normal(item, wrt)
        return normal(item)
    elif isinstance(wrt, Data) or wrt is None:
        branch = item.branch.handler.entry
    else:
        branch = wrt.branch
    new = branch.aggregate(string_function, item.variable, item.branch, True, name)
    return Dissociated(item.handler, new, new.action.target)


def sum(item, wrt=None):
    return _template_aggregator('sum({x})', 'sum', sum, item, wrt)


def max(item, wrt=None):
    return _template_aggregator('max({x})', 'max', max, item, wrt)


def min(item, wrt):
    return _template_aggregator('min({x})', 'min', min, item, wrt)


def all(item, wrt=None):
    return _template_aggregator('all({x})', 'all', all, item, wrt)


def any(item, wrt=None):
    return _template_aggregator('any({x})', 'any', any, item, wrt)


def count(item, wrt=None):
    return _template_aggregator('count({x})', 'count', len, item, wrt)
