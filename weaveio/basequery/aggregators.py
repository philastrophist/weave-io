from typing import Callable

import numpy as np

from .common import FrozenQuery
from .dissociated import Dissociated
from .functions import _convert
from ..data import Data

__all__ = ['sum', 'max', 'min', 'all', 'any', 'count', 'std']


python_any = any
python_all = all
python_max = max
python_min = min
python_sum = sum


def _template_aggregator(string_function, name, normal: Callable, item: Dissociated, wrt: FrozenQuery = None,
                         remove_infs: bool = True, convert_to_float: bool = False, args=None, kwargs=None):
    if not isinstance(item, FrozenQuery):
        if wrt is not None:
            args = (wrt, ) + args
        return normal(item, *args, **kwargs)
    elif isinstance(wrt, Data) or wrt is None:
        branch = item.branch.handler.entry
    else:
        branch = wrt.branch
    itembranch, itemvariable = _convert(item.branch, item.variable, remove_infs, convert_to_float)
    new = branch.aggregate(string_function, itemvariable, itembranch, namehint=name)
    return Dissociated(item.handler, new, new.action.target)


def sum(item, wrt=None, *args, **kwargs):
    return _template_aggregator('sum({x})', 'sum', python_sum, item, wrt, convert_to_float=True, args=args, kwargs=kwargs)


def max(item, wrt=None, *args, **kwargs):
    return _template_aggregator('max({x})', 'max', python_max, item, wrt, convert_to_float=True, args=args, kwargs=kwargs)


def min(item, wrt=None, *args, **kwargs):
    return _template_aggregator('min({x})', 'min', python_min, item, wrt, convert_to_float=True, args=args, kwargs=kwargs)


def all(item, wrt=None, *args, **kwargs):
    return _template_aggregator('all(i in {x} where i)', 'all', python_all, item, wrt, args=args, kwargs=kwargs)


def any(item, wrt=None, *args, **kwargs):
    return _template_aggregator('any(i in {x} where i)', 'any', python_any, item, wrt, args=args, kwargs=kwargs)


def count(item, wrt=None, *args, **kwargs):
    return _template_aggregator('count({x})', 'count', len, item, wrt, args=args, kwargs=kwargs)


def std(item, wrt=None, *args, **kwargs):
    return _template_aggregator('stDev({x})', 'std', np.std, item, wrt, convert_to_float=True, args=args, kwargs=kwargs)
