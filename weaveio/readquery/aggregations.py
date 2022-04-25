from typing import Callable, TYPE_CHECKING

import numpy as np

from .utilities import mask_infs
from .base import BaseQuery
if TYPE_CHECKING:
    pass

__all__ = ['sum', 'max', 'min', 'mean', 'std', 'count', 'any', 'all']

python_any = any
python_all = all
python_max = max
python_min = min
python_sum = sum


def _template_aggregator(string_op, predicate, python_func: Callable, item: BaseQuery, wrt: BaseQuery = None,
                         remove_infs: bool = True, args=None, kwargs=None):
    try:
        if remove_infs:
            string_op = string_op.format(mask_infs('{0}'))
        return item._aggregate(wrt, string_op, predicate)
    except AttributeError:
        return python_func(item, *args, **kwargs)


def sum(item, wrt=None, *args, **kwargs):
    return _template_aggregator('sum', False, python_sum, item, wrt, args=args, kwargs=kwargs)


def max(item, wrt=None, *args, **kwargs):
    return _template_aggregator('max', False, python_max, item, wrt, args=args, kwargs=kwargs)


def min(item, wrt=None, *args, **kwargs):
    return _template_aggregator('min', False, python_min, item, wrt, args=args, kwargs=kwargs)


def count(item, wrt=None, *args, **kwargs):
    return _template_aggregator('count', False, len, item, wrt, args=args, kwargs=kwargs)


def std(item, wrt=None, *args, **kwargs):
    return _template_aggregator('stDev', False, np.std, item, wrt, args=args, kwargs=kwargs)


def mean(item, wrt=None, *args, **kwargs):
    return _template_aggregator('avg', False, np.mean, item, wrt, args=args, kwargs=kwargs)

#predicates

def all(item, wrt=None, *args, **kwargs):
    return _template_aggregator('all', True, python_all, item, wrt, args=args, kwargs=kwargs)


def any(item, wrt=None, *args, **kwargs):
    return _template_aggregator('any', True, python_any, item, wrt, args=args, kwargs=kwargs)

