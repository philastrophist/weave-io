from typing import Callable

import numpy as np

from .utilities import mask_infs
from .objects import BaseQuery, AttributeQuery


def _template_operator(string_op: str, name: str, python_func: Callable, item: BaseQuery, remove_infs=True, *args, **kwargs):
    if not isinstance(item, AttributeQuery):
        return python_func(item, *args, **kwargs)
    if remove_infs:
        string_op = string_op.format(mask_infs('{0}'))
    return item._perform_arithmetic(string_op, name)


def sign(item, *args, **kwargs):
    return _template_operator('sign({0})', 'sign', np.sign, item, remove_infs=True, args=args, kwargs=kwargs)


def exp(item, *args, **kwargs):
    return _template_operator('exp({0})', 'exp', np.exp, item, remove_infs=True, args=args, kwargs=kwargs)


def log(item, *args, **kwargs):
    return _template_operator('log({0})', 'log', np.log, item, remove_infs=True, args=args, kwargs=kwargs)


def log10(item, *args, **kwargs):
    return _template_operator('log10({0})', 'log10', np.log10, item, remove_infs=True, args=args, kwargs=kwargs)


def sqrt(item, *args, **kwargs):
    return _template_operator('sqrt({0})', 'sqrt', np.sqrt, item, remove_infs=True, args=args, kwargs=kwargs)

def ismissing(item):
    return _template_operator('{0} is null' ,'isnull', lambda x: x is None, item, remove_infs=False)

def isnan(item):
    return _template_operator('{0} == 1.0/0.0', 'isnan', np.isnan, item, remove_infs=False)
