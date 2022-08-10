from typing import Callable
from math import floor, ceil

import numpy as np

from .utilities import mask_infs
from .objects import AttributeQuery
from .base import BaseQuery


def _template_operator(string_op: str, name: str, item: BaseQuery, python_func: Callable = None,
                       remove_infs=True, in_dtype=None, out_dtype=None, *args, **kwargs):
    if not isinstance(item, AttributeQuery):
        if python_func is None:
            raise NotImplementedError(f"{name} is not implemented for {type(item)}")
        return python_func(item, *args, **kwargs)
    if remove_infs:
        string_op = string_op.format(mask_infs('{0}'))
    return item._perform_arithmetic(string_op, name, expected_dtype=in_dtype, returns_dtype=out_dtype)


def sign(item, *args, **kwargs):
    return _template_operator('sign({0})', 'sign', item, np.sign, remove_infs=True, out_dtype='float', args=args, kwargs=kwargs)


def exp(item, *args, **kwargs):
    return _template_operator('exp({0})', 'exp', item, np.exp, remove_infs=True, out_dtype='float', args=args, kwargs=kwargs)


def log(item, *args, **kwargs):
    return _template_operator('log({0})', 'log', item, np.log, remove_infs=True, out_dtype='float', args=args, kwargs=kwargs)


def log10(item, *args, **kwargs):
    return _template_operator('log10({0})', 'log10', item, np.log10, remove_infs=True, out_dtype='float', args=args, kwargs=kwargs)


def sqrt(item, *args, **kwargs):
    return _template_operator('sqrt({0})', 'sqrt', item, np.sqrt, remove_infs=True, out_dtype='float', args=args, kwargs=kwargs)

def ismissing(item):
    return _template_operator('{0} is null', 'isnull', item, lambda x: x is None, remove_infs=False, out_dtype='boolean')
isnull = ismissing

def isnan(item):
    return _template_operator('{0} == 1.0/0.0', 'isnan', item, np.isnan, remove_infs=False, out_dtype='boolean')
