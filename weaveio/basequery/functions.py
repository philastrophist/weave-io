from functools import partial
from typing import Callable

import numpy as np

from .common import FrozenQuery
from .dissociated import Dissociated
from .tree import Branch
from ..writequery import CypherVariable

__all__ = ['abs', 'round', 'ceil', 'floor', 'random', 'sign', 'exp', 'log', 'log10', 'sqrt']

python_abs = abs
python_round = round


def _convert(branch: Branch, variable: CypherVariable, remove_infs: bool = True, convert_to_float: bool = False):
    if convert_to_float:
        branch = branch.operate('CASE WHEN {z} = true THEN 1 WHEN {z} = false THEN 0 ELSE {z} END', 'convert_to_float', z=variable)
        variable = branch.action.output_variables[0]
    if remove_infs:
        branch = branch.operate("CASE WHEN {variable} > apoc.math.maxLong() THEN null ELSE {variable} END", 'remove_infs', variable=variable)
        variable = branch.action.output_variables[0]
    return branch, variable


def _template_operator(string_function, name, normal: Callable, item: Dissociated, remove_infs: bool = True, convert_to_float: bool = False, args=None, kwargs=None):
    if not isinstance(item, FrozenQuery):
        return normal(item, *args, **kwargs)
    branch, variable = _convert(item.branch, item.variable, remove_infs, convert_to_float)
    branch = branch.operate(string_function, x=variable, namehint=name)
    return Dissociated(item.handler, branch, branch.action.output_variables[0])


def abs(item, *args, **kwargs):
    return _template_operator('abs({x})', 'abs', python_abs, item, convert_to_float=True, args=args, kwargs=kwargs)


def ceil(item, *args, **kwargs):
    return _template_operator('ceil({x})', 'ceil', np.ceil, item, convert_to_float=True, args=args, kwargs=kwargs)


def floor(item, *args, **kwargs):
    return _template_operator('floor({x})', 'floor', np.floor, item, convert_to_float=True, args=args, kwargs=kwargs)


def random(item, *args, **kwargs):
    return _template_operator('random({x})', 'random', np.uniform, item, convert_to_float=True, args=args, kwargs=kwargs)


def round(item, precision=0, *args, **kwargs):
    return _template_operator('round({x})', 'round', partial(python_round, precision), item, convert_to_float=True, args=args, kwargs=kwargs)


def sign(item, *args, **kwargs):
    return _template_operator('sign({x})', 'sign', np.sign, item, convert_to_float=True, args=args, kwargs=kwargs)


def exp(item, *args, **kwargs):
    return _template_operator('exp({x})', 'exp', np.exp, item, convert_to_float=True, args=args, kwargs=kwargs)


def log(item, *args, **kwargs):
    return _template_operator('log({x})', 'log', np.log, item, convert_to_float=True, args=args, kwargs=kwargs)


def log10(item, *args, **kwargs):
    return _template_operator('log10({x})', 'log10', np.log10, item, convert_to_float=True, args=args, kwargs=kwargs)


def sqrt(item, *args, **kwargs):
    return _template_operator('sqrt({x})', 'sqrt', np.sqrt, item, convert_to_float=True, args=args, kwargs=kwargs)
