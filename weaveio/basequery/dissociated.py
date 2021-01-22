from functools import reduce
from operator import and_
from typing import Union

import networkx as nx
import py2neo
from astropy.table import Table

from weaveio.basequery.common import FrozenQuery
from weaveio.basequery.tree import Branch
from weaveio.writequery import CypherVariable, CypherQuery



class Dissociated(FrozenQuery):
    def __init__(self, handler, branch: Branch, variable: CypherVariable, parent: 'FrozenQuery' = None):
        super().__init__(handler, branch, parent)
        self.variable = variable

    def _apply_func(self, string, other: 'Dissociated' = None):
        if other is not None:
            if isinstance(other, Dissociated):  # now we have to align first
                aligned = self.branch.align(other.branch)
                inputs = {
                    'x': aligned.action.transformed_variables.get(self.variable, self.variable),
                    'y': aligned.action.transformed_variables.get(other.variable, other.variable)
                }
                # try:
                #     y = aligned.action.transformed_variables[other.variable]
                #     inputs = {'x': self.variable, 'y': y}
                # except KeyError:  # the alignment swapped them round
                #     x = aligned.action.transformed_variables[self.variable]
                #     inputs = {'y': other.variable, 'x': x}
            elif isinstance(other, FrozenQuery):
                raise TypeError(f"Cannot compare types {self.__class__} and {other.__class__}")
            else:
                aligned = self.branch.add_data(other)
                inputs = {'x': self.variable, 'y': aligned.current_variables[0]}
        else: # only one variable
            inputs = {'x': self.variable}
            aligned = self.branch
        newbranch = aligned.operate(string, **inputs)
        return Dissociated(self.handler, newbranch, newbranch.current_variables[-1], self)

    def __add__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} + {y}', other)

    def __sub__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} - {y}', other)

    def __mul__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} * {y}', other)

    def __truediv__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} / {y}', other)

    def __pow__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} ** {y}', other)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__

    def __neg__(self) -> 'Dissociated':
        return self._apply_func('-{x}')

    def __abs__(self) -> 'Dissociated':
        return self._apply_func('abs({x})')

    def __eq__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} = {y}', other)

    def __ne__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} <> {y}', other)

    def __lt__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} < {y}', other)

    def __le__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} <= {y}', other)

    def __gt__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} > {y}', other)

    def __ge__(self, other: Union['Dissociated', int, float]) -> 'Dissociated':
        return self._apply_func('{x} >= {y}', other)

    def _post_process(self, result: py2neo.Cursor, squeeze: bool = True) -> Table:
        df = result.to_data_frame()
        assert len(df.columns) == 1
        vs = df.iloc[:, 0].values
        if squeeze and len(vs) == 1:
            return vs[0]
        return vs

    def _prepare_query(self) -> CypherQuery:
        with super()._prepare_query() as query:
            return query.returns(self.variable)