from typing import List, Union

import numpy as np
import py2neo
from astropy.table import Table, Column, hstack
from astropy.utils.data_info import ParentDtypeInfo

from weaveio.basequery.common import FrozenQuery, UnexpectedResult, NotYetImplementedError
from weaveio.basequery.query import FullQuery
from weaveio.utilities import quote


class FactorFrozenQuery(FrozenQuery):
    def __init__(self, handler, branch: FullQuery, factors, numbers: List[Union[int, None]], parent: 'FrozenQuery' = None):
        super().__init__(handler, branch, parent)
        self.factors = factors
        self.numbers = numbers

    def __repr__(self):
        if isinstance(self.factors, tuple):
            factors = f'{self.factors}'
        else:
            factors = f'[{self.factors}]'
        return f'{self.parent}{factors}'

    def _post_process(self, result: py2neo.Cursor) -> Table:
        coldata = list(zip(*result.to_table()))
        columns = []
        for i, (name, number, data) in enumerate(zip(self.factors, self.numbers, map(list, coldata))):
            if number is None:
                #  make an unstructured list
                shape = len(data)
                dtype = object
                data[0].append('placeholder')  # to force astropy to pay attention!!!
                data.append(['placeholder'])
            elif number == 0:
                #  make a single scalar column
                shape = len(data)
                dtype = None
            else:
                # make a shaped vector Column
                shape = (len(data), number)
                dtype = None
            column = Column(data, name=name, shape=shape, length=len(data), dtype=dtype)
            if number is None:
                del column[0][-1]  # delete the placeholder
                column = column[:-1]
            columns.append(column)
        table = hstack(columns)
        return table


class SingleFactorFrozenQuery(FactorFrozenQuery):
    """A single factor of a single hierarchy instance"""

    def _post_process(self, result: py2neo.Cursor):
        t = super(SingleFactorFrozenQuery, self)._post_process(result)
        if len(t) != 1 or len(t.colnames) != 1:
            raise UnexpectedResult(f"Query raised a shape of {(len(t), len(t.colnames))} instead of (1, 1)")
        return t[t.colnames[0]][0]


class ColumnFactorFrozenQuery(FactorFrozenQuery):
    """A list of the same factor values for different hierarchy instances"""
    def _post_process(self, result: py2neo.Cursor):
        t = super(ColumnFactorFrozenQuery, self)._post_process(result)
        if len(t.colnames) != 1:
            raise UnexpectedResult(f"Query raised {len(t.colnames)} instead of (..., 1)")
        return t[t.colnames[0]].data


class TableFactorFrozenQuery(FactorFrozenQuery):
    """
    A matrix of different factors against different hierarchy instances
    This is only possible if the hierarchies each have only one of the factors
    """
    def __init__(self, handler, branch, factors, numbers, return_keys: List[str] = None, parent: 'FrozenQuery' = None):
        super().__init__(handler, branch, factors, numbers, parent)
        self.return_keys = return_keys

    def _post_process(self, result):
        t = super(TableFactorFrozenQuery, self)._post_process(result)
        if self.return_keys is None:
            t = np.asarray([t[c].data for c in t.colnames]).T  # without column names
        return t

    def __getattr__(self, item):
        raise NotYetImplementedError


class RowFactorFrozenQuery(TableFactorFrozenQuery):
    """A list of different factors for one hierarchy"""
    def _post_process(self, result: py2neo.Cursor):
        df_or_array = super(RowFactorFrozenQuery, self)._post_process(result)
        assert len(df_or_array) == 1, f"Unexpected number of results ({len(df_or_array)} for {self}"
        return df_or_array[0]
