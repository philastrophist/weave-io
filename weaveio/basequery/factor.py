from typing import List

import py2neo
from astropy.table import Table

from weaveio.basequery.common import FrozenQuery, UnexpectedResult, NotYetImplementedError


class FactorFrozenQuery(FrozenQuery):

    def _post_process(self, result: py2neo.Cursor):
        return result.to_ndarray()


class SingleFactorFrozenQuery(FactorFrozenQuery):
    """A single factor of a single hierarchy instance"""

    def _post_process(self, result: py2neo.Cursor):
        array = super()._post_process(result)
        if array.shape != (1, 1):
            raise UnexpectedResult(f"Query raised a shape of {array.shape} instead of (1, 1)")
        return array[0, 0]


class ColumnFactorFrozenQuery(FactorFrozenQuery):
    """A list of the same factor values for different hierarchy instances"""
    def _post_process(self, result: py2neo.Cursor):
        array = super()._post_process(result)
        if array.shape[1] != 1:
            raise UnexpectedResult(f"Query raised {array.shape} instead of (..., 1)")
        return array[:, 0]


class RowFactorFrozenQuery(FactorFrozenQuery):
    """A list of different factors for one hierarchy"""

    def __init__(self, handler, query, return_keys: List[str] = None, parent: 'FrozenQuery' = None):
        super().__init__(handler, query, parent)
        self.return_keys = return_keys

    def _post_process(self, result: py2neo.Cursor):
        if self.return_keys is not None:
            return result.data()[0]
        df =  result.to_data_frame()
        df.columns = self.return_keys
        return Table.from_pandas(df)

    def __getattr__(self, item):
        raise NotYetImplementedError


class TableFactorFrozenQuery(RowFactorFrozenQuery):
    """
    A matrix of different factors against different hierarchy instances
    This is only possible if the hierarchies each have only one of the factors
    """
