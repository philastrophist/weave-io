from typing import List

import py2neo
from astropy.table import Table

from weaveio.basequery.common import FrozenQuery, UnexpectedResult, NotYetImplementedError


class FactorFrozenQuery(FrozenQuery):

    def _post_process(self, result):
        return result.to_data_frame()


class SingleFactorFrozenQuery(FactorFrozenQuery):
    """A single factor of a single hierarchy instance"""

    def _post_process(self, result: py2neo.Cursor):
        df = super(SingleFactorFrozenQuery, self)._post_process(result)
        if df.shape != (1, 1):
            raise UnexpectedResult(f"Query raised a shape of {df.shape} instead of (1, 1)")
        return df.iloc[0, 0]


class ColumnFactorFrozenQuery(FactorFrozenQuery):
    """A list of the same factor values for different hierarchy instances"""
    def _post_process(self, result: py2neo.Cursor):
        df = super(ColumnFactorFrozenQuery, self)._post_process(result)
        if df.shape[1] != 1:
            raise UnexpectedResult(f"Query raised {df.shape} instead of (..., 1)")
        return df.iloc[:, 0].tolist()



class TableFactorFrozenQuery(FactorFrozenQuery):
    """
    A matrix of different factors against different hierarchy instances
    This is only possible if the hierarchies each have only one of the factors
    """
    def __init__(self, handler, query, return_keys: List[str] = None, parent: 'FrozenQuery' = None):
        super().__init__(handler, query, parent)
        self.return_keys = return_keys

    def _post_process(self, result):
        df = super(TableFactorFrozenQuery, self)._post_process(result)
        if self.return_keys is None:
            return df.values
        return Table.from_pandas(df)

    def __getattr__(self, item):
        raise NotYetImplementedError


class RowFactorFrozenQuery(TableFactorFrozenQuery):
    """A list of different factors for one hierarchy"""
    def _post_process(self, result: py2neo.Cursor):
        df_or_array = super(RowFactorFrozenQuery, self)._post_process(result)
        assert len(df_or_array) == 1, f"Unexpected number of results ({len(df_or_array)} for {self}"
        return df_or_array[0]
