from typing import List

import numpy as np
import py2neo
from astropy.table import Table, Column

from weaveio.basequery.common import FrozenQuery, UnexpectedResult, NotYetImplementedError
from weaveio.basequery.dissociated import Dissociated
from weaveio.basequery.functions import OperableMixin
from weaveio.basequery.tree import Branch
from weaveio.writequery import CypherVariable, CypherQuery


class FactorFrozenQuery(Dissociated):
    def __init__(self, handler, branch: Branch, factors: List[str], factor_variables: List[CypherVariable],
                 plurals: List[bool], parent: FrozenQuery = None):
        super().__init__(handler, branch, parent)
        self.factors = factors
        self.factor_variables = factor_variables
        self.plurals = plurals

    def _prepare_query(self) -> CypherQuery:
        with super()._prepare_query() as query:
            return query.returns(*self.factor_variables)

    def __repr__(self):
        if isinstance(self.factors, tuple):
            factors = f'{self.factors}'
        else:
            factors = f'[{self.factors}]'
        return f'{self.parent}{factors}'

    def _post_process(self, result: py2neo.Cursor) -> Table:
        df = result.to_data_frame()
        table = Table.from_pandas(df)
        for colname, plural in zip(df.columns, self.plurals):
            if plural:
                if df.dtypes[colname] == 'O':
                    lengths = set(map(len, df[colname]))
                    if len(lengths) == 1:  # all the same length
                        table[colname] = Column(df[colname], name=colname, shape=lengths.pop(), length=len(df))
        return table


class SingleFactorFrozenQuery(FactorFrozenQuery, OperableMixin):
    """A single factor of a single hierarchy instance"""

    def _post_process(self, result: py2neo.Cursor):
        t = super(SingleFactorFrozenQuery, self)._post_process(result)
        if len(t) != 1 or len(t.colnames) != 1:
            raise UnexpectedResult(f"Query raised a shape of {(len(t), len(t.colnames))} instead of (1, 1)")
        return t[t.colnames[0]][0]


class ColumnFactorFrozenQuery(FactorFrozenQuery, OperableMixin):
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
    def __init__(self, handler, branch, factors, factor_variables, numbers, return_keys: List[str] = None, parent: 'FrozenQuery' = None):
        super().__init__(handler, branch, factors, factor_variables, numbers, parent)
        self.return_keys = return_keys

    def _prepare_query(self) -> CypherQuery:
        with super()._prepare_query() as query:
            variables = {k: v for k, v in zip(self.return_keys, self.factor_variables)}
            return query.returns(**variables)

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
