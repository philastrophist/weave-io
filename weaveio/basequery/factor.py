from typing import List

import py2neo
from astropy.table import Table, Column

from weaveio.basequery.common import FrozenQuery, NotYetImplementedError
from weaveio.basequery.dissociated import Dissociated
from weaveio.basequery.tree import Branch
from weaveio.writequery import CypherVariable, CypherQuery


class FactorFrozenQuery(Dissociated):
    def __init__(self, handler, branch: Branch, factors: List[str], factor_variables: List[CypherVariable],
                 plurals: List[bool], parent: FrozenQuery = None):
        super().__init__(handler, branch, factor_variables[0], parent)
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

    def _post_process(self, result: py2neo.Cursor, squeeze: bool = True) -> Table:
        df = result.to_data_frame()
        table = Table.from_pandas(df)
        for colname, plural in zip(df.columns, self.plurals):
            if plural:
                if df.dtypes[colname] == 'O':
                    lengths = set(map(len, df[colname]))
                    if len(lengths) == 1:  # all the same length
                        table[colname] = Column(df[colname], name=colname, shape=lengths.pop(), length=len(df))
        if len(table) == 1 and squeeze:
            table = table[0]
        if len(table.colnames) == 1 and squeeze:
            table = table[table.colnames[0]]
        return table


class SingleFactorFrozenQuery(FactorFrozenQuery):
    def __init__(self, handler, branch: Branch, factor: str, factor_variable: CypherVariable, plural: bool, parent: FrozenQuery = None):
        super().__init__(handler, branch, [factor], [factor_variable], [plural], parent)

    def _post_process(self, result: py2neo.Cursor, squeeze: bool = True) -> Table:
        t = super(SingleFactorFrozenQuery, self)._post_process(result, squeeze)
        return t


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

    def _post_process(self, result, squeeze: bool = True):
        t = super(TableFactorFrozenQuery, self)._post_process(result, squeeze)
        return t

    def __getattr__(self, item):
        raise NotYetImplementedError
