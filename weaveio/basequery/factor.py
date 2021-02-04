from pathlib import Path
from typing import List

import py2neo
from astropy.io import fits
from astropy.table import Table, Column
import pandas as pd
import numpy as np
from tqdm import tqdm

from weaveio.basequery.common import FrozenQuery
from weaveio.basequery.dissociated import Dissociated
from weaveio.basequery.tree import Branch
from weaveio.writequery import CypherVariable, CypherQuery


def replace_with_data(row, files):
    hdus = files[row['sourcefile']]
    hdu = hdus[row['extn']]
    if not pd.isnull(row['column_name']):
        table = Table(hdu.data)
        table = table[row['column_name']]
    else:
        table = hdu.data
    if not pd.isnull(row['index']):
        return table[row['index']]
    return table


class FactorFrozenQuery(Dissociated):
    def __init__(self, handler, branch: Branch, factors: List[str], factor_variables: List[CypherVariable],
                 plurals: List[bool], is_products: List[bool], parent: FrozenQuery = None):
        super().__init__(handler, branch, factor_variables[0], parent)
        self.factors = factors
        self.factor_variables = factor_variables
        self.plurals = plurals
        self.is_products = is_products

    def _filter_by_boolean(self, boolean_filter: 'Dissociated'):
        new = self._make_filtered_branch(boolean_filter)
        return self.__class__(self.handler, new, self.factors, self.factor_variables, self.plurals,
                              self.is_products, self)

    def _prepare_query(self) -> CypherQuery:
        with super()._prepare_query() as query:
            return query.returns(*self.factor_variables)

    def __repr__(self):
        if isinstance(self.factors, tuple):
            factors = f'{self.factors}'
        else:
            factors = f'[{self.factors}]'
        return f'{self.parent}{factors}'

    def _parse_products(self, df: pd.DataFrame):
        """
        Take a pandas dataframe and replace the structure of ['sourcefile', 'extn', 'index', 'column_name']
        with the actual data
        """
        product_columns = []
        for colname, is_product in zip(df.columns, self.is_products):
            if is_product:
                df[colname] = df[colname].apply(lambda x: [None]*4 if len(x) == 0 else x[0])
                split = df[colname].apply(pd.Series)
                split.columns = ['sourcefile', 'extn', 'index', 'column_name']
                split.index.name = 'rown'
                split['colname'] = colname
                split.set_index('colname', append=True, inplace=True)
                product_columns.append(split)
        if len(product_columns):
            sourcefiles = {}
            concat = pd.concat(product_columns)
            for fname in tqdm(concat.sourcefile.drop_duplicates(), desc='reading fits files'):
                path = Path(self.handler.data.rootdir) / fname
                sourcefiles[fname] = fits.open(path)
            data = concat.apply(replace_with_data, files=sourcefiles, axis='columns').sort_index(level=0)  # type: pd.Series
            for name, group in data.groupby('colname'):
                df[name] = group.values
        return df

    def _post_process(self, result: py2neo.Cursor, squeeze: bool = True) -> Table:
        df = self._parse_products(result.to_data_frame())
        # replace lists with arrays
        for c in df.columns:
            if df.dtypes[c] == 'O' and not isinstance(df[c].iloc[0], str):
                df[c] = df[c].apply(np.asarray)
        table = Table.from_pandas(df)
        for colname, plural, is_product in zip(df.columns, self.plurals, self.is_products):
            if plural or is_product or df.dtypes[colname] == 'O':
                shapes = set(map(np.shape, df[colname]))
                if len(shapes) == 1:  # all the same length
                    table[colname] = Column(np.stack(df[colname].values), name=colname, shape=shapes.pop(), length=len(df))
        if len(table) == 1 and squeeze:
            table = table[0]
        if len(table.colnames) == 1 and squeeze:
            table = table[table.colnames[0]]
        return table


class SingleFactorFrozenQuery(FactorFrozenQuery):
    def __init__(self, handler, branch: Branch, factor: str, factor_variable: CypherVariable, is_product: bool, parent: FrozenQuery = None):
        super().__init__(handler, branch, [factor], [factor_variable], [False], [is_product], parent)

    def _post_process(self, result: py2neo.Cursor, squeeze: bool = True) -> Table:
        row = super()._post_process(result, squeeze)
        if isinstance(row, Column):
            return row.data
        return row


class TableFactorFrozenQuery(FactorFrozenQuery):
    """
    A matrix of different factors against different hierarchy instances
    This is only possible if the hierarchies each have only one of the factors
    """
    def __init__(self, handler, branch, factors, factor_variables, plurals, is_products,
                 return_keys: List[str] = None, parent: 'FrozenQuery' = None):
        super().__init__(handler, branch, factors, factor_variables, plurals, is_products, parent)
        self.return_keys = return_keys

    def _prepare_query(self) -> CypherQuery:
        with super()._prepare_query() as query:
            variables = {k: v for k, v in zip(self.return_keys, self.factor_variables)}
            return query.returns(**variables)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                i = self.factors.index(item)
            except ValueError:
                raise KeyError(f"{item} is not a factor contained within {self}. Only {self.factors} are accessible.")
            else:
                return SingleFactorFrozenQuery(self.handler, self.branch, item, self.factor_variables[i], self.is_products[i], self)
        else:
            return super(TableFactorFrozenQuery, self).__getitem__(item)
