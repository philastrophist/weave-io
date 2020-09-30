from functools import reduce
from typing import Union, List, Dict, Any, Type, Iterable
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from astropy.table import vstack, Table as AstropyTable


def header2table(header):
    keys = set(header.keys())
    data = AstropyTable([[header[k]] for k in keys], names=list(keys), meta=header.comments)
    data.comments = data.meta
    return data


class Product:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    @classmethod
    def concatenate(cls, *products):
        return cls(cls.concatenate_data(*products), cls.concatenate_index(*products))

    @classmethod
    def concatenate_index(cls, *products):
        return  pd.concat([p.index for p in products])

    @classmethod
    def concatenate_data(cls, *products):
        raise NotImplementedError

    def sort(self):
        raise NotImplementedError

    def __getitem__(self, item: pd.Series) -> 'Product':
        if isinstance(item, pd.Series):
            item = item.values
        return self.__class__(self.data[item], self.index[item])


class Array(Product):
    def __init__(self, data, index):
        super().__init__(np.asarray(data), index)

    @classmethod
    def concatenate_data(cls, *products):
        return np.stack([p.data for p in products])


class Table(Product):
    def __init__(self, data, index):
        super().__init__(AstropyTable(data), index)

    @classmethod
    def concatenate_data(cls, *products):
        return vstack([p.data for p in products])


class Header(Product):
    @classmethod
    def concatenate(cls, *products):
        return Table.concatenate([Table(header2table(p), p.index) for p in products])


def get_product(files: List['File'], product_name: str, product_index: pd.DataFrame) -> Product:
    product_type = files[0].products[product_name]
    for f in files:
        if f.products[product_name] is not product_type:
            raise TypeError(f"Stacking the {product_name} product from this query is not supported since {product_name} implies different types of products")
        if not f.is_concatenatable_with(files[0], product_name):
            raise TypeError(f"Stacking the {product_name} product from this query is not supported since they are unequal in concatenation_constants")
    products = []
    for f in files:
        product = f.read_product(product_name)
        if product_index is not None:
            filt = f.match_index(product_index)
            product = product[filt]
        products.append(product)
    return product_type.concatenate(*products).data