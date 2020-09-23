from typing import Union, List, Dict, Any, Type
from pathlib import Path

from astropy.io import fits

from weaveio.data import Address
import pandas as pd
import numpy as np
from weaveio.hierarchy import File


class MasterStore:
    def __init__(self, index: pd.DataFrame):
        self.index = index

    def contains_index(self, index: pd.DataFrame):
        keys = ['fname', 'rowid']
        df = index[keys].isin(self.index[keys])
        return df.values

    def filter_by_index(self, index):
        keys = [i for i in index.columns if i not in ['fname', 'rowid']]
        filt = index[keys].isin(self.index[keys])
        return self[filt]

    def sort(self):
        self.index.sort_values(['fname', 'rowid'], inplace=True)
        self.data = self.data[self.index.index.values]

    def append(self, data, index: pd.DataFrame):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class MasterArray(MasterStore):
    pass

class MasterTable(MasterStore):
    pass

class MasterColumn(MasterStore):
    pass


class Product:
    @classmethod
    def index_from_query(cls, address: Address, ids: Dict[str, Any]) -> pd.DataFrame:
        d = {}
        for i in cls.indexable_by:
            try:
                value = address[i]
            except KeyError:
                try:
                    value = ids[i]
                except KeyError:
                    continue
            d[i] = value
        vectors = {k: v for k, v in d.items() if isinstance(v, (tuple, list))}
        scalars = {k: v for k, v in d.items() if not isinstance(v, (tuple, list))}
        l = max(map(len, vectors))
        assert all(len(v) == l for v in vectors.values())
        index = pd.DataFrame(vectors)
        for name, value in scalars.items():
            index[name] = value
        return index

    def append(self, data, filename: str, irows: List[int], ):


def _get_data_view(required_index: pd.DataFrame, files: List[File], data_name: str,
                   concatenation_constants, master_type):
    array = master_type(files[0].__class__, data_name, concatenation_constants)
    for ifile, file in enumerate(files):
        index = file.match_index(required_index)  # rows which match
        missing = ~array.contains_index(index)  # bool array of which ones we already have in the master array
        data = file.extract_data(data_name)  # all data
        array.append(data[index[missing].irows], index[missing], do_sort=False)  # add new rows
    array.sort()  # sort by file and irow such that files a,b,c are in order: a[0], a[1], b[0], b[2], c[1], c[100]
    return array.filter_by_index(required_index)  # boolean index array

def _get_product_view(factors, ids, files, product_name):
    product_type = files[0].basic_product_types[product_name]  # type: Type[Product]
    required_index = product_type.index_from_query(factors, ids)
    return get_data_view(required_index, files, product_name, product_type.concatenation_constants, product_type.master_type)

def get_product_view(factors: Address, ids: Dict[str, Any], files: List[File], product_name: str) -> Product:
    assert all(isinstance(f, files[0].__class__) for f in files)
    try:
        return _get_product_view(factors, ids, files, product_name)  # if you want the basic product
    except KeyError:
        product_type = files[0].composite_product_types[product_name]
        required = {name: _get_product_view(factors, ids, files, name) for name in product_type.required_product_names}
        return product_type(**required)