import re
from pathlib import Path
from typing import Union, List, Tuple, Dict

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU

from weaveio.graph import Graph
from weaveio.hierarchy import Hierarchy, Multiple


class File(Hierarchy):
    is_template = True
    idname = 'fname'
    match_pattern = '*.file'
    antimatch_pattern = '^$'
    recommended_batchsize = None

    def open(self):
        try:
            return fits.open(self.data.rootdir / self.fname)
        except AttributeError:
            return fits.open(self.fname)

    def __init__(self, fname, **kwargs):
        super().__init__(tables=None, fname=str(fname), **kwargs)

    @classmethod
    def get_batches(cls, path, batch_size):
        if batch_size is None:
            return [slice(None, None)]
        n = cls.length(path)
        return (slice(i, i + batch_size) for i in range(0, n, batch_size))

    @classmethod
    def match_file(cls, directory: Union[Path, str], fname: Union[Path, str], graph: Graph):
        """Returns True if the given fname in a given directory can be read by this class of file hierarchy object"""
        fname = Path(fname)
        return re.match(cls.match_pattern, fname.name, re.IGNORECASE) and not re.match(cls.antimatch_pattern, fname.name, re.IGNORECASE)

    @classmethod
    def match_files(cls,  directory: Union[Path, str], graph: Graph):
        """Returns all matching files within a directory"""
        return (f for f in Path(directory).rglob('*.fit*') if cls.match_file(directory, f, graph))

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str], slc: slice = None) -> 'File':
        raise NotImplementedError

    @classmethod
    def read_hdus(cls, directory: Union[Path, str], fname: Union[Path, str],
                  **hierarchies: Union[Hierarchy, List[Hierarchy]]) -> Tuple[Dict[str,'HDU'], 'File', List[_BaseHDU]]:
        path = Path(directory) / Path(fname)
        file = cls(fname, **hierarchies)
        hdus = [i for i in fits.open(path)]
        if len(hdus) != len(cls.hdus):
            raise TypeError(f"Class {cls} asserts there are {len(cls.hdus)} HDUs ({list(cls.hdus.keys())})"
                            f" whereas {path} has {len(hdus)} ({[i.name for i in hdus]})")
        hduinstances = {}
        for i, ((hduname, hduclass), hdu) in enumerate(zip(cls.hdus.items(), hdus)):
            hduinstances[hduname] = hduclass.from_hdu(hduname, hdu, i, file)
        return hduinstances, file, hdus

    def read_product(self, product_name):
        self.build_index()
        return getattr(self, f'read_{product_name}')()


class HDU(Hierarchy):
    is_template = True
    parents = [File]
    factors = ['extn', 'name']
    identifier_builder = ['file', 'extn', 'name']
    products = ['header', 'data']

    @classmethod
    def from_hdu(cls, name, hdu, extn, file):
        input_dict = cls._from_hdu(hdu)
        input_dict[cls.parents[0].singular_name] = file
        input_dict['extn'] = extn
        input_dict['sourcefile'] = file.fname
        input_dict['name'] = name
        return cls(**input_dict)


class BaseDataHDU(HDU):
    is_template = True
    factors = HDU.factors + ['nrows', 'ncols']


class PrimaryHDU(HDU):
    products = ['header']


class TableHDU(BaseDataHDU):
    factors = BaseDataHDU.factors + ['columns']

    @classmethod
    def _from_hdu(cls, hdu):
        input_dict = {}
        if hdu.data is None:
            input_dict['columns'] = []
            input_dict['nrows'] = 0
            input_dict['ncols'] = 0
        else:
            colnames = [str(i) for i in hdu.data.names]
            input_dict['columns'] = colnames
            input_dict['nrows'], input_dict['ncols'] = hdu.data.shape[0], len(colnames)
        return input_dict


class BinaryHDU(BaseDataHDU):
    @classmethod
    def _from_hdu(cls, hdu):
        input_dict = {}
        if hdu.data is None:
            input_dict['nrows'], input_dict['ncols'] = 0, 0
        else:
            input_dict['nrows'], input_dict['ncols'] = hdu.data.shape
        return input_dict
