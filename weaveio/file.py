from pathlib import Path
from typing import Union, Dict, List

from astropy.io import fits

from weaveio.graph import Graph
from weaveio.hierarchy import Hierarchy


class File(Hierarchy):
    is_template = True
    idname = 'fname'
    match_pattern = '*.file'
    hdus = {}
    produces = []

    def __init__(self, fname, **kwargs):
        super().__init__(tables=None, fname=str(fname), **kwargs)

    @classmethod
    def match_file(cls, directory: Union[Path, str], fname: Union[Path, str], graph: Graph):
        """Returns True if the given fname in a given directory can be read by this class of file hierarchy object"""
        fname = Path(fname)
        return fname.match(cls.match_pattern)

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str]) -> 'File':
        raise NotImplementedError

    @classmethod
    def read_hdus(cls, directory: Union[Path, str], fname: Union[Path, str],
                  **hierarchies: Union[Hierarchy, List[Hierarchy]]):
        path = Path(directory) / Path(fname)
        file = cls(fname, **hierarchies)
        hdus = [i for i in fits.open(path)]
        if len(hdus) != len(cls.hdus):
            raise TypeError(f"Class {cls} asserts there are {len(cls.hdus)} HDUs ({list(cls.hdus.keys())})"
                            f" whereas {path} has {len(hdus)} ({[i.name for i in hdus]})")
        hduinstances = {}
        for i, ((hduname, hduclass), hdu) in enumerate(zip(cls.hdus.items(), hdus)):
            hduinstances[hduname] = hduclass.from_hdu(hdu, i, file)
        return hduinstances, file

    def read_product(self, product_name):
        self.build_index()
        return getattr(self, f'read_{product_name}')()


class HDU(Hierarchy):
    is_template = True
    parents = [File]
    factors = ['extn']
    identifier_builder = ['file', 'extn']
    binaries = ['header', 'data']
    concatenation_constants = None

    @classmethod
    def _from_hdu(cls, hdu):
        return {}

    @classmethod
    def from_hdu(cls, hdu, extn, file):
        input_dict = cls._from_hdu(hdu)
        input_dict[cls.parents[0].singular_name] = file
        input_dict['extn'] = extn
        if cls.concatenation_constants is not None:
            if len(cls.concatenation_constants):
                for c in cls.concatenation_constants:
                    if c not in input_dict:
                        input_dict[c] = hdu.header[c]
                input_dict['concatenation_constants'] = cls.concatenation_constants
        return cls(**input_dict)


class PrimaryHDU(HDU):
    is_template = True
    binaries = ['header']
    concatenation_constants = []


class BaseDataHDU(HDU):
    concatenation_constants = ['ncols']
    factors = HDU.factors + ['nrows', 'ncols']


class TableHDU(BaseDataHDU):
    is_template = True
    concatenation_constants = ['columns']

    @classmethod
    def _from_hdu(cls, hdu):
        input_dict = BaseDataHDU._from_hdu(hdu)
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
    is_template = True

    @classmethod
    def _from_hdu(cls, hdu):
        input_dict = BaseDataHDU._from_hdu(hdu)
        if hdu.data is None:
            input_dict['nrows'], input_dict['ncols'] = 0, 0
        else:
            input_dict['nrows'], input_dict['ncols'] = hdu.data.shape
        return input_dict


class SpectralBlockHDU(BinaryHDU):
    is_template = True
    concatenation_constants = ['naxis1', 'naxis2']


class SpectralRowableBlock(BinaryHDU):
    is_template = True
    concatenation_constants = ['naxis1', 'crval1', 'cunit1', 'cd1_1']
