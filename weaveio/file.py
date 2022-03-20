import re
from pathlib import Path
from typing import Union, List, Tuple, Dict

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU

from weaveio.graph import Graph
from weaveio.hierarchy import Hierarchy


class File(Hierarchy):
    is_template = True
    idname = 'fname'
    factors = ['path']
    match_pattern = '*.file'
    antimatch_pattern = '^$'
    hdus = {}
    produces = []
    recommended_batchsize = None

    def open(self):
        try:
            return fits.open(self.data.rootdir / self.fname)
        except AttributeError:
            return fits.open(self.fname)

    def __init__(self, fname, path, **kwargs):
        # fname should only be the name of a file, not a relative/absolute path
        # path should be the path to the file relative to the rootdir (which is not stored)
        # if fname is a path containing the filename
        fname = Path(fname)
        path = Path(path)
        if len(fname.parents) == 1 and fname.parents[0] == Path('.'):  # if fname is just a filename
            fname = fname.name
            path = path
        else:
            try:
                fname = fname.relative_to(path)  # throws error if fname is not under path, which indicates they are separated already
            except ValueError:
                path = fname.parents[0]
                fname = fname.name
            else:
                path = fname.parents[0]  # get the path again
                fname = fname.name


        super().__init__(tables=None, fname=str(fname), path=str(path), **kwargs)

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
        abspath = Path(directory) / Path(fname)
        file = cls(fname, directory, **hierarchies)
        hdus = [i for i in fits.open(abspath)]
        if len(hdus) != len(cls.hdus):
            raise TypeError(f"Class {cls} asserts there are {len(cls.hdus)} HDUs ({list(cls.hdus.keys())})"
                            f" whereas {abspath} has {len(hdus)} ({[i.name for i in hdus]})")
        hduinstances = {}
        for i, ((hduname, hduclass), hdu) in enumerate(zip(cls.hdus.items(), hdus)):
            hduinstances[hduname] = hduclass.from_hdu(hduname, hdu, i, file)
        return hduinstances, file, hdus

    def read_product(self, product_name):
        self.build_index()
        return getattr(self, f'read_{product_name}')()

    @classmethod  # fudgy
    def without_creation(cls, **kwargs):
        if 'path' not in kwargs:
            kwargs['path'] = '.'
        return super().without_creation(**kwargs)


class HDU(Hierarchy):
    is_template = True
    parents = [File]
    factors = ['sourcefile', 'extn', 'name']
    identifier_builder = ['sourcefile', 'extn', 'name']
    binaries = ['header', 'data']
    concatenation_constants = None

    @classmethod
    def _from_hdu(cls, hdu):
        return {}

    @classmethod
    def from_hdu(cls, name, hdu, extn, file):
        input_dict = cls._from_hdu(hdu)
        input_dict[cls.parents[0].singular_name] = file
        input_dict['extn'] = extn
        input_dict['sourcefile'] = file.path
        input_dict['name'] = name
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
    is_template = True
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
    concatenation_constants = ['naxis1', 'crval1', 'cd1_1']
