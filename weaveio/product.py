from pathlib import Path
from typing import Union, List

from astropy.io import fits

from weaveio.hierarchy import File


class Product:
    idname = 'filehdu'  # e.g. "r101249.fits[1]"
    indexers = []

    def __init__(self, file_hierarchy: File, index: Union[slice,List[int],int] = None):
        self.file_hierarchy = file_hierarchy
        self.index = index

    def read(self):
        raise NotImplementedError

    @classmethod
    def from_files(cls, *files):
        raise NotImplementedError

class CompositeProduct:
    pass


class SpectraBlock(Product):

    @classmethod
    def from_files(cls, index, *files):
        if len(files) == 1:
            return SpectraBlock(files[0], index)
        elif len(files)


class UnalignedCompositeSpectraBlock(SpectraBlock, CompositeProduct):
    pass


class AlignedCompositeSpectraBlock(SpectraBlock, CompositeProduct):
    pass


class Spectrum(SpectraBlock, SingularProduct):
    pass