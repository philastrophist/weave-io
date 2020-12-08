from pathlib import Path
from typing import Union

from weaveio.hierarchy import Hierarchy


class File(Hierarchy):
    is_template = True
    idname = 'fname'
    match_pattern = '*.file'

    def __init__(self, fname, **kwargs):
        super().__init__(tables=None, fname=str(fname), **kwargs)

    @classmethod
    def read(cls, directory: Union[Path, str], fname: Union[Path, str]) -> 'File':
        raise NotImplementedError

    def read_product(self, product_name):
        self.build_index()
        return getattr(self, f'read_{product_name}')()
