from pathlib import Path

from weaveio.hierarchy import Hierarchy


class File(Hierarchy):
    is_template = True
    idname = 'fname'

    def __init__(self, fname, **kwargs):
        super().__init__(tables=None, fname=str(fname), **kwargs)

    @classmethod
    def match(cls, directory: Path):
        raise NotImplementedError

    @classmethod
    def read(cls, directory: Path, fname: Path) -> 'File':
        raise NotImplementedError

    def read_product(self, product_name):
        self.build_index()
        return getattr(self, f'read_{product_name}')()
