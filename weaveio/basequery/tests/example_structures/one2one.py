from typing import Tuple, Dict

from weaveio.data import Data
from weaveio.file import File
from weaveio.hierarchy import Hierarchy


class HierarchyA(Hierarchy):
    factors = ['a_factor_a', 'a_factor_b']
    idname = 'id'


class File1(File):
    parents = [HierarchyA]

    def read(self) -> Tuple[Dict[str, 'Hierarchy'], dict]:
        return {'hierarchya': HierarchyA(id=str(self.fname), a_factor_a='a', a_factor_b='b')}, {}

    @classmethod
    def match(cls, directory):
        return directory.glob('*.fits')


class MyData(Data):
    filetypes = [File1]
