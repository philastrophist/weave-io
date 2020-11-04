from typing import Tuple, Dict

from weaveio.data import Data
from weaveio.file import File
from weaveio.hierarchy import Hierarchy


"""
File1<-HierarchyA<-HierarchyB
"""

class HierarchyB(Hierarchy):
    factors = ['b_factor_a', 'b_factor_b']
    idname = 'otherid'


class HierarchyA(Hierarchy):
    parents = [HierarchyB]
    factors = ['a_factor_a', 'a_factor_b']
    idname = 'id'


class File1(File):
    parents = [HierarchyA]

    def read(self) -> Tuple[Dict[str, 'Hierarchy'], dict]:
        fname = str(self.fname).split('/')[-1]
        hierarchyb = HierarchyB(otherid=fname, b_factor_b='b',  b_factor_a='a')
        return {'hierarchya': HierarchyA(id=fname, hierarchyb=hierarchyb, a_factor_a='a', a_factor_b='b')}, {}

    @classmethod
    def match(cls, directory):
        return directory.glob('*.fits')


class MyData(Data):
    filetypes = [File1]
