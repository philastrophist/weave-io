from typing import Tuple, Dict

from weaveio.data import Data
from weaveio.file import File
from weaveio.hierarchy import Hierarchy, Multiple

"""
File1<-HierarchyA<-HierarchyB<-Multiple(HierarchyC)
"""

class HierarchyC(Hierarchy):
    factors = ['c_factor_a', 'c_factor_b']


class HierarchyB(Hierarchy):
    parents = [Multiple(HierarchyC, 2, 2)]
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
        hierarchyc1 = HierarchyC(id=fname+'1', c_factor_a='a', c_factor_b='b')
        hierarchyc2 = HierarchyC(id=fname+'2', c_factor_a='a', c_factor_b='b')
        hierarchyb = HierarchyB(otherid=fname, b_factor_b='b',  b_factor_a='a', hierarchycs=[hierarchyc1, hierarchyc2])
        return {'hierarchya': HierarchyA(id=fname, hierarchyb=hierarchyb, a_factor_a='a', a_factor_b='b')}, {}

    @classmethod
    def match(cls, directory):
        return directory.glob('*.fits')


class MyData(Data):
    filetypes = [File1]
