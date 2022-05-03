from weaveio.data import Data as BaseData
from weaveio.opr3.l1files import RawFile, L1SingleFile, L1OBStackFile, L1SuperStackFile, L1SuperTargetFile
from weaveio.opr3.l2files import L2OBStackFile, L2SuperStackFile, L2SingleFile, L2SuperTargetFile


class Data(BaseData):
    filetypes = [
        RawFile,
        L1SingleFile, L2SingleFile,
        L1OBStackFile, L2OBStackFile,
        L1SuperStackFile, L2SuperStackFile,
        L1SuperTargetFile, L2SuperTargetFile
]