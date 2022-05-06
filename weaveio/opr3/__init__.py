from weaveio.data import Data as BaseData
from weaveio.opr3.l1files import RawFile, L1SingleFile, L1OBStackFile, L1SuperstackFile, L1SupertargetFile
from weaveio.opr3.l2files import L2OBStackFile, L2SuperstackFile, L2SingleFile, L2SupertargetFile


class Data(BaseData):
    filetypes = [
        RawFile,
        L1SingleFile, L2SingleFile,
        L1OBStackFile, L2OBStackFile,
        L1SuperstackFile, L2SuperstackFile,
        L1SupertargetFile, L2SupertargetFile
]