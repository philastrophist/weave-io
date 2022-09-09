from weaveio.data import Data as BaseData
from weaveio.opr3.hierarchy import *
from weaveio.opr3.l1files import *
from weaveio.opr3.l2files import *


class Data(BaseData):
    filetypes = [
        RawFile,
        L1SingleFile, L2SingleFile,
        L1StackFile, L2StackFile,
        L1SuperstackFile, L2SuperstackFile,
        L1SupertargetFile, L2SupertargetFile
]