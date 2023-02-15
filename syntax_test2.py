from weaveio import *
from weaveio.opr3 import Data
from weaveio.opr3.l2files import L2File

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')

from astropy.table import Table
import weaveio

s = data.l1single_spectra.noss
q = s.ob
print(q._debug_output()[0])
