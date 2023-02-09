from astropy.table import Table

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')
s = data.l1single_spectra
print(data.ppxfs.l2single_file._debug_output()[0])
print(s.file.fname._debug_output()[0])
print(s.l2single.l2file.fname._debug_output()[0])