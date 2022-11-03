import logging


logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='testwrite')

from weaveio.opr3.l1 import L1SingleSpectrum
fs = data.find_files('l1single_file', skip_extant_files=True)
with data.write:
    data.write_files(*fs, debug_time=True, debug=True)
