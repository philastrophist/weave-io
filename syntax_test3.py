from astropy.table import Table

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(dbname='bambase')
with data.write:
    fs = data.find_files('l1single', 'l1stack')
    data.write_files(*fs,
                     timeout=10*60, debug=False, test_one=False, debug_time=False,
                     debug_params=False, dryrun=True)
# data.validate()