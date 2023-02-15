from astropy.table import Table

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')
with data.write:
    fs = data.find_files('l2single', 'l2stack')
    data.write_files(*fs,
                     timeout=10*60, debug=True, test_one=False, debug_time=True,
                     debug_params=False, dryrun=False)
data.validate()