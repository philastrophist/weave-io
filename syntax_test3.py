import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='lowleveltest')
with data.write:
    # data.apply_constraints()
    # data.drop_all_constraints()
    fs = data.find_files('l1single_file', skip_extant_files=True)
    data.write_files(*fs, dryrun=False)