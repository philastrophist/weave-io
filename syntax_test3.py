from astropy.table import Table

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)
# data = Data(dbname='opr3b')
# stack = data.l1stack_files['stack_1002249.fit']
# single_fnames = stack.l1single_files.path()
# raw_fnames = stack.raw_files.path()
# stack_fname = stack.path()[0]

data = Data(dbname='opr3bretest')
with data.write:
    fs = data.find_files('raw', 'l1single', 'l1stack')
    # data.drop_all_constraints()
    # fs = [data.rootdir / r for r in raw_fnames]
    # fs += [data.rootdir / r for r in single_fnames]
    # fs += [data.rootdir / stack_fname]
    data.write_files(*fs, timeout=5*60, debug=True,
                     test_one=False,
                     debug_time=True, debug_params=False, dryrun=False, halt_on_error=True)

data.validate()
# stack = Table.read(data.rootdir / stack_fname, hdu='FIBTABLE')
# singles = [Table.read(data.rootdir / f, hdu='FIBTABLE') for f in single_fnames]
# raws = [Table.read(data.rootdir / f, hdu='FIBTABLE') for f in raw_fnames]