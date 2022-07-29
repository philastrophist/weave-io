import logging


logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='incrementaltest')
with data.write:
    # data.restore_state(1653303490523)  # restore back to L1 only

    print(len(fs))
    data.write_files(*fs[1:], debug_time=True, dryrun=False, batch_size=50, test_one=False, parts=['GAND'], halt_on_error=False)
