import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='weaveio', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
with data.write:
    fs = sorted(data.find_files('l1single_file', skip_extant_files=False), key=lambda f: f.name)[:10]
    data.write_files(*fs, timeout=5*60, debug=True, do_not_apply_constraints=False, batch_size=None,
                              debug_time=True, dryrun=False, halt_on_error=True)
    data.validate(fs)