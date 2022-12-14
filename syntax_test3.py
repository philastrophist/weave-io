import logging

import networkx as nx

from weaveio.opr3.hierarchy import Run, SurveyTarget

logging.basicConfig(level=logging.INFO)
from weaveio import *
from weaveio.opr3 import Data

data = Data(dbname='weaveio', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
fs = data.find_files('raw', skip_extant_files=False)[:1]
data.write_files(*fs, timeout=5*60, debug=True, do_not_apply_constraints=True, batch_size=None,
                          debug_time=True, dryrun=True, halt_on_error=True)
