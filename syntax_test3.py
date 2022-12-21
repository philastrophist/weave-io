import logging

import networkx as nx

from weaveio.opr3.hierarchy import Run, SurveyTarget

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)
data = Data(dbname='test', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
fs = data.find_files('l2single', skip_extant_files=False)[:1]
with data.write:
    data.write_files(*fs, timeout=5*60, debug=True, do_not_apply_constraints=False, batch_size=100,
                     test_one=True, debug_time=True, debug_params=False, dryrun=False, halt_on_error=True, parts=['RR', 'GAND', 'PPXF'])
data.validate()
