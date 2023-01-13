import logging

import networkx as nx
from tqdm import tqdm

from weaveio.opr3.hierarchy import Run, SurveyTarget

from weaveio import *
from weaveio.opr3 import Data, RawFile

logging.basicConfig(level=logging.INFO)
data = Data(dbname='test', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
# data.runs[103453].raw_file
with data.write:
    fs = data.find_files('l2single', skip_complete_files=True)
#     for f in tqdm(fs):
#         cypher, d = data.mark_batch_complete_query(f.name, slice(None), None, RawFile.length(fs[0]), [None])
#         data.graph.execute(cypher, **d)
    data.write_files(*fs, timeout=5*60, debug=False, do_not_apply_constraints=False,
                     test_one=False, debug_time=False, debug_params=False, dryrun=True, halt_on_error=True,
                     batch_size=50, parts=['RR'])
# # data.validate()
