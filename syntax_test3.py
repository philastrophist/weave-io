import logging

import networkx as nx

from weaveio.opr3.hierarchy import Run, SurveyTarget

logging.basicConfig(level=logging.INFO)
from weaveio import *
from weaveio.opr3 import Data

data = Data(dbname='weaveio', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
# data.hierarchy_graph.find_paths(Run, SurveyTarget, False)

G = data.hierarchy_graph
paths = nx.all_shortest_paths(G, Run, SurveyTarget)
paths

# with data.write:
#     fs = data.find_files('l2stack', skip_extant_files=False)[:1]
#     data.write_files(*fs, timeout=5*60, debug=True, do_not_apply_constraints=False, batch_size=None,
#                               debug_time=True, dryrun=False, halt_on_error=True)
# data.validate()
# print(max(data.l1single_spectra[data.l1single_spectra.camera == 'blue'].snr)())
# q = sum(data.runs.targuses == 'S', wrt=data.runs)()
# print(q)