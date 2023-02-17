from collections import Counter

import networkx as nx

from weaveio import *
from weaveio.opr3 import Data
from weaveio.opr3.l2files import L2File

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')

from astropy.table import Table
import weaveio

# s = data.l1single_spectra.noss
# q = s.ob
# print(q._debug_output()[0])

# counter = Counter([(u, v) for u, v, k in data.hierarchy_graph.parents.edges(keys=True)])
#
# for edge, n in counter.most_common(1):
#     for kedge in data.hierarchy_graph.parents.edges(keys=True):
#         if kedge[:-1] == edge:
#             print(kedge)
from weaveio.opr3.hierarchy import *
from weaveio.opr3.l1 import *
from weaveio.opr3.l2 import *

a, b = WeaveTarget, Redrock

# for p in nx.all_simple_paths(data.hierarchy_graph.parents_and_inheritance, a, b):
#     print([i.__name__ for i in p])
print('---')
for p in nx.all_shortest_paths(data.hierarchy_graph.parents_and_inheritance, a, b):
    print([i.__name__ for i in p])

# just used forking shortest paths
# all paths valid
# if there is a singular path, return only that
# if there