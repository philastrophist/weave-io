import logging

from weaveio.data import plot_graph
from weaveio.opr3 import l1, l2, hierarchy, l1files, l2files
from weaveio.opr3.l2 import *

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data()
# data.class_hierarchies['RedrockFit']
for arrows, singular, path in data.paths_to_hierarchy('redrock_fit', 'ingested_spectrum', False):
    print(path, singular)
# q = data.l2obstacks.redrock_fit.best_redshift['*']
#
# lines, params, names = q._compile()
# for line in lines:
#     print(line)
# print(params, names)
import networkx as nx
plot_graph(nx.subgraph(data.relation_graphs[-1],
                       # set(l2.hierarchies) - set(hierarchy.hierarchies) -
                       # set(l1.hierarchies) - {l2.L2OBStack, l2.L2SuperStack, l2.L2SuperTarget}),
                       [RedrockFit, RedrockTemplate, RedshiftMeasurement, IngestedSpectrum, CombinedSpectrum,
                        ] + [Hierarchy._hierarchies['GalaxyRedrockTemplate']]),

           'l2', 'pdf')