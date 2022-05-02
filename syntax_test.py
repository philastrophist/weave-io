import logging

from weaveio.data import plot_graph
from weaveio.opr3 import l1, l2, hierarchy, l1files, l2files
from weaveio.opr3.l2 import *

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data()
# q = data.galaxy_redrock_fit.redshifts

# data.class_hierarchies['RedrockFit']
# for arrows, singular, path in data.paths_to_hierarchy('redrock_fit', 'ingested_spectrum', False):
#     print(path, singular)
# q = data.l2obstacks.redrock_fit.best_redshift['*']
#
# lines, params, names = q._compile()
# for line in lines:
#     print(line)
# print(params, names)
import networkx as nx
G = nx.subgraph_view(nx.subgraph(data.relation_graphs[-1],
                       [
                           # L1SingleSpectrum, L2Single,
                        L2OBStack, L1OBStackSpectrum,
                        RedrockModelSpectrum, RedrockFit, RedrockIngestedSpectrum, RedrockVersion]),
           # filter_edge=lambda a, b: 'relation' not in data.relation_graphs[-1].edges[(a, b)])
                     )
plot_graph(nx.relabel.relabel_nodes(G, {n: n.__name__ for n in G.nodes}),'l2', 'pdf')