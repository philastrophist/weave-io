import networkx as nx
import numpy as np
from networkx import NetworkXNoPath

from weaveio import Data
from weaveio.hierarchy import Hierarchy, Multiple


# class SurveyTarget(Hierarchy):
#     idname = 'id'
#
#
# class OBSpec(Hierarchy):
#     idname = 'id'
#
#
# class FibreTarget(Hierarchy):
#     idname = 'id'
#     parents = [SurveyTarget]
#     children = [OBSpec]
#
#
# class OB(Hierarchy):
#     idname = 'id'
#
#
# class Run(Hierarchy):
#     idname = 'id'
#     parents = [OB]
#
#
# class File(Hierarchy):
#     idname = 'id'
#     parents = [Multiple(Run, maxnumber=100)]
#
#
# class DataTest(Data):
#     filetypes = [File]
#

def all_shortest_paths(G, a, b):
    try:
        for path in nx.all_shortest_paths(G, a, b, 'weight'):
            yield path#[G.short_edge(*e, weight='weight') for e in zip(path[:-1], path[1:])]
    except NetworkXNoPath:
        return []


def find_paths_simple(hier, a, b):
    G = hier.parents_and_inheritance
    assert nx.is_directed_acyclic_graph(G)
    if b in nx.ancestors(G, a):
        a, b = b, a
        reversed = True
    else:
        reversed = False
    forward = all_shortest_paths(G, a, b)
    path = next(forward)
    if reversed:
        path = path[::-1]
        weight = sum(hier.children_and_inheritance.edge_weights(path))
    else:
        weight = sum(G.edge_weights(path))
    yield path, weight
    for path in forward:
        if reversed:
            path = path[::-1]
        newweight = sum(hier.children_and_inheritance.edge_weights(path))
        if newweight == weight:
            weight = newweight
            yield path, weight
        else:
            break



if __name__ == '__main__':
    from weaveio.opr3.hierarchy import *
    from weaveio.opr3.l1 import *
    from weaveio.opr3.l2 import *

    data = Data()
    G = data.hierarchy_graph.nonoptional
    for path, w in find_paths_simple(G, OB, FibreTarget):
        print(path, w)

