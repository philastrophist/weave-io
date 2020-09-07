from functools import reduce

from weaveio.hierarchy import Run, Raw, Target, graph2pdf
from weaveio.graph import Graph
import networkx as nx
# instance_graph = nx.DiGraph()
# raw = Raw('r1002813.fit').read_hierarchy()
# add_hierarchies(instance_graph, raw.predecessors['ArmConfig'][0])
# add_hierarchies(instance_graph, raw.predecessors['Exposure'][0])

with Graph() as instance_graph:
    r1 = Raw('r1002813.fit')
    r2 = Raw('r1002009.fit')
    instance_graph.remove_node('Factor')



invisible = ['Target'] + [f.lower() for f in Target.factors]
view = nx.subgraph_view(instance_graph, lambda n: not any(n.startswith(i) for i in invisible))

query = ['Raw', dict(camera='red', resolution='LowRes', mode='MOS')]
query_factors = [f'{k}({v})' for k, v in query[1].items()]

# Take shortest paths from Camera -> Raw
# Remove those which dont go through camera(blue)
# Take shortest paths from Resolution -> Raw
# remove those which dont go through resolution(LowRes)
# Take files from the paths
# Take intersection of the resultant files

def match_files(graph, factor_name, factor_value, filetype):
    """
    Match files by building a list of shortest paths that have the subpath:
        (factor_name)->(factor_value)->...->(file)->(filetype)
    return [(file) from each valid path]
    """
    paths, lengths = zip(*[(p, len(p)) for p in nx.all_shortest_paths(graph, factor_name, filetype)])
    node = f'{factor_name}({factor_value})'
    paths = [p for p in paths if len(p) == min(lengths) and node in p]
    files = {p[-2] for p in paths}
    return files

def query_files(graph, factors, filetype):
    """
    Match all valid files
    Do this for each given factor and take then intersection of the file set as the result
    """
    return reduce(lambda a, b: a & b, [match_files(graph, k, v, filetype) for k, v in factors.items()])



fs = query_files(instance_graph, query[1], query[0])
print(fs)
# view = nx.subgraph_view(view, lambda n: 'peripheries' not in instance_graph.nodes[n],
#                         lambda a, b: instance_graph.edges[(a, b)]['type'] != 'is_a')
graph2pdf(view, 'instance_graph')