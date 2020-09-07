# from functools import reduce

from weaveio.hierarchy import Run, Raw, Target, graph2pdf, L1Single, L1Stack
# from weaveio.graph import Graph
import networkx as nx
from weaveio.data import OurData


# with Graph() as instance_graph:
#     r1 = Raw('r1002813.fit')
#     r2 = Raw('r1002009.fit')
#     s1 = L1Single('single_1002009.fit')
#     st1 = L1Stack('stacked_1002045-lifu.fit')
#     instance_graph.remove_node('Factor')



# invisible = ['Target'] + [f.lower() for f in Target.factors]
# view = nx.subgraph_view(instance_graph, lambda n: not any(n.startswith(i) for i in invisible))

# Take shortest paths from Camera -> Raw
# Remove those which dont go through camera(blue)
# Take shortest paths from Resolution -> Raw
# remove those which dont go through resolution(LowRes)
# Take files from the paths
# Take intersection of the resultant files
#
# def match_files(graph, factor_name, factor_value, filetype):
#     """
#     Match files by building a list of shortest paths that have the subpath:
#         (factor_name)->(factor_value)->...->(file)->(filetype)
#     return [(file) from each valid path]
#     """
#     paths, lengths = zip(*[(p, len(p)) for p in nx.all_shortest_paths(graph, factor_name, filetype)])
#     node = f'{factor_name}({factor_value})'
#     paths = [p for p in paths if len(p) == min(lengths) and node in p]
#     files = {p[-2] for p in paths}
#     return files
#
# def query_files(graph, filetype, factors):
#     """
#     Match all valid files
#     Do this for each given factor and take then intersection of the file set as the result
#     """
#     return reduce(lambda a, b: a & b, [match_files(graph, k, v, filetype) for k, v in factors.items()])

query = ['L1Stack', dict(camera='blue')]
data = OurData('.')

fs = data._query_files(*query)
print(fs)
# view = nx.subgraph_view(view, lambda n: nx.has_path(instance_graph, n, 'L1Stack'))
view = data.graph_view(exclude=['Target', 'ra', 'dec'])
graph2pdf(view, 'instance_graph')