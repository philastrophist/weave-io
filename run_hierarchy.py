# from functools import reduce

from weaveio.hierarchy import Run, Raw, Target, graph2pdf, L1Single, L1Stack
# from weaveio.graph import Graph
import networkx as nx
from weaveio.data import OurData, Address

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
# from weaveio.graph import Graph
# g = Graph(host='host.docker.internal')
# data = OurData('data/')
# query = data[Address(camera='red')].OBs
# ob_list = query()

from weaveio.data import BasicQuery, Address
# data[camera=red].runs[runid].OBspec.runs[vph=1]()
q = BasicQuery().index_by_address(Address(camera='red', mode='MOS')).index_by_hierarchy_name('Run').\
    index_by_id('1002793').index_by_hierarchy_name('OBSpec', 'above').\
    index_by_hierarchy_name('Run', 'below').index_by_address(Address(vph=1))
cypher = q.make(branch=False)
print(cypher)
from py2neo import Graph
graph = Graph(host='host.docker.internal')
print(q.current_label, graph.run(cypher).to_ndarray().T[0])

cypher = q.make(branch=True)
print(cypher)
from py2neo import Graph
graph = Graph(host='host.docker.internal')
data = graph.run(cypher).data()
print(data)

# query = ['L1Single', dict(camera='red', vph=1)]
# fs = data._query_nodes(*query)
# print(fs)

# data[Address(camera='red')].l1stacks.fnames
# obj = data[Address(camera='red')]
# view = nx.subgraph_view(view, lambda n: nx.has_path(instance_graph, n, 'L1Stack'))
# view = data.graph_view(exclude=['Target', 'ra', 'dec'])
# graph2pdf(view, 'instance_graph')
