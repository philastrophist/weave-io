import networkx as nx
import graphviz
from networkx import dfs_tree, dfs_edges
from networkx.drawing.nx_pydot import to_pydot

def plot_graph(graph):
    g = nx.DiGraph()
    for n in graph.nodes():
        g.add_node(n)
    for e in graph.edges():
        g.add_edge(*e, **graph.edges[e])
    return graphviz.Source(to_pydot(g).to_string())

def graph2string(graph: nx.DiGraph):
    sources = {n for n in graph.nodes() if len(list(graph.predecessors(n))) == 0}
    return ','.join('->'.join(dfs_tree(graph, source)) for source in sources)


def make_node(graph: nx.DiGraph, parent, subgraph: nx.DiGraph, scalars: list,
              label: str, type: str, opertation: str):
    _name = label
    i = graph.number_of_nodes()
    try:
        label = f'{i}\n{graph.nodes[label]["_name"]}'
    except KeyError:
        label = f"{i}\n{label}"
    path = graph2string(subgraph)
    label += f'\n{path}'
    graph.add_node(label, subgraph=subgraph, scalars=scalars, _name=_name, i=i)
    if parent is not None:
        graph.add_edge(parent, label, type=type, label=f"{type}-{opertation}", operation=opertation)
    return label

def add_start(graph: nx.DiGraph, name):
    g = nx.DiGraph()
    g.add_node(name)
    return make_node(graph, None, g, [], name, '', '')

def add_traversal(graph: nx.DiGraph, parent, path):
    subgraph = graph.nodes[parent]['subgraph'].copy()  # type: nx.DiGraph
    subgraph.add_edge(graph.nodes[parent]['_name'], path[0])
    for a, b in zip(path[:-1], path[1:]):
        subgraph.add_edge(a, b)
    return make_node(graph, parent, subgraph, [], ''.join(path[-1:]), 'traversal', '')

def add_filter(graph: nx.DiGraph, parent, dependencies, operation):
    subgraph = graph.nodes[parent]['subgraph'].copy()
    n = make_node(graph, parent, subgraph, [], graph.nodes[parent]['_name'], 'filter', operation)
    for d in dependencies:
        graph.add_edge(d, n)
    return n

def add_aggregation(graph: nx.DiGraph, parent, wrt, operation):
    subgraph = graph.nodes[wrt]['subgraph'].copy() # type: nx.DiGraph
    n = make_node(graph, parent, subgraph, graph.nodes[parent]['scalars'] + [operation],
                     operation, 'aggr', operation)
    graph.add_edge(n, wrt, type='wrt', style='dashed')
    return n

def add_operation(graph: nx.DiGraph, parent, dependencies, operation):
    subgraph = graph.nodes[parent]['subgraph'].copy()  # type: nx.DiGraph
    n = make_node(graph, parent, subgraph, graph.nodes[parent]['scalars'] + [operation],
                  operation, 'operation', operation)
    for d in dependencies:
        graph.add_edge(d, n)
    return n

class QueryGraph:
    """
    Rules of adding nodes/edges:
    Traversal:
        Can only traverse to another hierarchy object if there is a path between them
        Always increases/maintains cardinality
    Aggregation:
        You can only aggregate back to a predecessor of a node (the parent)
        Nodes which require another aggregation node must share the same parent as just defined above

    Golden rule:
        dependencies of a node must share an explicit parent node
        this basically says that you can only compare nodes which have the same parents

    optimisations:
        If the graph is duplicated in multiple positions, attempt to not redo effort
        For instance, if you traverse and then agg+filter back to a parent and the traverse the same path
        again after filtering, then the aggregation is changed to conserve the required data and the duplicated traversal is removed
        
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.start = add_start(self.G, 'data')

    def export(self, fname):
        return plot_graph(self.G).view(fname)

    def add_traversal(self, path, parent=None):
        if parent is None:
            parent = self.start
        return add_traversal(self.G, parent, path)

    def add_operation(self, parent, dependencies, operation):
        return add_operation(self.G, parent, dependencies, operation)

    def add_aggregation(self, parent, wrt, operation):
        return add_aggregation(self.G, parent, wrt, operation)

    def add_filter(self, parent, dependencies, operation):
        return add_filter(self.G, parent, dependencies, operation)

    def parse(self):
        """
        Traverse this query graph in the order that will produce a valid cypher query
        Rules:
            1. DAG rules apply: dependencies must be completed before their dependents
            2. When an aggregation route is traversed, you must follow its outward line back to wrt
                This means collecting the node and
        """


        dag = nx.subgraph_view(self.G, filter_edge=lambda a, b: self.G.edges[(a, b)].get('type', '') != 'wrt')
        for node in nx.topological_sort(dag):
            print(self.G.nodes[node]['i'])


if __name__ == '__main__':
    # obs[all(obs.runs.runid*2 > 0, wrt=obs)].runs.runid
    G = QueryGraph()
    obs = G.add_traversal(['OB'])  # obs = data.obs
    runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    runid2 = G.add_operation(runs, [], 'runid*2 > 0')  # runid2 = runs.runid * 2 > 0
    agg = G.add_aggregation(runid2, wrt=obs, operation='all(run.runid*2 > 0)')
    obs = G.add_filter(obs, [agg], 'all(run.runid > 0)')
    runs = G.add_traversal(['run'], obs)
    runid = G.add_operation(runs, [], 'runid')
    G.export('parser')
    G.parse()