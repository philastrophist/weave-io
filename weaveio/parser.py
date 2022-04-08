from collections import deque


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
    cypher = f'OPTIONAL MATCH {path}'
    return make_node(graph, parent, subgraph, [], ''.join(path[-1:]), 'traversal', cypher)

def add_filter(graph: nx.DiGraph, parent, dependencies, operation):
    subgraph = graph.nodes[parent]['subgraph'].copy()
    n = make_node(graph, parent, subgraph, [], graph.nodes[parent]['_name'], 'filter',
                  f'WHERE {operation}')
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
                  operation, 'operation', f'WITH *, {operation} as ...')
    for d in dependencies:
        graph.add_edge(d, n)
    return n

def add_unwind(graph: nx.DiGraph, wrt, sub_dag_nodes):
    sub_dag = nx.subgraph_view(graph, lambda n: n in sub_dag_nodes+[wrt]).copy()  # type: nx.DiGraph
    for node in sub_dag_nodes:
        for edge in sub_dag.in_edges(node):
            graph.remove_edge(*edge) # will be collaped, so remove the original edge
    # for node in sub_dag_nodes[:-1]:  # don't link the aggregation again, it can only
    others = list(graph.successors(sub_dag_nodes[0]))
    if any(i not in sub_dag_nodes for i in others):
        # if the node is going to be used later, add a way to access it again
        graph.add_edge(wrt, sub_dag_nodes[0], label='unwind', operation='unwind')  # TODO: input correct operation
    graph.remove_node(sub_dag_nodes[-1])
    for node in sub_dag_nodes[:-1]:
        if graph.in_degree[node] + graph.out_degree[node] == 0:
            graph.remove_node(node)

def parse_edge(graph: nx.DiGraph, a, b, dependencies):
    # TODO: will do it properly
    return graph.edges[(a, b)]['operation']

def aggregate(graph: nx.DiGraph, wrt, sub_dag_nodes):
    """
    modifies `graph` inplace
    """
    statement = parse_edge(graph, sub_dag_nodes[-2], sub_dag_nodes[-1], [])
    add_unwind(graph, wrt, sub_dag_nodes)
    return statement

import heapq

class UniqueDeque(deque):

    def __init__(self, iterable, maxlen=None, maintain=True) -> None:
        super().__init__([], maxlen)
        for x in iterable:
            self.append(x, maintain)

    def append(self, x, maintain=False) -> None:
        if x in self:
            if maintain:
               return
            self.remove(x)
        super().append(x)

    def appendleft(self, x, maintain=False) -> None:
        if x in self:
            if maintain:
               return
            self.remove(x)
        super().appendleft(x)

    def extend(self, iterable, maintain=False) -> None:
        for x in iterable:
            self.append(x, maintain)

    def extendleft(self, iterable, maintain=False) -> None:
        for x in reversed(iterable):
            self.appendleft(x, maintain)

    def insert(self, i: int, x, maintain=False) -> None:
        if x in self:
            ii = self.index(x)
            if i == ii:
                return
            if maintain:
                return
            self.remove(x)
            if ii < i:
                super().insert(i-1, x)
            elif ii == i:
                super().insert(i, x)



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
        return plot_graph(self.G).render(fname)

    def add_traversal(self, path, parent=None):
        if parent is None:
            parent = self.start
        return add_traversal(self.G, parent, path)

    def add_operation(self, parent, dependencies, operation):
        # do not allow
        return add_operation(self.G, parent, dependencies, operation)

    def add_aggregation(self, parent, wrt, operation):
        return add_aggregation(self.G, parent, wrt, operation)

    def add_filter(self, parent, dependencies, operation):
        return add_filter(self.G, parent, dependencies, operation)

    def optimise(self):
        # TODO: combine get-attribute statements etc...
        pass

    def parse(self):
        """
        Traverse this query graph in the order that will produce a valid cypher query
        Rules:
            1. DAG rules apply: dependencies must be completed before their dependents
            2. When an aggregation route is traversed, you must follow its outward line back to wrt
            3. Do aggregations as early as possible
            4. Aggregations change the graph by collecting
        """
        G = self.G.copy()
        statements = []
        dag = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') != 'wrt')  # type: nx.DiGraph
        backwards = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') == 'wrt')  # type: nx.DiGraph
        ordering = UniqueDeque(nx.topological_sort(dag))
        previous_node = None
        branch = []
        while ordering:
            node = ordering.popleft()
            branches = []
            # find the simplest aggregation and add the required nodes to the front of the queue
            for future_aggregation in backwards.predecessors(node):
                agg_ancestors = nx.ancestors(dag, future_aggregation)
                node_ancestors = nx.ancestors(dag, node)
                sub_dag = nx.subgraph_view(dag, lambda n: n in agg_ancestors and n not in node_ancestors)
                branches.append(list(nx.topological_sort(sub_dag))[1:]+[future_aggregation, node])
            branches.sort(key=lambda x: len(x))
            if branches:
                branch = branches[0]
                ordering.extendleft(branch)
            if previous_node is None:
                previous_node = node
                continue
            edge_type = self.G.edges[(previous_node, node)]['type']
            if edge_type == 'aggr':
                # now change the graph to reflect that we've collected things
                wrt = next(G.successors(node))
                statement = aggregate(G, wrt, branch[:-1])
                before = nx.ancestors(dag, wrt)
                ordering = UniqueDeque(nx.topological_sort(nx.subgraph_view(dag, lambda n: n not in before)))
                previous_node = None
            elif self.G.edges[(previous_node, node)]['type'] == 'filter':
                # make sure everyone is finished with previous_node before proceeding
                others = [n for n in G.successors(previous_node) if n != node]
                if others:
                    ordering.appendleft(node)
                    ordering.extendleft(others)
                    continue
                statement = parse_edge(G, previous_node, node, [])
                previous_node = node
            else:
                # just create the statement given by the edge
                statement = parse_edge(G, previous_node, node, [])
                previous_node = node
            statements.append(statement)
        return statements


# TODO: for each visited node do all aggregations first (sorted by topological ordering)

if __name__ == '__main__':
    G = QueryGraph()
    obs = G.add_traversal(['OB'])  # obs = data.obs
    runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    spectra = G.add_traversal(['spectra'], runs)  # runs.spectra
    l2 = G.add_traversal(['l2'], runs)  # runs.l2
    runid2 = G.add_operation(runs, [], 'runid*2 > 0')  # runs.runid * 2 > 0
    agg = G.add_aggregation(runid2, wrt=obs, operation='all(run.runid*2 > 0)')
    spectra = G.add_filter(spectra, [agg], 'spectra = spectra[all(run.runid*2 > 0)]')
    agg_spectra = G.add_aggregation(spectra, wrt=obs, operation='any(spectra.snr > 0)')
    l2 = G.add_filter(l2, [agg_spectra], 'l2[any(ob.runs.spectra[all(ob.runs.runid*2 > 0)].snr > 0)]')

    G.export('parser')
    for s in G.parse():
        print(s)