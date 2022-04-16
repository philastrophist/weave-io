from collections import deque
from functools import cmp_to_key

import networkx as nx
import graphviz
from networkx import dfs_tree, dfs_edges, NodeNotFound
from networkx.drawing.nx_pydot import to_pydot
from typing import List, Tuple, Optional


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
              label: str, type: str, operation: str, **edge_data):
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
        graph.add_edge(parent, label, type=type, label=f"{type}-{operation}", operation=operation, **edge_data)
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
        graph.add_edge(d, n, type='dep', style='dotted')
    return n

def add_aggregation(graph: nx.DiGraph, parent, wrt, operation, type='aggr'):
    subgraph = graph.nodes[wrt]['subgraph'].copy() # type: nx.DiGraph
    n = make_node(graph, parent, subgraph, graph.nodes[parent]['scalars'] + [operation],
                     operation, type, operation)
    graph.add_edge(n, wrt, type='wrt', style='dashed')
    return n

def add_operation(graph: nx.DiGraph, parent, dependencies, operation):
    subgraph = graph.nodes[parent]['subgraph'].copy()  # type: nx.DiGraph
    n = make_node(graph, parent, subgraph, graph.nodes[parent]['scalars'] + [operation],
                  operation, 'operation', f'WITH *, {operation} as ...')
    for d in dependencies:
        graph.add_edge(d, n, type='dep', style='dotted')
    return n

def add_unwind(graph: nx.DiGraph, wrt, sub_dag_nodes):
    sub_dag = nx.subgraph_view(graph, lambda n: n in sub_dag_nodes+[wrt]).copy()  # type: nx.DiGraph
    for node in sub_dag_nodes:
        for edge in sub_dag.in_edges(node):
            if graph.edges[edge]['type'] != 'unwind':  # in case someone else needs it
                graph.remove_edge(*edge) # will be collapsed, so remove the original edge
    others = list(graph.successors(sub_dag_nodes[0]))
    if any(i not in sub_dag_nodes for i in others):
        # if the node is going to be used later, add a way to access it again
        graph.add_edge(wrt, sub_dag_nodes[0], label='unwind', operation='unwind', type='unwind')  # TODO: input correct operation
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

def aggregate_reused_filtered_nodes(graph: nx.DiGraph):
    """


    """

    for n in list(graph.nodes):
        filts = [edge for edge in graph.out_edges(n) if graph.edges[edge]['type'] == 'filter']
        nonfilts = [edge for edge in graph.out_edges(n) if graph.edges[edge]['type'] != 'filter']
        if filts and nonfilts:
            wrt = next(graph.predecessors(n))  # TODO: this fails if its not a traversal edge
            add_aggregation(graph, n, wrt, f'collect({n})')


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

    def parse(self, output):
        """
        Traverse this query graph in the order that will produce a valid cypher query
        Rules:
            1. DAG rules apply: dependencies must be completed before their dependents
            2. When an aggregation route is traversed, you must follow its outward line back to wrt
            3. Do aggregations as early as possible
            4. Aggregations change the graph by collecting
            5. If a node is subsequently filtered, do all unfiltered ops first
        Order of operations at a node N are:
            1. Aggregation branches that perform any filter (
            2. Aggregation branches that perform any filter (add a collection before doing this)
            3. Root continuation
        """
        G = nx.subgraph_view(self.G, filter_node=lambda n: nx.has_path(self.G, n, output)).copy()  # type: nx.DiGraph
        # return parse(G, output)
        statements = []
        dag = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') != 'wrt')  # type: nx.DiGraph
        backwards = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') == 'wrt')  # type: nx.DiGraph
        ordering = UniqueDeque(nx.topological_sort(dag))
        # TODO: when do we break into subqueries? at aggregation?
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
            simple_branches = [branch for branch in branches if all(dag.edges[a, b]['type'] != 'filter' for a, b in zip(branch[:-1], branch[1:]))]

            if simple_branches:
                branches = simple_branches  # do the filters later
            elif branches:
                # there are filters annoying us here, so we protect the previous+current state by aggregating it
                # this bit isn't hit if there is a filter-branch that doesn't aggregate (ie. its the main root)
                statement = parse_edge(G, previous_node, node, [])
                statements.append(statement)
                path_backwards = nx.shortest_path(dag, self.start, node)[::-1]  # only one path exists
                for b, a in zip(path_backwards[:-1], path_backwards[1:]):
                    if G.edges[(a, b)]['type'] == 'traversal':
                        break
                else:
                    raise ValueError("No part of this branch contains a traversal")
                statement = aggregate(G, a, path_backwards[:path_backwards.index(a)+1])  # TODO: does this work when its a single traversal?
                statements.append(statement)
                before = nx.ancestors(dag, a)
                ordering = UniqueDeque(nx.topological_sort(nx.subgraph_view(dag, lambda n: n not in before)))
                previous_node = None
                continue
            branches.sort(key=lambda x: len(x))

            # if a branch depends on another branch then that first branch will contain the required branch
            # therefore, taking the branch with minimum number of nodes will suffice
            # However, if a filter is required and the nfiltered node is required later, then we either
            # have to repeat ourselves or collect everything
            if branches:
                branch = branches[0]
                ordering.extendleft(branch)
            if previous_node is None:
                previous_node = node
                continue
            edge_type = self.G.edges[(previous_node, node)]['type']
            if edge_type == 'aggr':
                # now change the graph to reflect that we've collected things
                wrt = next(backwards.successors(node))
                statement = aggregate(G, wrt, branch[:-1])
                before = nx.ancestors(dag, wrt)
                ordering = UniqueDeque(nx.topological_sort(nx.subgraph_view(dag, lambda n: n not in before)))
                previous_node = None
            else:
                # just create the statement given by the edge
                statement = parse_edge(G, previous_node, node, [])
                previous_node = node
            statements.append(statement)
        return statements


@cmp_to_key
def compare_two_dags(dag1: str, dag2: str):
    if any(n in dag2 for n in dag1):
        return +1
    if any(n in dag1 for n in dag2):
        return -1
    else:
        return 0

def preserve_state_for_subqueries(graph: nx.DiGraph, start: str, node: str):
    dag = nx.subgraph_view(graph, filter_edge=lambda a, b: graph.edges[(a, b)].get('type', '') != 'wrt')
    wrts = nx.subgraph_view(graph, filter_edge=lambda a, b: graph.edges[(a, b)].get('type', '') == 'wrt')
    path_backwards = nx.shortest_path(dag, start, node)[::-1]  # only one path exists
    for b, wrt in zip(path_backwards[:-1], path_backwards[1:]):
        if graph.edges[(wrt, b)]['type'] == 'traversal':
            break
    else:
        raise ValueError("Fatal error: BUG: No part of this branch contains a traversal")
    aggregated_node = add_aggregation(graph, node, wrt, f'collect()', type='pre-subquery')
    unwound = make_node(graph, aggregated_node, graph.nodes[node]['subgraph'], [], 'unwind', 'unwind', 'unwind')
    for out_edge in list(graph.out_edges(node)):
        if out_edge[1] != aggregated_node:
            graph.add_edge(unwound, out_edge[1], **graph.edges[out_edge])
            graph.remove_edge(*out_edge)
    for in_edge in list(wrts.in_edges(node)):
        graph.add_edge(in_edge[0], unwound, **graph.edges[in_edge])
        graph.remove_edge(*in_edge)
    return aggregated_node, unwound
    # return aggregate(graph, wrt, path_backwards[:path_backwards.index(wrt) + 1]), wrt  # this works when its a single traversal

def get_branches(dag: nx.DiGraph, backwards: nx.DiGraph, parent: str) -> Tuple[List[List[str]], bool]:
    """
    finds the sub DAG nodes which comprise branches separating off from `parent`
    returns: branches: List[List[str]], unwound: True/False
    """
    branches = []
    for future_aggregation in backwards.predecessors(parent):
        agg_ancestors = nx.ancestors(dag, future_aggregation)
        node_ancestors = nx.ancestors(dag, parent)
        sub_dag = nx.subgraph_view(dag, lambda n: n in agg_ancestors and n not in node_ancestors)
        # sub_dag = subgraph_view(dag, only_nodes=agg_ancestors | {future_aggregation, parent},
        #                         excluded_nodes=node_ancestors,
        #                         excluded_edge_type='dep',
        #                         path_to=future_aggregation)
        # sub_dag is only the absolutely required nodes to reach `future_aggregation` from `parent`
        # sub_dag = nx.shortest_simple_paths(dag, parent, future_aggregation)
        branches.append(list(nx.topological_sort(sub_dag))[1:] + [future_aggregation, parent])
    branches.sort(key=compare_two_dags)
    return branches, not any(i[-1]['type'] == 'unwind' for i in dag.in_edges(parent, data=True, default=(None, None, {'type': ''})))


def sever_branch_wrt_connections(graph: nx.DiGraph, *branches: List[str]) -> None:
    for branch in branches:
        agg = branch[-2]
        wrt = branch[-1]
        graph.remove_edge(agg, wrt)


def rephrase_filtered_operations(graph: nx.DiGraph) -> None:
    """
    You are allowed to filter on operations such as:
    (ob.runids * sum(ob.expmjds, wrt=ob))[ob.expmjd > 0 & ob.runid > 0]
    but for the filtering to work, it needs to be rephrased as a filtering on a hierarchy
    ob.runs[ob.expmjd > 0 & ob.runs.runid > 0].runid * sum(ob.expmjds, wrt=ob))
    [off-branch dependencies are already folded in]
    steps:
    1. find all filter edges where output is operation
    2. replace (hier)--(op)--(op)--...--(op)-|-(op) with (hier)-|-(hier)--(get
    """
    for edge in graph.edges:
        if graph.edges[edge]['type'] == 'filter':
            if any(graph.edges[e]['type'] == 'operation' for e in graph.in_edges(edge[1])):
                raise NotImplementedError(f"Filtering an operation is not yet supported")



# def parse(G: nx.DiGraph, output: str):
#     G = nx.subgraph_view(G, filter_node=lambda n: nx.has_path(G, n, output)).copy()  # type: nx.DiGraph
#     rephrase_filtered_operations(G)
#     statements = []
#     dag = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') != 'wrt')  # type: nx.DiGraph
#     backwards = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') == 'wrt')  # type: nx.DiGraph
#     ordering = UniqueDeque(nx.topological_sort(dag))
#     previous_node = None
#     while ordering:
#         node = ordering.popleft()
#         branches, needs_preserving = get_branches(dag, backwards, node)
#         if branches:
#             branches.sort(key=compare_two_dags)
#             branch = branches[0]
#             ordering.extendleft(branch)
#         if previous_node is None:
#             previous_node = node
#             continue
#         edge_type = G.edges[(previous_node, node)]['type']
#         if edge_type == 'aggr':
#             previous_node =

def parse2(G: nx.DiGraph, output: str, subquery=False):
    """
    Traverse the dependency DAG in order of dependency
    Open and close subqueries on branches off of the main trunk that contain filters
    """
    G = nx.subgraph_view(G, filter_node=lambda n: nx.has_path(G, n, output)).copy()  # type: nx.DiGraph
    statements = []
    dag = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') != 'wrt')  # type: nx.DiGraph
    backwards = nx.subgraph_view(G, filter_edge=lambda a, b: G.edges[(a, b)].get('type', '') == 'wrt')  # type: nx.DiGraph
    ordering = list(nx.topological_sort(dag))
    start = ordering[0]
    ordering = UniqueDeque(ordering)
    previous_node = None
    branch = []

    if subquery:
        statements.append('CALL { with variables ')
        # previous_node = ordering.popleft()

    while ordering:
        node = ordering.popleft()
        branches, needs_preserving = get_branches(dag, backwards, node)
        if branches:
            if needs_preserving:
                aggregated, unwound = preserve_state_for_subqueries(G, start, node)
                ordering.appendleft(unwound)
                ordering.appendleft(aggregated)
            else:
                branches.sort(key=compare_two_dags)
                # if a branch depends on another branch then that first branch will contain the required branch
                # therefore, taking the branch with minimum number of nodes will suffice
                branch = branches[0]
                sever_branch_wrt_connections(G, branch)
                collection = next(G.predecessors(node))
                sub_graph = nx.subgraph_view(G, lambda n: (n in branch) or (n == collection)).copy()
                sub_statements = parse(sub_graph, branch[-2], node)
                for node in branch[:-1]:
                    ordering.remove(node)
                previous_node = node
                statements.append(sub_statements)
                continue
        if previous_node is None:
            previous_node = node
            continue
        # just create the statement given by the edge
        statement = parse_edge(G, previous_node, node, [])
        previous_node = node
        statements.append(statement)
    if subquery:
        statements.append('return variables }')
    return statements


def get_statement_data(graph, a, b, is_aggregating):
    return graph.nodes[a], graph.nodes[b], graph.edges[(a, b)], is_aggregating


def save_state(graph: nx.DiGraph, wrt: str, node: str):
    """
    collects up the given node into a state list which can be unwound to get back to where you were
    this replaces the `traversal` with an `unwind`
    """
    aggregated = add_aggregation(graph, node, wrt, 'save-state', 'save-state')  # new node attached to node and then to wrt
    d = graph.edges[(wrt, node)]  # original edge
    d['label'] = f'load-state {graph.nodes[node]["_name"]}'
    d['type'] = 'load-state'
    d['operation'] = 'load-state'
    # graph.remove_edge(wrt, node)
    # graph.add_edge(wrt, aggregated)
    for successor in list(graph.successors(node)):
        if successor != aggregated:
            loaded_state = make_node(graph, wrt, graph.nodes[node]['subgraph'], [], **d)
            graph.add_edge(loaded_state, successor, **graph.edges[(node, successor)])
            graph.remove_edge(node, successor)
    # graph.remove_node(node)
    return aggregated

def has_outside_dependencies(super_graph: nx.DiGraph, graph: nx.DiGraph, node: str) -> bool:
    return any(n not in graph for n in super_graph.successors(node) if super_graph.edges[(node, n)]['type'] != 'load-state')

def subgraph_view(graph: nx.DiGraph, excluded_edge_type=None, only_edge_type=None,
                  only_nodes: List = None, excluded_nodes: List = None,
                  only_edges: List[Tuple] = None, excluded_edges: List[Tuple] = None,
                  path_to = None,
                  ) -> nx.DiGraph:
    """
    filters out edges and nodes
    """
    excluded_edges = set([] if excluded_edges is None else excluded_edges)
    excluded_nodes = set([] if excluded_nodes is None else excluded_nodes)
    if excluded_edge_type is not None:
        excluded_edges |= {e for e in graph.edges if graph.edges[e].get('type', '') == excluded_edge_type}
    if only_edge_type is not None:
        excluded_edges |= {e for e in graph.edges if graph.edges[e].get('type', '') != only_edge_type}
    if only_nodes is not None:
        excluded_nodes |= {n for n in graph.nodes if n not in only_nodes}
    if only_edges is not None:
        excluded_edges |= {e for e in graph.edges if e not in only_edges}
    r = nx.restricted_view(graph, excluded_nodes, excluded_edges)  # type: nx.DiGraph
    if path_to:
        r = nx.subgraph_view(r, lambda n:  nx.has_path(graph, n, path_to))
    return r


class ParserError(Exception):
    pass

def traverse_query_graph(G: nx.DiGraph, output: str, node: str, original_graph: nx.DiGraph = None, aggregating=False):
    """
    if there is a fork in the graph, then this means the state should be saved
    """
    if original_graph is None:
        # all graph operations must be done on the original_graph, since only Views get passed on to recursion
        original_graph = G.copy()
        G = subgraph_view(original_graph)
    dag = subgraph_view(G, excluded_edge_type='wrt')
    backwards = subgraph_view(G, only_edge_type='wrt')
    start = node

    while G.edges:
        branches, _ = get_branches(dag, backwards, node)  # sorted in terms of inter-dependency
        if branches:
            branch = branches[0]
            forbidden_edge = (branch[-2], node)
            sub_graph = subgraph_view(G, only_nodes=branch, excluded_edges=[forbidden_edge],
                                      path_to=branch[-2])
            # the below recursion may edit the graph
            yield from traverse_query_graph(sub_graph, branch[-2], node, original_graph, True)  # do the sub branch first
            # TODO: remove the aggregation node, move output edges to come from the wrt node?
            original_graph.remove_edge(branch[-2], branch[-1])
            original_graph.remove_node(branch[-2])
            continue
        if len(list(subgraph_view(dag, excluded_edge_type='load-state').successors(node))) > 1:
            raise ParserError(f"Fatal: Query node has > 1 unaggregated-successors. This is a bug")
        successors = list(backwards.successors(node)) + list(dag.successors(node))
        if not successors:
            return  # terminate traversal
        successor = successors[0]
        if G.edges[(node, successor)]['type'] == 'dep':
            raise ParserError(f"Fatal: traversed a dependency indicator. This is a bug")
        yield get_statement_data(G, node, successor, aggregating)
        if aggregating:
            if successor != output:
                # TODO: outside dependencies, not global deps because, hang on think about this
                if has_outside_dependencies(original_graph, G, successor):
                    save_state(original_graph, start, successor)
                    G = subgraph_view(G, path_to=output)
                    dag = subgraph_view(G, excluded_edge_type='wrt')
                    backwards = subgraph_view(G, only_edge_type='wrt')
        original_graph.remove_edge(node, successor)  # edge will not be used again so delete it
        node = successor

def prepare_graph(graph: nx.DiGraph):
    """
    1. uses on aggregated nodes must take the wrt node as input and the aggregation as dep
    2.
    """
    backwards = subgraph_view(graph, only_edge_type='wrt')
    for a, b, data in list(graph.edges(data=True)):
        if data['type'] == 'operation':
            try:
                wrt = next(backwards.successors(a))
                graph.add_edge(wrt, b, **graph.edges[(a, b)])
                graph.remove_edge(a, b)
                graph.add_edge(a, b, type='dep', style='dotted')
            except StopIteration:
                pass


def verify(graph: nx.DiGraph, ):
    """
    Check that edges and nodes are allowed:
        - There is only one output node and one input node (no hanging nodes)
        - There is a path from input->output
        - can only aggregate to a parent
        - There are no cyclic dependencies in the dag
        - can only use an aggregation when it's wrt is a parent
        - all operations must be aggregated
        - Multiple inputs into a node should comprise:
            all deps that are aggregated
            one other (can be anything)
        - For an agg node, there is only one wrt
        - You can have > 1 inputs when they are ops

        - Multiple outputs from a node:
            no more than one out-path should be unaggregated in the end
            (i.e. there should only be one path from start-output which contains no aggregations)
    """
    dag = subgraph_view(graph, excluded_edge_type='wrt')
    traversal = subgraph_view(dag, excluded_edge_type='dep')
    if not nx.is_arborescence(traversal):
        raise ParserError(f"Invalid query: The DAG for this query is not a directed tree with max 1 parent per node")
    starts = [n for n in dag.nodes if dag.in_degree(n) == 0]
    ends = [n for n in dag.nodes if dag.out_degree(n) == 0]
    if len(starts) != 1:
        raise ParserError("Only one input node is allowed")
    if len(ends) != 1:
        raise ParserError("Only one output node is allowed")
    backwards = subgraph_view(graph, only_edge_type='wrt')
    without_agg = subgraph_view(dag, excluded_edge_type='aggr')
    main_paths = nx.all_simple_paths(without_agg, starts[0], ends[0])
    try:
        next(main_paths)
        next(main_paths)
    except StopIteration:
        pass
    else:
        # there can be 0 in the case where the output is itself an aggregation
        raise ParserError(f"There can only be at maximum one path from {starts[0]} to {ends[0]} that is not aggregated")
    if not nx.is_directed_acyclic_graph(dag):
        raise ParserError(f"There are cyclical dependencies")
    if not nx.has_path(dag, starts[0], ends[0]):
        raise ParserError(f"There must be a path from {starts[0]} to {ends[0]}")
    for agg, wrt in backwards.edges:
        if not nx.has_path(graph, wrt, agg):
            raise ParserError(f"{wrt} must be a parent of {agg} in order to aggregate")
        for node in dag.successors(agg):
            if not nx.has_path(graph, wrt, node):
                raise ParserError(f"{node} can an only use what is aggregated above it. failure on {agg} (parent={wrt})")
    for node in graph.nodes:
        inputs = [graph.edges[i]['type'] for i in graph.in_edges(node)]
        inputs = [i for i in inputs if i != 'wrt']
        outputs = [graph.edges[i]['type'] for i in graph.out_edges(node)]
        if sum(o == 'wrt' for o in outputs) > 1:
            raise ParserError(f"Cannot put > 1 wrt paths as output from an aggregation")
        outputs = [o for o in outputs if o != 'wrt']
        nfilters = sum(i == 'filter' for i in inputs)
        ntraversals = sum(i == 'traversal' for i in inputs)
        ndeps = sum(i == 'dep' for i in inputs)
        nops = sum(i == 'operation' for i in inputs)
        naggs = sum(i == 'aggr' for i in inputs)
        if naggs > 1:
            raise ParserError(f"Cannot aggregate more than one node at a time: {node}")
        elif naggs:
            if not all(o in ['dep', 'operation'] for o in outputs):
                raise ParserError(f"Can only use aggregations as a dependency/operation afterwards {node}")
        if nfilters > 2:
            raise ParserError(f"Can only have one filter input: {node}")
        elif nfilters:
            if ntraversals + nops + naggs > 0:
                raise ParserError(f"A filter can only take dependencies not traversals/ops/aggregations: {node}")
        if ntraversals > 2:
            raise ParserError(f"Can only have one traversal input: {node}")
        elif ntraversals:
            if len(inputs) > 1:
                raise ParserError(f"Can only traverse with one input: {node}")
        if nops > 1:
            raise ParserError(f"Can only have one op input: {node}")
        elif nops:
            try:
                if graph.edges[list(graph.out_edges(node))[0]]['type'] != 'aggr':
                    raise ParserError(f"All operations must be aggregated back: {node}")
            except IndexError:
                raise ParserError(f"All operations must be aggregated back: {node}")
            if ntraversals + naggs + nfilters > 1:
                raise ParserError(f"Can only have dependencies as input for an operation: {node}")
        if ndeps:
            if ntraversals or naggs:
                raise ParserError(f"A traversal/aggregation cannot take any other inputs: {node}")
            if not (nops ^ nfilters):
                raise ParserError(f"A dependency link necessitates an operation or filter: {node}")

def traverse_backtrack(graph):
    """
    There is always only one way to get to a node (in the dag)

    Rules:
        - wrt are traversed exactly once
        -
    """
    edge_graph = nx.line_graph(subgraph_view(graph, excluded_edge_type='dep')).copy()
    start = [n for n in edge_graph.nodes if edge_graph.in_degree(n) == 0][0]
    end = [n for n in edge_graph.nodes if edge_graph.out_degree(n) == 0][0]

if __name__ == '__main__':
    from json import dumps
    G = QueryGraph()

    # # # 0
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    # spectra = G.add_traversal(['spectra'], runs)  # runs.spectra
    # result = spectra

    # 1
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    # spectra = G.add_traversal(['spectra'], runs)  # runs.spectra
    # l2 = G.add_traversal(['l2'], runs)  # runs.l2
    # runid2 = G.add_operation(runs, [], 'runid*2 > 0')  # runs.runid * 2 > 0
    # agg = G.add_aggregation(runid2, wrt=obs, operation='all(run.runid*2 > 0)')
    # spectra = G.add_filter(spectra, [agg], 'spectra = spectra[all(run.runid*2 > 0)]')
    # agg_spectra = G.add_aggregation(spectra, wrt=obs, operation='any(spectra.snr > 0)')
    # result = G.add_filter(l2, [agg_spectra], 'l2[any(ob.runs.spectra[all(ob.runs.runid*2 > 0)].snr > 0)]')

    # # 2
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    # red_runs = G.add_filter(runs, [], 'run.camera==red')
    # red_snr = G.add_aggregation(G.add_operation(red_runs, [], 'run.snr'), obs, 'mean(run.camera==red, wrt=obs)')
    # spec = G.add_traversal(['spec'], runs)
    # spec = G.add_filter(spec, [red_snr], 'spec[spec.snr > red_snr]')
    # result = G.add_traversal(['l2'], spec)

    # 3
    # obs = data.obs
    # x = all(obs.l2s[obs.l2s.ha > 2].hb > 0, wrt=obs)
    # y = mean(obs.runs[all(obs.runs.l1s[obs.runs.l1s.camera == 'red'].snr > 0, wrt=runs)].l1s.snr, wrt=obs)
    # z = all(obs.targets.ra > 0, wrt=obs)
    # result = obs[x & y & z]
    obs = G.add_traversal(['OB'])  # obs = data.obs
    l2s = G.add_traversal(['l2'], obs)  # l2s = obs.l2s
    has = G.add_traversal(['ha'], l2s)  # l2s = obs.l2s.ha
    above_2 = G.add_aggregation(G.add_operation(has, [], '> 2'), l2s, 'single')  # l2s > 2
    hb = G.add_traversal(['hb'], G.add_filter(l2s, [above_2], ''))
    hb_above_0 = G.add_operation(hb, [], '> 0')
    x = G.add_aggregation(hb_above_0, obs, 'all')

    runs = G.add_traversal(['runs'], obs)
    l1s = G.add_traversal(['l1'], runs)
    camera = G.add_traversal(['camera'], l1s)
    red = G.add_aggregation(G.add_operation(camera, [], '==red'), l1s, 'single')
    red_l1s = G.add_filter(l1s, [red], '')
    red_snrs = G.add_operation(red_l1s, [], 'snr > 0')
    red_runs = G.add_filter(runs, [G.add_aggregation(red_snrs, runs, 'all')], '')
    red_l1s = G.add_traversal(['l1'], red_runs)
    y = G.add_aggregation(G.add_operation(red_l1s, [], 'snr'), obs, 'mean')

    targets = G.add_traversal(['target'], obs)
    z = G.add_aggregation(G.add_operation(targets, [], 'target.ra > 0'), obs, 'all')

    # TODO: need to somehow make this happen in the syntax
    op = G.add_aggregation(G.add_operation(obs, [x, y, z], 'x&y&z'), obs, 'single')
    # op = G.add_aggregation(G.add_operation(obs, [x], 'x'), obs, 'single')

    result = G.add_filter(obs, [op], '')


    # ## 4
    # obs = G.add_traversal(['ob'])  # obs
    # exps = G.add_traversal(['exp'], obs)  # obs.exps
    # runs = G.add_traversal(['run'], exps)  # obs.exps.runs
    # l1s = G.add_traversal(['l1'], runs)  # obs.exps.runs.l1s
    # snr = G.add_operation(l1s, [], 'snr')  # obs.exps.runs.l1s.snr
    # avg_snr_per_exp = G.add_aggregation(snr, exps, 'avg')  # x = mean(obs.exps.runs.l1s.snr, wrt=exps)
    # avg_snr_per_run = G.add_aggregation(snr, runs, 'avg')  # y = mean(obs.exps.runs.l1s.snr, wrt=runs)
    #
    # exp_above_1 = G.add_aggregation(G.add_operation(avg_snr_per_exp, [], '> 1'), exps, 'single')  # x > 1
    # run_above_1 = G.add_aggregation(G.add_operation(avg_snr_per_run, [], '> 1'), runs, 'single')  # y > 1
    # l1_above_1 = G.add_aggregation(G.add_operation(snr, [], '> 1'), l1s, 'single')  # obs.exps.runs.l1s.snr > 1
    #
    # # cond = (x > 1) & (y > 1) & (obs.exps.runs.l1s.snr > 1)
    # condition = G.add_aggregation(G.add_operation(l1s, [l1_above_1, run_above_1, exp_above_1], '&'), l1s, 'single')  # chosen the lowest
    # l1s = G.add_filter(l1s, [condition], '')  # obs.exps.runs.l1s[cond]
    # result = G.add_traversal(['l2'], l1s)



    # used to use networkx 2.4
    verify(G.G)
    # prepare_graph(G.G)
    G.export('parser')
    dag = subgraph_view(G.G, excluded_edge_type='wrt')
    backwards = subgraph_view(G.G, only_edge_type='wrt')
    traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
    dep_graph = subgraph_view(G.G, only_edge_type='dep')
    plot_graph(traversal_graph).render('parser-traversal')


    def get_node_i(graph, i):
        return next(n for n in graph.nodes if graph.nodes[n].get('i', -1) == i)

    def where_next(shared, backwards_graph, traversal_graph, dep_graph, dag):
        try:
            return next(backwards_graph.successors(shared))  # always follow wrt
        except StopIteration:
            sort = nx.topological_sort(nx.dfs_tree(dag, shared))
            next(sort)
            return next(sort)

    def dependencies_of_branches(graph, node):
        dag = subgraph_view(graph, excluded_edge_type='wrt')
        backwards_graph = subgraph_view(graph, only_edge_type='wrt')
        aggregations = list(backwards_graph.predecessors(node))
        traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
        if not aggregations:
            return []
        ancestors = nx.ancestors(dag, node)
        ancestors.add(node)
        branches = []
        for aggregation in aggregations:
            path = nx.shortest_path(traversal_graph, node, aggregation)
            deps = {d for n in path for d in subgraph_view(dag, path_to=n).nodes} - ancestors
            deps -= set(path)
            branches.append((path[1:], deps))
        branches.sort(key=lambda b: len(b[1]))
        return branches

    def dependencies_of_forks(graph, node):
        dag = subgraph_view(graph, excluded_edge_type='wrt')
        backwards_graph = subgraph_view(graph, only_edge_type='wrt')
        traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
        successors = list(node.successors(node))
        if not successors:
            return []
        ancestors = nx.ancestors(dag, node)
        ancestors.add(node)
        branches = []
        for successor in successors:
            tree = nx.dfs_tree(traversal_graph, node)
            deps = {d for n in tree for d in subgraph_view(dag, path_to=n).nodes} - ancestors
            deps -= set(tree.nodes)
            branches.append(deps)
        branches.sort(key=lambda b: len(b[1]))
        return branches





    def dangling_branch(node, traversal_graph):
        for ancestor in nx.dfs_tree(traversal_graph.reverse(), node):
            if traversal_graph.out_degree(ancestor) > 1:
                return
            yield ancestor

    def traverse(graph, node=None):
        """still needs backtracking"""
        graph = graph.copy()
        dag = subgraph_view(graph, excluded_edge_type='wrt')
        backwards_graph = subgraph_view(graph, only_edge_type='wrt')
        traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
        dep_graph = subgraph_view(graph, only_edge_type='dep')
        if node is None:
            node = next(nx.topological_sort(traversal_graph))  # get top node
        yield node
        while True:
            try:
                # store state here so we can revert to later
                # so make `where_next` an iterator so we can keep trying
                successor = where_next(node, backwards_graph, traversal_graph, dep_graph, dag)
            except ValueError:
                return  # no more to do
            yield successor
            if graph.edges[(node, successor)]['type'] == 'wrt':
                graph.remove_edge(node, successor)
            if not traversal_graph.out_degree(node):
                to_remove = list(dangling_branch(node, traversal_graph))
                graph.remove_nodes_from(to_remove)
            node = successor

    def node_dependencies(graph, node):
        dag = subgraph_view(graph, excluded_edge_type='wrt')
        return {n for n in graph.nodes if nx.has_path(dag, n, node)} - {node}

    def verify_traversal(graph, traversal_order):
        edges = list(zip(traversal_order[:-1], traversal_order[1:]))
        if any(graph.edges[e]['type'] == 'dep' for e in edges):
            raise ParserError(f"Some dep edges where traversed. This is a bug")
        semi_dag = subgraph_view(graph, excluded_edge_type='dep')
        if set(semi_dag.edges) != set(edges):
            raise ParserError(f"Not all edges were traversed. This is a bug")
        done = set()
        for n in traversal_order:
            if n not in done:
                if not all(dep in done for dep in node_dependencies(graph, n)):
                    raise ParserError(f"node {n} does not have all its dependencies satisfied. This is a bug")
                done.add(n)


    def traverse2(graph, start=None, end=None):
        graph = graph.copy()
        dag = subgraph_view(graph, excluded_edge_type='wrt')
        backwards_graph = subgraph_view(graph, only_edge_type='wrt')
        traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
        dep_graph = subgraph_view(graph, only_edge_type='dep')
        naive_ordering = list(nx.topological_sort(dag))
        if start is None:
            start = naive_ordering[0]  # get top node
        if end is None:
            end = naive_ordering[-1]
        path = nx.shortest_path(traversal_graph, start, end)
        n = path.pop(0)
        yield n
        while n != end:
            try:
                successor = next(backwards_graph.successors(n))
                if not path:
                    graph.remove_edge(n, successor)
                n = successor
            except StopIteration:
                if not path:
                    path = nx.shortest_path(traversal_graph, n, end)[1:]
                n = path.pop(0)
            yield n
            branches = dependencies_of_branches(graph, n)
            if branches:
                path, deps = branches[0]


    class DeadEndException(Exception):
        pass

    def traverse3(graph, start=None, end=None, done=None):
        """
        traverse the traversal_graph with backtracking
        """
        dag = subgraph_view(graph, excluded_edge_type='wrt')
        semi_traversal = subgraph_view(graph, excluded_edge_type='dep')   # can go through wrt and traversals
        dep_graph = subgraph_view(graph, only_edge_type='dep')
        if start is None or end is None:
            naive_ordering = list(nx.topological_sort(dag))
            if start is None:
                start = naive_ordering[0]  # get top node
            if end is None:
                end = naive_ordering[-1]
        ordering = [start]
        node = start
        done = set() if done is None else done  # stores wrt edges and visited nodes
        while True:
            dependencies = dep_graph.predecessors(node)
            if not all(dep in done for dep in dependencies):
                raise DeadEndException
            options = list(semi_traversal.successors(node))   # where to go next?
            if not options:
                # if you cant go anywhere and you're not done, then this recursive path is bad
                if node != end:
                    raise DeadEndException
                else:
                    return ordering
            elif len(options) == 1:
                # if there is only one option, go there... obviously
                edge = (node, options[0])
                if edge in done:
                    # recursive path is bad if you have to go over the same wrt edge more than once
                    raise DeadEndException
                elif graph.edges[edge]['type'] == 'wrt':
                    done.add(edge)
                done.add(node)
                node = options[0]
                ordering.append(node)
            else:
                # open up recursive paths from each available option
                # this is such a greedy algorithm
                for option in options:
                    try:
                        new_done = done.copy()
                        ordering += traverse3(graph, option, end, new_done)
                        done.update(new_done)
                        node = ordering[-1]
                        break
                    except DeadEndException:
                        pass  # try another option
                else:
                    raise DeadEndException  # all options exhausted, entire recursive path is bad


    ordering = []
    import time
    start_time = time.perf_counter()
    for n in traverse3(G.G.copy()):
        end_time = time.perf_counter()
        print(G.G.nodes[n]["i"])
        ordering.append(n)
    verify_traversal(G.G, ordering)
    print(end_time - start_time)



    # edge_graph = nx.line_graph(subgraph_view(G.G, excluded_edge_type='dep')).copy()
    # start = [n for n in edge_graph.nodes if edge_graph.in_degree(n) == 0][0]
    # end = [n for n in edge_graph.nodes if edge_graph.out_degree(n) == 0][0]
    # edge_graph.add_edge(end, start)
    # ordering = nx.topological_sort(G.G)
    #
    # plot_graph(edge_graph).render('parser-edge')
    # edges = list(nx.dfs_edges(edge_graph))
    #
    # for a, b in edges[::2]:
    #     assert a[1] == b[0]
    #     print(a, b)


    # todo: save states first
    # todo: visit each edge once
    # todo: travelling salesman on edge_graph
    # graph = subgraph_view(G.G, path_to=result).copy()
    # verify(graph)
    # statements = []
    # for statement_data in traverse_query_graph(graph, result, G.start):
    #     statements.append(statement_data)
    #     print(statement_data[0]['i'], statement_data[1]['i'], statement_data[2]['type'])
    # statements