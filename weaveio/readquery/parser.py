from collections import defaultdict
from typing import List, Tuple

import networkx as nx

from weaveio.readquery.digraph import HashedDiGraph, plot_graph, add_start, add_traversal, add_filter, add_aggregation, add_operation, add_return, add_unwind, subgraph_view, get_above_state_traversal_graph, node_dependencies
from weaveio.readquery.statements import StartingMatch, Traversal, NullStatement, Operation, GetItem, AssignToVariable, DirectFilter, CopyAndFilter, Aggregate, Return, Unwind


class ParserError(Exception):
    pass


class DeadEndException(Exception):
    pass


def traverse(graph, start=None, end=None, done=None):
    """
    traverse the traversal_graph with backtracking
    """
    dag = subgraph_view(graph, excluded_edge_type='wrt')
    backwards_graph = subgraph_view(graph, only_edge_type='wrt')
    traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
    # semi_traversal = subgraph_view(graph, excluded_edge_type='dep')   # can go through wrt and traversals
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
        options = [b for b in backwards_graph.successors(node) if (node, b) not in done]  # must do wrt first
        if not options:
            options = list(traversal_graph.successors(node))   # where to go next?
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
                    ordering += traverse(graph, option, end, new_done)
                    done.update(new_done)
                    node = ordering[-1]
                    break
                except DeadEndException:
                    pass  # try another option
            else:
                raise DeadEndException  # all options exhausted, entire recursive path is bad


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


def verify(graph):
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
    if len(ends) > 1:
        raise ParserError("Only one output node is allowed")
    if not ends:
        raise ParserError("An output node is required")
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
        nreturns = sum(i == 'return' for i in inputs)
        if naggs > 1:
            raise ParserError(f"Cannot aggregate more than one node at a time: {node}")
        elif naggs:
            if not all(o in ['dep', 'operation', 'aggr'] for o in outputs):
                raise ParserError(f"Can only use aggregations as a dependency/operation/aggregation afterwards {node}")
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
                if graph.edges[list(graph.out_edges(node))[0]]['type'] not in ['aggr', 'operation']:
                    raise ParserError(f"All operations must be aggregated back at some point: {node}")
            except IndexError:
                raise ParserError(f"All operations must be aggregated back at some point: {node}")
            if ntraversals + naggs + nfilters > 1:
                raise ParserError(f"Can only have dependencies as input for an operation: {node}")
        if ndeps:
            if ntraversals or naggs:
                raise ParserError(f"A traversal/aggregation cannot take any other inputs: {node}")
            if not (nops ^ nfilters ^ nreturns):
                raise ParserError(f"A dependency link necessitates an operation or filter: {node}")


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
        self.G = HashedDiGraph()
        self.start = add_start(self.G, 'data')
        self.variable_names = defaultdict(int)
        self.dag_G = nx.subgraph_view(self.G, filter_edge=lambda a, b: self.G.edges[(a, b)]['type'] != 'wrt')  # type: nx.DiGraph
        self.backwards_G = nx.subgraph_view(self.G, filter_edge=lambda a, b: self.G.edges[(a, b)]['type'] == 'wrt')  # type: nx.DiGraph
        self.traversal_G = nx.subgraph_view(self.G, filter_edge=lambda a, b: self.G.edges[(a, b)]['type'] != 'dep')  # type: nx.DiGraph
        self.parameters = {}

    @property
    def statements(self):
        return {d['statement']: (a, b) for a, b, d in self.G.edges(data=True) if 'statement' in d}

    def latest_shared_ancestor(self, *nodes):
        return sorted(set.intersection(*[self.above_state(n, no_wrt=True) for n in nodes]), key=lambda n: len(nx.ancestors(self.traversal_G, n)))[-1]

    def latest_object_node(self, a, b):
        """
        a and b have a shared ancestor
        Scenarios:
            a = single; b = single -> shared
            a = single; b = plural -> choose originating object of b
            a = plural; b = single -> choose ordering object of a
            a = plural; b = plural -> disallowed [at least one must be aggregated back]
        """
        cardinal_a = next(self.backwards_G.successors(a))
        cardinal_b = next(self.backwards_G.successors(b))
        shared = self.latest_shared_ancestor(cardinal_a, cardinal_b)
        if shared == cardinal_a:
            return cardinal_b
        elif shared == cardinal_b:
            return cardinal_a
        else:
            raise ParserError(f"One of [{a}, {b}] must be a parent of the other. {shared} != [{a}, {b}] ")


    @property
    def above_graph(self):
        return get_above_state_traversal_graph(self.G)

    def node_holds_type(self, node, *types):
        return any(d['type'] in types for a, b, d in self.G.in_edges(node, data=True))

    def get_host_nodes_for_operation(self, op_node) -> set:
        """
        Find nodes which are necessary to construct `op_node`
        traverses until it encounters a traversal/filter/aggr then it stops
        An operation can have a single path (just scalar ops) or a branching mess (combining ops),
        """
        necessary = set()
        for input_edge in self.dag_G.in_edges(op_node, data=True):
            input = input_edge[0]
            if self.node_holds_type(input, 'traversal', 'filter', 'aggr'):
                necessary.add(input)
            else:
                necessary |= self.get_host_nodes_for_operation(input)
        return necessary

    def above_state(self, node, no_wrt=False):
        states = nx.descendants(self.above_graph, node)
        states.add(node)
        if not no_wrt:
            for n in states.copy():
                aggregates = self.backwards_G.predecessors(n)
                for aggr in aggregates:
                    edge = list(self.traversal_G.in_edges(aggr))[0]
                    if isinstance(self.G.edges[edge]['statement'], NullStatement):
                        # this is a fake aggregation so get all required nodes as well
                        # TODO: do not do this if the branch is not SINGLE!
                        op_stuff = self.get_host_nodes_for_operation(aggr)
                        states |= op_stuff
                    else:
                        states.add(aggr)
        return states

    def get_variable_name(self, name):
        name = name.lower()
        new_name = f'{name}{self.variable_names[name]}'
        self.variable_names[name] += 1
        return new_name

    def export(self, fname, result_node=None):
        return plot_graph(self.restricted(result_node)).render(fname)

    def add_start_node(self, node_type):
        parent_node = self.start
        statement = StartingMatch(node_type, self)
        return add_traversal(self.G, parent_node, statement)

    def add_traversal(self, parent_node, path: str, end_node_type: str, single=False, unwind=None):
        statement = Traversal(self.G.nodes[parent_node]['variables'][0], end_node_type, path, unwind, self)
        return add_traversal(self.G, parent_node, statement, single=single)

    def fold_to_cardinal(self, parent_node):
        """
        Adds a fake aggregation such that a node can be used as a dependency later.
        For operation chains, you follow the traversal route backwards.
        If the node
        """
        try:
            next(self.backwards_G.successors(parent_node))  # if its already aggregated then do nothing
            return parent_node
        except StopIteration:
            path = nx.shortest_path(self.traversal_G, self.start, parent_node)[::-1]
            for b, a in zip(path[:-1], path[1:]):
                if not self.G.edges[(a, b)]['single']:
                    wrt = b
                    break
            else:
                raise ParserError
            if wrt == parent_node:
                return parent_node
            statement = NullStatement(self.G.nodes[parent_node]['variables'], self)
            return add_aggregation(self.G, parent_node, wrt, statement, 'aggr', True)

    def add_scalar_operation(self, parent_node, op_format_string, op_name) -> Tuple:
        """
        A scalar operation is one which takes only one input and returns one output argument
        the input can be one of [object, operation, aggregation]
        """
        if any(d['type'] == 'aggr' for a, b, d in self.G.in_edges(parent_node, data=True)):
            wrt = next(self.backwards_G.successors(parent_node))
            return self.add_combining_operation(op_format_string, op_name, parent_node, wrt=wrt)
        statement = Operation(self.G.nodes[parent_node]['variables'][0], [], op_format_string, op_name, self)
        return add_operation(self.G, parent_node, [], statement), parent_node

    def add_combining_operation(self, op_format_string, op_name, *nodes, wrt=None) -> Tuple:
        """
        A combining operation is one which takes multiple inputs and returns one output
        Operations should be inline (no variables) for as long as possible.
        This is so they can be used in match where statements
        """
        # if this is combiner operation, then we do everything with respect to the nearest ancestor
        dependency_nodes = [self.fold_to_cardinal(d) for d in nodes]  # fold back when combining
        if wrt is None:
            wrt = self.latest_object_node(*dependency_nodes)
        deps = [self.G.nodes[d]['variables'][0] for d in dependency_nodes]
        statement = Operation(deps[0], deps[1:], op_format_string, op_name, self)
        return add_operation(self.G, wrt, dependency_nodes, statement), wrt

    def add_getitem(self, parent_node, item):
        statement = GetItem(self.G.nodes[parent_node]['variables'][0], item, self)
        return add_operation(self.G, parent_node, [], statement)

    def assign_to_variable(self, parent_node, only_if_op=False):
        if only_if_op and any(d['type'] in ['operation', 'aggr'] for a, b, d in self.G.in_edges(parent_node, data=True)):
            stmt = AssignToVariable(self.G.nodes[parent_node]['variables'][0], self)
            return add_operation(self.G, parent_node, [], stmt)
        return parent_node

    def add_generic_aggregation(self, parent_node, wrt_node, op_format_string, op_name):
        if wrt_node not in nx.ancestors(self.dag_G, parent_node):
            raise SyntaxError(f"{parent_node} cannot be aggregated to {wrt_node} ({wrt_node} is not an ancestor of {parent_node})")
        statement = Aggregate(self.G.nodes[parent_node]['variables'][0], wrt_node, op_format_string, op_name, self)
        return add_aggregation(self.G, parent_node, wrt_node, statement)

    def add_aggregation(self, parent_node, wrt_node, op):
        return self.add_generic_aggregation(parent_node, wrt_node, f"{op}({{0}})", op)

    def add_predicate_aggregation(self, parent, wrt_node, op_name):
        op_format_string = f'{op_name}(x in collect({{0}}) where toBoolean(x))'
        return self.add_generic_aggregation(parent, wrt_node, op_format_string, op_name)

    def add_filter(self, parent_node, predicate_node, direct=False):
        wrt = self.latest_shared_ancestor(parent_node, predicate_node)
        predicate_node = self.fold_to_cardinal(predicate_node, wrt, raise_error=False)
        predicate = self.G.nodes[predicate_node]['variables'][0]
        if direct:
            FilterClass = DirectFilter
        else:
            FilterClass = CopyAndFilter
        statement = FilterClass(self.G.nodes[parent_node]['variables'][0], predicate, self)
        return add_filter(self.G, parent_node, [predicate_node], statement)

    def add_unwind_parameter(self, wrt_node, to_unwind):
        statement = Unwind(wrt_node, to_unwind, to_unwind.replace('$', ''), self)
        return add_unwind(self.G, wrt_node, statement)

    def collect_or_not(self, index_node, other_node, want_single):
        """
        Collect `other_node` with respect to the shared common ancestor of `index_node` and `other_node`.
        If other_node is above the index, fold back to cardinal node
        if not, fold back to shared ancestor
        """
        shared = self.latest_shared_ancestor(index_node, other_node)
        try:
            if next(self.backwards_G.successors(other_node)) == index_node:
                if want_single:
                    return other_node
                else:
                    return self.add_aggregation(other_node, shared, 'collect')
        except StopIteration:
            pass
        if want_single:
            return self.fold_to_cardinal(other_node)
        return self.add_aggregation(other_node, shared, 'collect')  # coalesce

    def add_results_table(self, index_node, column_nodes, request_singles: List[bool]):
        # fold back column data into the index node
        column_nodes = [self.collect_or_not(index_node, d, s) for d, s in zip(column_nodes, request_singles)] # fold back when combining
        deps = [self.G.nodes[d]['variables'][0] for d in column_nodes]
        try:
            vs = self.G.nodes[index_node]['variables'][0]
        except IndexError:
            vs = None
        statement = Return(deps, vs, self)
        return add_return(self.G, index_node, column_nodes, statement)

    def add_scalar_results_row(self, *column_nodes):
        """data already folded back"""
        deps = [self.G.nodes[d]['variables'][0] for d in column_nodes]
        statement = Return(deps, None, self)
        return add_return(self.G, self.start, column_nodes, statement)

    def add_parameter(self, value, name=None):
        name = f'${name}' if name is not None else '$'
        varname = self.get_variable_name(name)
        self.parameters[varname] = value
        return varname

    def restricted(self, result_node=None):
        if result_node is None:
            return nx.subgraph_view(self.G)
        return nx.subgraph_view(self.G, lambda n: nx.has_path(self.dag_G, n, result_node))

    def traverse_query(self, result_node=None):
        graph = self.restricted(result_node)
        verify(graph)
        return traverse(graph)

    def verify_traversal(self, goal, ordering):
        graph = self.restricted(goal)
        return verify_traversal(graph, ordering)

    def cypher_lines(self, result):
        import time
        start_time = time.perf_counter()
        ordering = self.traverse_query(result)
        self.verify_traversal(result, ordering)
        statements = []
        for i, e in enumerate(zip(ordering[:-1], ordering[1:])):
            try:
                statement = self.G.edges[e]['statement'].make_cypher(ordering[:i+1])
                if statement is not None:
                    statements.append(statement)
            except KeyError:
                pass
        end_time = time.perf_counter()
        timed = end_time - start_time
        return statements



if __name__ == '__main__':
    G = QueryGraph()


    # def get_node_i(graph, i):
    #     return next(n for n in graph.nodes if graph.nodes[n].get('i', -1) == i)

    # # # 0
    # obs = G.add_start_node('OB')  # obs = data.obs
    # runs = G.add_traversal(obs, '-->', 'Run')  # runs = obs.runs
    # spectra = G.add_traversal(runs, '-->', 'L1SingleSpectrum') # runs.spectra
    # result = G.add_results(spectra, G.add_getitem(spectra, 'snr'))

    # # 1
    # obs = G.add_start_node('OB')
    # runs = G.add_traversal(obs, '-->', 'Run')  # runs = obs.runs
    # spectra = G.add_traversal(runs, '-->', 'L1SingleSpectrum')  # runs.spectra
    # l2 = G.add_traversal(runs, '-->', 'L2')  # runs.l2
    # runid = G.add_getitem(runs, 'runid')
    # runid2 = G.add_scalar_operation(runid, '{0} * 2 > 0', '2>0')  # runs.runid * 2 > 0
    # agg = G.add_predicate_aggregation(runid2, obs, 'all')
    # spectra = G.add_filter(spectra, agg)
    # snr_above0 = G.add_scalar_operation(G.add_getitem(spectra, 'snr'), '{0}>0', '>')
    # agg_spectra = G.add_predicate_aggregation(snr_above0, obs, 'any')
    # result = G.add_filter(l2, agg_spectra)  # l2[any(ob.runs.spectra[all(ob.runs.runid*2 > 0)].snr > 0)]

    # 2
    obs = G.add_start_node('OB')  # obs = data.obs
    runs = G.add_traversal(obs, '-->(:Exposure)-->', 'Run')  # runs = obs.runs
    camera = G.add_getitem(G.add_traversal(runs, '<--', 'ArmConfig', single=True), 'camera')
    is_red, _ = G.add_scalar_operation(camera, '{0} = "red"', '==')
    red_runs = G.add_filter(runs, is_red)
    snr = G.add_getitem(red_runs, 'snr')
    red_snr = G.add_aggregation(snr, obs, 'avg')  #  'mean(run.camera==red, wrt=obs)'
    spec = G.add_traversal(runs, '-->(:Observation)-->(:RawSpectrum)-->', 'L1SingleSpectrum')
    snr = G.add_getitem(spec, 'snr')
    compare, _ = G.add_combining_operation('{0} > {1}', '>', snr, red_snr)
    spec = G.add_filter(spec, compare)
    l2 = G.add_traversal(spec, '-->', 'L2')
    snr = G.add_getitem(l2, 'snr')
    result = G.add_results(l2, snr)

    # # 3
    # # obs = data.obs
    # # x = all(obs.l2s[obs.l2s.ha > 2].hb > 0, wrt=obs)
    # # y = mean(obs.runs[all(obs.runs.l1s[obs.runs.l1s.camera == 'red'].snr > 0, wrt=runs)].l1s.snr, wrt=obs)
    # # z = all(obs.targets.ra > 0, wrt=obs)
    # # result = obs[x & y & z]
    # obs = G.add_start_node('OB')  # obs = data.obs
    # l2s = G.add_traversal(obs, '-->', 'l2')  # l2s = obs.l2s
    # has = G.add_traversal(l2s, '-->', 'ha', single=True)  # l2s = obs.l2s.ha
    # above_2 = G.add_scalar_operation(has, '{0} > 2', '>')  # l2s > 2
    # hb = G.add_traversal(G.add_filter(l2s, above_2), '-->', 'hb', single=True)
    # hb_above_0 = G.add_scalar_operation(hb, '{0} > 0', '>0')
    # x = G.add_predicate_aggregation(hb_above_0, obs, 'all')
    #
    # runs = G.add_traversal(obs, '-->', 'runs')
    # l1s = G.add_traversal(runs, '-->', 'l1')
    # camera = G.add_traversal(l1s, '-->', 'camera')
    # is_red = G.add_scalar_operation(camera, '{0}= "red"', '=red')
    # red_l1s = G.add_filter(l1s, is_red)
    # red_snrs = G.add_scalar_operation(G.add_getitem(red_l1s, 'snr'), '{0}> 0', '>0')
    # all_red_snrs = G.add_predicate_aggregation(red_snrs, runs, 'all')
    # red_runs = G.add_filter(runs, all_red_snrs)
    # red_l1s = G.add_traversal(red_runs, '-->', 'l1')
    # y = G.add_scalar_operation(G.add_aggregation(G.add_getitem(red_l1s, 'snr'), obs, 'avg'), '{0}>1', '>1')
    #
    # targets = G.add_traversal(obs, '-->', 'target')
    # z = G.add_predicate_aggregation(G.add_scalar_operation(G.add_getitem(targets, 'ra'), '{0}>0', '>0'), obs, 'all')
    #
    # # TODO: need to somehow make this happen in the syntax
    # x_and_y = G.add_combining_operation('{0} and {1}', '&', x, y)
    # x_and_y_and_z = G.add_combining_operation('{0} and {1}', '&', x_and_y, z)
    # result = G.add_filter(obs, x_and_y_and_z)

    #
    # # 4
    # obs = G.add_start_node('OB')  # obs
    # exps = G.add_traversal(obs, '-->', 'Exposure')  # obs.exps
    # runs = G.add_traversal(exps, '-->', 'Run')  # obs.exps.runs
    # l1s = G.add_traversal(runs, '-->', 'L1')  # obs.exps.runs.l1s
    # snr = G.add_getitem(l1s, 'snr')  # obs.exps.runs.l1s.snr
    # avg_snr_per_exp = G.add_aggregation(snr, exps, 'avg')  # x = mean(obs.exps.runs.l1s.snr, wrt=exps)
    # avg_snr_per_run = G.add_aggregation(snr, runs, 'avg')  # y = mean(obs.exps.runs.l1s.snr, wrt=runs)
    #
    # exp_above_1 = G.add_scalar_operation(avg_snr_per_exp, '{0} > 1', '>1')  # x > 1
    # run_above_1 = G.add_scalar_operation(avg_snr_per_run, '{0} > 1', '> 1')  # y > 1
    # l1_above_1 = G.add_scalar_operation(snr, '{0} > 1', '> 1')  # obs.exps.runs.l1s.snr > 1
    #
    # # cond = (x > 1) & (y > 1) & (obs.exps.runs.l1s.snr > 1)
    # l1_and_run = G.add_combining_operation('{0} and {1}', '&', l1_above_1, run_above_1)
    # condition = G.add_combining_operation('{0} and {1}', '&', l1_and_run, exp_above_1)
    # l1s = G.add_filter(l1s, condition)  # obs.exps.runs.l1s[cond]
    # result = G.add_traversal(l1s, '-->', 'L2')

    #
    # obs = G.add_start_node('OB')  # obs
    # exps = G.add_traversal(obs, '-->', 'Exposure')  # obs.exps
    # runs = G.add_traversal(exps, '-->', 'Run')  # obs.exps.runs
    # l1s = G.add_traversal(runs, '-->', 'L1')  # obs.exps.runs.l1s
    # snr = G.add_getitem(l1s, 'snr')  # obs.exps.runs.l1s.snr
    # avg_snr_per_exp = G.add_aggregation(snr, exps, 'avg')  # x = mean(obs.exps.runs.l1s.snr, wrt=exps)
    # avg_snr_per_run = G.add_aggregation(snr, runs, 'avg')  # y = mean(obs.exps.runs.l1s.snr, wrt=runs)
    #
    # exp_above_1 = G.add_scalar_operation(avg_snr_per_exp, '{0} > 1', '>1')  # x > 1
    # run_above_1 = G.add_scalar_operation(avg_snr_per_run, '{0} > 1', '> 1')  # y > 1
    # l1_above_1 = G.add_scalar_operation(snr, '{0} > 1', '> 1')  # obs.exps.runs.l1s.snr > 1
    #
    # # cond = (x > 1) & (y > 1) & (obs.exps.runs.l1s.snr > 1)
    # l1_and_run = G.add_combining_operation('{0} and {1}', '&', l1_above_1, run_above_1)
    # condition = G.add_combining_operation('{0} and {1}', '&', l1_and_run, exp_above_1)
    # l1s = G.add_filter(l1s, condition)  # obs.exps.runs.l1s[cond]
    # result = G.add_traversal(l1s, '-->', 'L2')


    # used to use networkx 2.4
    G.export('parser', result)
    dag = subgraph_view(G.G, excluded_edge_type='wrt')
    backwards = subgraph_view(G.G, only_edge_type='wrt')
    traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
    dep_graph = subgraph_view(G.G, only_edge_type='dep')
    plot_graph(traversal_graph).render('parser-traversal')


    ordering = []
    import time
    start_time = time.perf_counter()
    for n in G.traverse_query(result):
        end_time = time.perf_counter()
        print(G.G.nodes[n]["i"])
        ordering.append(n)
    verify_traversal(G.G, ordering)
    print(end_time - start_time)

    # ordering = [1, 2, 3, 4, 1, 2, 3, 4, 5, 3, 4, 6, 7]
    # ordering = [G.G.nodes[o]['i'] for o in ordering]
    # edge_ordering = insert_load_saves(ordering)
    # for o in edge_ordering:
    #     if isinstance(o, Store):
    #         print(f"Store: {o.state} -> {o.reference}")
    #     elif isinstance(o, Load):
    #         print(f"Load: {o.reference} --> {o.state}")
    #     else:
    #         print(f"Trav: {o[0]} -> {o[1]}")
    # verify_saves(edge_ordering, ordering)
    # edge_ordering = collapse_chains_of_loading(edge_ordering)
    for i, e in enumerate(zip(ordering[:-1], ordering[1:])):
        try:
            statement = G.G.edges[e]['statement'].make_cypher(ordering[:i+1])
            if statement is not None:
                print(statement)
        except KeyError:
            pass


    # TODO: Single load/save  and then unroll recursively