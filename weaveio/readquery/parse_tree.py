from collections import defaultdict
from functools import reduce
from operator import and_
from typing import List

import networkx as nx

from weaveio.readquery.tree import Branch, plot, Alignment, sort_rooted_dag, Collection, BranchHandler, TraversalPath, Results


def parse_tree(input_branch: Branch):
    """
    Given a relevant graph with a defined start and end:
    1. Traverse graph, writing each step to cypher
    2. When an align is reached
        a. Get the branch the input branches to this align were previously aligned
        b. Move the written statements between that hierarchy and the align to their
           own subquery
        c. Move back to the branch that made that hierarchy and try again
    3. When all ancillary paths are done, process the alignment inside the last subquery
    4. Continue
    """
    graph = input_branch.handler.relevant_graph(input_branch)
    ordering = list(sort_rooted_dag(graph))
    todo = ordering.copy()

    current_query = []
    subqueries = []
    previous_branch = None  # type: Branch
    current_alignment = (None, 0, 0)
    while len(todo):
        branch = todo.pop(0)
        if isinstance(branch.action, Alignment):
            assert len(current_query)
            if current_alignment[0] is None:  # set criteria for completion
                current_alignment = (branch, 0, len(branch.parents))
            else:
                assert current_alignment[0] is branch
            if current_alignment[1] != current_alignment[2]:  # if not complete
                branches = reduce(and_, [set(p.find_hierarchy_branches()) for p in branch.parents])
                distances = [(b, nx.shortest_path_length(graph, branch.handler.entry, b)) for b in branches]
                shared_branch = max(distances, key=lambda x: x[1])[0]
                hierarchy = shared_branch.find_hierarchies()[-1]
                for i, done_branch in enumerate(current_query):
                    if hierarchy in done_branch.hierarchies:
                        break
                subquery = current_query[i:]
                subqueries.append(subquery)
                current_query = current_query[:i+1]  # remove subquery from current_query
                do_again = ordering[ordering.index(done_branch):ordering.index(branch)]
                todo = do_again + todo
                current_alignment[1] += 1
                continue
            else:
                current_alignment = (None, 0, 0)  # if complete, reset
        current_query.append(branch)
        previous_branch = branch


def shared_hierarchy_branch(graph, branch):
    branches = reduce(and_, [set(p.find_hierarchy_branches()) for p in branch.parents])
    distances = [(b, nx.shortest_path_length(graph, branch.handler.entry, b)) for b in branches]
    return max(distances, key=lambda x: x[1])[0]


def non_collection_degree(graph, branch):
    degree = 0
    for s in graph.successors(branch):
        if isinstance(s.action, Collection):
            if s.action._reference is branch:
                continue
        degree += 1
    return degree


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def parse_tree2(graph, start, end, l = None):
    """
    This will be recursive

    Forks in the graph imply a subquery (collect references don't count)
    Aligns imply a joining of subqueries

    From a given starting point traverse down, writing each node to a list
    When you hit an align, go up to the parent recursion frame, returning the list and align
    When a fork is encountered:
        Find the align
        Find the shared_parent for the align inputs
        For each child, recurse this function using the the shared_parent as a starting point
    """
    nodes = list(sort_rooted_dag(graph))
    a = nodes.index(start)
    b = nodes.index(end)
    nodes = nodes[a:b+1]

    l = [] if l is None else l
    view = nx.subgraph_view(graph, filter_node=lambda n: n in nodes)
    todo = list(sort_rooted_dag(view))
    while len(todo):
        branch = todo.pop(0)
        l.append(branch)
        if isinstance(branch.action, Alignment):
            # stop iteration and go up
            return l  # includes branch
        if non_collection_degree(graph, branch) > 1:
            # make a new fork from shared_parent until the align
            for align in graph.successors(branch):
                if isinstance(align.action, Alignment):
                    break  # this finds the branch whose reference is this branch
            shared_parent = shared_hierarchy_branch(graph, align)  # this is where we join on
            prior = []
            for child in graph.successors(branch):
                new = parse_tree2(graph, child, align)[:-1]  # chop off child because its already in list
                for i in flatten(new):
                    del todo[todo.index(i)]
                l.append(prior + new)
            l.append(align)
            del todo[todo.index(align)]
    return l


def parse_tree3(branch: Branch):
    ordering = list(nx.algorithms.topological_sort(branch.relevant_graph))
    aligns = [o for o in ordering if isinstance(o.action, (Alignment, Results))]
    paths = {}
    for align in aligns:
        subpaths = []
        for p in align.parents:
            order = list(nx.algorithms.topological_sort(p.relevant_graph))
            stop = 0
            for i, o in enumerate(order):
                if isinstance(o.action, (Alignment, Results)):
                    stop = i
            subpaths.append(order[stop:])
        paths[align] = subpaths

    expanded = defaultdict(list)
    align = None
    for align in aligns:
        for lst in paths[align]:
            newlst = []
            for entry in lst:
                if entry in expanded:
                    newlst += [expanded[entry], entry]
                elif entry in paths:
                    newlst += [paths[entry], entry]
                else:
                    newlst.append(entry)
            expanded[align].append(newlst)
    result = expanded[align][0]
    result.append(ordering[-1])
    return result


def _reduce_stages(stages: List):
    l = []
    shared = None
    for stage in stages:
        if isinstance(stage, list):
            if shared is None:
                shared = stage
            else:
                shared = shared_path_root(shared, stage)
    l += shared
    for stage in stages:
        if all(s in stage for s in shared):
            l.append(stage[len(shared):])
    return l

def shared_path_root(*paths):
    if len(paths) == 0:
        return []
    shared = None
    for path in paths:
        if shared is None:
            shared = path
        else:
            shared = [a for a, b in zip(shared, path) if a == b]
    return shared

def _reduce_stages(stages: List):
    if all(not isinstance(s, list) for s in stages):
        return stages

    newstages = []
    to_compare = []
    for stage in stages:
        if isinstance(stage, list):
            new = reduce_stages(stage)
            newstages.append(new)
            to_compare.append(new)
        else:
            newstages.append(stage)
    shared = shared_path_root(*to_compare)
    cropped = [i[len(shared):] if isinstance(i, list) else i for i in stages]
    return shared + [i for i in cropped if not (isinstance(i, list) and len(i) == 0)]


def reduce_stages(stages: List):
    if all(not isinstance(s, list) for s in stages):
        return stages
    new = []
    for s in stages:
        if isinstance(s, list):
            new.append(reduce_stages(s))
        else:
            new.append(s)
    shared = shared_path_root(*[s for s in new if isinstance(s, list)])
    new = [s[len(shared):] if isinstance(s, list) else s for s in new]
    final = []
    for i in new:
        if not isinstance(i, list):
            final.append(i)
        elif len(i):
            final.append(i)
    return shared + final

def print_nested_list(nested, tab=0):
    for entry in nested:
        if isinstance(entry, list):
            print_nested_list(['--'] + entry, tab+1)
        else:
            print('    '*tab, entry)


def shared_hierarchy_branch(graph, branch):
    branches = reduce(and_, [set(p.find_hierarchy_branches()) for p in branch.parents])
    distances = [(b, nx.shortest_path_length(graph, b, branch)) for b in branches]
    return max(distances, key=lambda x: x[1])[0]

def parse(graph) -> List:
    aligns = [i for i in graph.nodes if isinstance(i.action, Alignment)][::-1]
    shared_aligns = defaultdict(list)
    for align in aligns:
        shared_aligns[shared_hierarchy_branch(graph, align)].append(align)
    query = []
    todo = list(nx.algorithms.topological_sort(graph))
    while todo:
        node = todo.pop(0)
        if node in shared_aligns:
            align_list = shared_aligns[node]
            for align in align_list:
                inputs = align.action.branches
                inputs += (align.action.reference, )
                subqueries = []
                for input_node in inputs:
                    before = list(nx.descendants(graph, node)) + [node]
                    after = list(nx.ancestors(graph, input_node)) + [input_node]
                    newgraph = nx.subgraph_view(graph, lambda n: n in before and n in after)
                    subquery = parse(newgraph)
                    subqueries.append(subquery)
                done = list(set(flatten(subqueries)))
                for d in done:
                    if d in todo:
                        del todo[todo.index(d)]
                reference_subquery = subqueries.pop(-1)
                subqueries += reference_subquery
                query += subqueries
        else:
            query.append(node)
    return query


if __name__ == '__main__':
    handler = BranchHandler()
    ob = handler.begin('OB')
    target = ob.traverse(TraversalPath('->', 'target'))
    run = ob.traverse(TraversalPath('->', 'run'))
    exposure = ob.traverse(TraversalPath('->', 'exposure'))
    ob_targets = ob.collect([], [target])
    any_ob_targets = ob_targets.operate('{any}')
    ob_runs = ob.collect([], [run])
    any_ob_runs = ob_runs.operate('{any}')
    ob_exposures = ob.collect([], [exposure])
    any_ob_exposures = ob_exposures.operate('{any}')

    align0 = any_ob_runs.align(any_ob_targets)
    or1 = align0.operate('{or}')
    align1 = or1.align(any_ob_exposures)
    or2 = align1.operate('{or}')
    final = or2.filter('')
    final = final.results({final: [final.hierarchies[-1].get('obid')]})
    graph = final.relevant_graph
    plot(graph, '/opt/project/weaveio_example_querytree_test_branch.png')
    subqueries = parse(graph)
    print_nested_list(subqueries)
    #
    # from tree_test_weaveio_example import red_spectra
    # final = red_spectra.results({})
    # graph = final.relevant_graph
    # plot(graph, '/opt/project/weaveio_example_querytree_red_branch.png')
    # plot(final.accessible_graph, '/opt/project/weaveio_example_querytree_red_branch_accessible.png')
    # subqueries = parse(graph)
    # print_nested_list(subqueries)

