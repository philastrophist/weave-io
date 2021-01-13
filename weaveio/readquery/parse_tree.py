from collections import defaultdict
from functools import reduce
from operator import and_
from typing import List

import networkx as nx

from weaveio.readquery.tree import Alignment, Branch
from weaveio.writequery import CypherQuery
from weaveio.writequery.base import Statement


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def print_nested_list(nested, tab=0):
    for entry in nested:
        if isinstance(entry, list):
            print_nested_list(['--'] + entry, tab+1)
        else:
            print('    '*tab, entry)


def shared_hierarchy_branch(graph, branch: Branch):
    branches = reduce(and_, [set(p.find_hierarchy_branches()) for p in branch.parents])
    distances = [(b, nx.shortest_path_length(graph, b, branch)) for b in branches]
    try:
        return max(distances, key=lambda x: x[1])[0]
    except ValueError:
        return branch.handler.entry  # no shared parents means going back to the beginning


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
            align_list = shared_aligns[node][:1]
            query.append(node)
            for align in align_list:
                inputs = align.action.branches
                inputs += (align.action.reference, )
                subqueries = []
                for input_node in inputs:
                    before = list(nx.descendants(graph, node)) + [node]
                    after = list(nx.ancestors(graph, input_node)) + [input_node]
                    newgraph = nx.subgraph_view(graph, lambda n: n in before and n in after)
                    subquery = parse(newgraph)[1:]
                    subqueries.append(subquery)
                done = list(set(flatten(subqueries)))
                for d in done:
                    if d in todo:
                        del todo[todo.index(d)]
                # reference_subquery = subqueries.pop(-1)
                query += subqueries
                # query += reference_subquery
        else:
            query.append(node)
    return query

class OpenSubquery(Statement):
    def __init__(self, input_variables, output_variables, hidden_variables=None):
        super().__init__(input_variables, output_variables, hidden_variables)

    def to_cypher(self):
        inputs = ', '.join(map(str, ['time0'] + self.input_variables))
        return f"CALL {{with {inputs}"


class CloseSubquery(Statement):
    def __init__(self, input_variables, output_variables, hidden_variables=None):
        super().__init__(input_variables, output_variables, hidden_variables)

    def to_cypher(self):
        inputs = ', '.join(map(str, self.output_variables))
        return f"RETURN {inputs}\n}}"


def write_tree(parsed_tree):
    query = CypherQuery.get_context()  # type: CypherQuery
    if not isinstance(parsed_tree, list):
        query.add_statement(parsed_tree.action)
        return parsed_tree.find_variables()
    else:
        inputs = [i for n in flatten(parsed_tree) for i in n.action.input_variables]
        outputs = [i for n in flatten(parsed_tree) for i in n.action.output_variables]
        subquery_inputs = list({i for i in inputs if i not in outputs})

        open = OpenSubquery(subquery_inputs, [])
        query.add_statement(open, safe=False)
        for node in parsed_tree:
            output = write_tree(node)
        output = list({v for v in output if v not in subquery_inputs})
        close = CloseSubquery(outputs, outputs)
        query.add_statement(close)
        return output


if __name__ == '__main__':
    from weaveio.readquery.tree import plot, Alignment, BranchHandler, TraversalPath

    handler = BranchHandler()
    ob = handler.begin('OB')
    target = ob.traverse(TraversalPath('<-', 'OBSpec', '->', 'FibreTarget'))
    run = ob.traverse(TraversalPath('->', 'Exposure', '->', 'Run'))
    exposure = ob.traverse(TraversalPath('->', 'Exposure'))

    ob_targets = ob.collect([], [target])
    any_ob_targets = ob_targets.operate('any(x in {targets} where true)', targets=ob_targets.action.transformed_variables[target.action.target])
    ob_runs = ob.collect([], [run])
    any_ob_runs = ob_runs.operate('any(x in {runs} where true)', runs=ob_runs.action.transformed_variables[run.action.target])
    ob_exposures = ob.collect([], [exposure])
    any_ob_exposures = ob_exposures.operate('any(x in {exposures} where true)', exposures=ob_exposures.action.transformed_variables[exposure.action.target])

    align1 = any_ob_runs.align(any_ob_targets)
    or1 = align1.operate('{run} or {target}', run=any_ob_runs.action.target, target=any_ob_targets.action.target)

    align2 = or1.align(any_ob_exposures)
    or2 = align2.operate('{exp} or {or1}', exp=any_ob_exposures.action.target, or1=or1.action.target)
    final = or2.filter('{or2}', or2=or2.action.target)
    final = final.results({final: [final.hierarchies[-1].get('obid')]})

    for v in align1.find_variables():
        print(v.namehint)

    # graph = final.relevant_graph
    # plot(graph, '/opt/project/weaveio_example_querytree_test_branch.png')
    # subqueries = parse(graph)
    # print_nested_list(subqueries)
    #
    # with CypherQuery() as query:
    #     write_tree(subqueries)
    #
    # cypher, params = query.render_query()
    # print(cypher)

    # print('================================================')
    #
    # from tree_test_weaveio_example import red_spectra
    # final = red_spectra.results({})
    # graph = final.relevant_graph
    # plot(graph, '/opt/project/weaveio_example_querytree_red_branch.png')
    # plot(final.accessible_graph, '/opt/project/weaveio_example_querytree_red_branch_accessible.png')
    # subqueries = parse(graph)
    # print_nested_list(subqueries)
    #
    # with CypherQuery() as query:
    #     write_tree(subqueries)
    #
    # cypher, params = query.render_query()
    # print(cypher)