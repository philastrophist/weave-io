import pytest
import numpy as np

from weaveio.writequery import CypherQuery, merge_node, match_node, unwind, collect, groupby, CypherData
from weaveio.graph import Graph


PREFIX = "WITH *, timestamp() as time0"


def test_context_compatibility():
    query = CypherQuery()
    graph = Graph(host='host.docker.internal')
    with graph:
        with query:
            assert CypherQuery.get_context() is query
            assert Graph.get_context() is graph


def test_merge_node():
    with CypherQuery() as query:
        merge_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
    cypher = query.render_query().split('\n')
    assert cypher[-2] == "MERGE (b0:A:B {a: 1, b: 2, c: [1, 2, 3]}) ON CREATE SET b0.dbcreated = time0"


def test_match_node():
    with CypherQuery() as query:
        match_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
    cypher = query.render_query().split('\n')
    assert cypher[-2] == "MATCH (b0:A:B {a: 1, b: 2, c: [1, 2, 3]})"


def test_merge_many2one_with_one():
    with CypherQuery() as query:
        parent = match_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
        child = merge_node(['C'], {'p': 1}, parents={parent: 'req'}, versioned_labels=['A'])
    cypher = query.render_query().split('\n')
    assert cypher[1] == "MATCH (b0:A:B {a: 1, b: 2, c: [1, 2, 3]})"
    assert cypher[2] == "WITH *, [[b0, 'req', {order: 0}]] as bspec0"
    assert cypher[3] == "WITH *, bspec0 as specs0"
    assert cypher[-3] == "CALL custom.multimerge(specs0, ['C'], {p: 1}, {dbcreated: time0}, {dbcreated: time0}) YIELD child as c0"
    assert cypher[-2] == "CALL custom.version(specs0, c0, ['A'], 'version') YIELD version as version0"


def test_merge_many2one_with_mixture():
    with CypherQuery() as query:
        parent1 = match_node(labels=['A', 'B'], properties={'a': 1})
        parent2 = match_node(labels=['A', 'B', 'other'], properties={'a': 2})
        child = merge_node(['C'], {'p': 1}, parents={parent1: 'req1', parent2: 'req2'}, versioned_labels=['other'])
    cypher = query.render_query()
    expected = [PREFIX,
        "MATCH (b0:A:B {a: 1})",
        "MATCH (other0:A:B:Other {a: 2})",
        "WITH *, [[b0, 'req1', {order: 0}]] as bspec0",
        "WITH *, [[other0, 'req2', {order: 0}]] as otherspec0",
        "WITH *, bspec0+otherspec0 as specs0",
        "CALL custom.multimerge(specs0, ['C'], {p: 1}, {dbcreated: time0}, {dbcreated: time0}) YIELD child as c0",
        "CALL custom.version(specs0, c0, ['other'], 'version') YIELD version as version0",
        "RETURN time0"
                ]
    expected = '\n'.join(expected)
    assert cypher == expected


def test_merge_many2one_with_mixture_run(procedure_tag, database):
    with CypherQuery() as query:
        parent1 = merge_node(labels=['A', 'B'], properties={'a': 1})
        parent2 = merge_node(labels=['A', 'B', 'other'], properties={'a': 2})
        child = merge_node(['C'], {'p': 1}, parents={parent1: 'req1', parent2: 'req2'}, versioned_labels=['other'])
    cypher = query.render_query(procedure_tag)
    database.neograph.run(cypher)
    result = database.neograph.run('MATCH (n: C) MATCH (p: A)-[]->(n) WITH n, collect(p.a) as a return n.p, a').to_table()
    assert len(result) == 1
    row = result[0]
    assert row[0] == 1
    assert row[1] == [1, 2]


def test_getattribute():
    with CypherQuery() as query:
        parent1 = merge_node(labels=['A', 'B'], properties={'a': 1, 'b': 2})
    query.returns(parent1.a, parent1.b)
    cypher = query.render_query()
    assert cypher.split('\n')[-1] == 'RETURN b0.a, b0.b'


def test_getitem():
    with CypherQuery() as query:
        parent1 = merge_node(labels=['A', 'B'], properties={'a': 1, 'b': 2})
    query.returns(parent1['a'], parent1['b'])
    cypher = query.render_query()
    assert cypher.split('\n')[-1] == "RETURN b0['a'], b0['b']"


@pytest.mark.parametrize('input_data', [1, [1,2], 'a', [1, 'a'], np.array([1,2])])
def test_input_data_retrievable(input_data):
    with CypherQuery() as query:
        data1 = CypherData(input_data, 'mydata')
        data2 = CypherData(input_data, 'mydata')
    query.returns(data1, data2)
    cypher = query.render_query()
    assert cypher == PREFIX + '\nRETURN $mydata0, $mydata1'


def test_unwind_input_list():
    with CypherQuery() as query:
        data = CypherData([1,2,3], 'mydata')
        merge_node(['B'], {'b': 1})
        with unwind(data) as number:
            node = merge_node(['A'], {'a': number})
        nodes = collect(node)
    query.returns(nodes)
    cypher = query.render_query()
    expected = [
        PREFIX,
        'MERGE (b0:B {b: 1}) ON CREATE SET b0.dbcreated = time0',
        'UNWIND $mydata0 as unwound_mydata0',
            'MERGE (a0:A {a: unwound_mydata0}) ON CREATE SET a0.dbcreated = time0',
        'WITH time0, b0, collect(a0) as as0',
        "RETURN as0"
    ]
    assert cypher == '\n'.join(expected)


def test_unwind_nested_contexts():
    with CypherQuery() as query:
        data = CypherData([1,2,3], 'mydata')
        merge_node(['B'], {'b': 1})
        with unwind(data) as number1:
            with unwind(data) as number2:
                node = merge_node(['A'], {'a': number1, 'b': number2})
            nodelist = collect(node)
        nodelistlist = collect(nodelist)
    query.returns(nodelistlist)
    cypher = query.render_query()
    expected = [
        PREFIX,
        'MERGE (b0:B {b: 1}) ON CREATE SET b0.dbcreated = time0',
        'UNWIND $mydata0 as unwound_mydata0',
            'UNWIND $mydata0 as unwound_mydata1',
                'MERGE (a0:A {a: unwound_mydata0, b: unwound_mydata1}) ON CREATE SET a0.dbcreated = time0',
            'WITH time0, b0, unwound_mydata0, collect(a0) as as0',
        'WITH time0, b0, collect(as0) as ass0',
        "RETURN ass0"
    ]
    assert cypher == '\n'.join(expected)


def test_autoclose_unwind_with_no_variables_carried_over():
    with CypherQuery() as query:
        data = CypherData([1,2,3], 'mydata')
        merge_node(['B'], {'b': 1})
        with unwind(data) as number:
            node = merge_node(['A'], {'a': number})
        with pytest.raises(ValueError):
            merge_node(['B'], {'b': number})



def test_closed_unwind_context_not_accessible():
    assert False


def test_multiply_returned_nested_unwinds_with_collect():
    assert False


def test_groupby_makes_dict():
    assert False


def test_groupby_fails_if_input_is_not_collection():
    assert False

