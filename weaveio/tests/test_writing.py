import pytest

from weaveio.writequery import CypherQuery, merge_node, match_node, unwind, collect, groupby
from weaveio.graph import Graph


PREFIX = "WITH *, timestamp() as time0\n"


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
    assert cypher[-1] == "MERGE (b0:A:B {a: 1, b: 2, c: [1, 2, 3]})"


def test_match_node():
    with CypherQuery() as query:
        match_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
    cypher = query.render_query().split('\n')
    assert cypher[-1] == "MATCH (b0:A:B {a: 1, b: 2, c: [1, 2, 3]})"


def test_merge_many2one_with_one():
    with CypherQuery() as query:
        parent = match_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
        child = merge_node(['C'], {'p': 1}, parents={parent: 'req'}, versioned_labels=['A'])
    cypher = query.render_query().split('\n')
    assert cypher[1] == "MATCH (b0:A:B {a: 1, b: 2, c: [1, 2, 3]})"
    assert cypher[2] == "WITH *, [[b0, 'req', {order: 0}]] as bspec0"
    assert cypher[3] == "WITH *, bspec0 as specs0"
    assert cypher[-2] == "CALL custom.multimerge(specs0, ['C'], {p: 1}, {dbcreated: time0}, {}) YIELD child as c0"
    assert cypher[-1] == "CALL custom.version(specs0, c0, ['A'], 'version') YIELD version as version0"


def test_merge_many2one_with_mixture():
    assert False