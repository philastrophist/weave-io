import pytest
import numpy as np

from weaveio.writequery import CypherQuery, merge_node, match_node, unwind, collect, groupby, CypherData, merge_relationship, merge_node_relationship, set_version
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
    cypher = query.render_query()[0].split('\n')
    assert cypher[-2] == "MERGE (b0:A:B {a: 1, b: 2, c: [1, 2, 3]}) ON CREATE SET b0.dbcreated = time0"


def test_match_node():
    with CypherQuery() as query:
        match_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
    cypher = query.render_query()[0].split('\n')
    assert cypher[-2] == "MATCH (b0:A:B {a: 1, b: 2, c: [1, 2, 3]})"


def test_merge_many2one_with_one():
    with CypherQuery() as query:
        parent = match_node(labels=['A', 'B'], properties={'a': 1, 'b': 2, 'c': [1,2,3]})
        child = merge_node(['C'], {'p': 1}, parents={parent: 'req'}, versioned_labels=['A'])
    cypher = query.render_query()[0].split('\n')
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
    cypher = query.render_query()[0]
    expected = [PREFIX,
        "MATCH (b0:A:B {a: 1})",
        "MATCH (other0:A:B:Other {a: 2})",
        "WITH *, [[b0, 'req1', {order: 0}]] as bspec0",
        "WITH *, [[other0, 'req2', {order: 0}]] as otherspec0",
        "WITH *, bspec0+otherspec0 as specs0",
        "CALL custom.multimerge(specs0, ['C'], {p: 1}, {dbcreated: time0}, {dbcreated: time0}) YIELD child as c0",
        "CALL custom.version(specs0, c0, ['Other'], 'version') YIELD version as version0",
        "RETURN time0"
                ]
    expected = '\n'.join(expected)
    assert cypher == expected


def test_merge_many2one_with_mixture_run(procedure_tag, write_database):
    with CypherQuery() as query:
        parent1 = merge_node(labels=['A', 'B'], properties={'a': 1})
        parent2 = merge_node(labels=['A', 'B', 'other'], properties={'a': 2})
        child = merge_node(['C'], {'p': 1}, parents={parent1: 'req1', parent2: 'req2'}, versioned_labels=['other'])
    cypher = query.render_query(procedure_tag)[0]
    write_database.neograph.run(cypher)
    result = write_database.neograph.run('MATCH (n: C) MATCH (p: A)-[]->(n) WITH n, collect(p.a) as a return n.p, a').to_table()
    assert len(result) == 1
    row = result[0]
    assert row[0] == 1
    assert row[1] == [1, 2]


def test_getitem():
    with CypherQuery() as query:
        parent1 = merge_node(labels=['A', 'B'], properties={'a': 1, 'b': 2})
    query.returns(parent1['a'], parent1['b'])
    cypher = query.render_query()[0]
    assert cypher.split('\n')[-1] == "RETURN b0['a'], b0['b']"


@pytest.mark.parametrize('input_data', [1, [1,2], 'a', [1, 'a'], np.array([1,2])])
def test_input_data_retrievable(input_data):
    with CypherQuery() as query:
        data1 = CypherData(input_data, 'mydata')
        data2 = CypherData(input_data, 'mydata')
    query.returns(data1, data2)
    cypher = query.render_query()[0]
    assert cypher == PREFIX + '\nRETURN $mydata0, $mydata1'


def test_unwind_input_list():
    with CypherQuery() as query:
        data = CypherData([1,2,3], 'mydata')
        merge_node(['B'], {'b': 1})
        with unwind(data) as number:
            node = merge_node(['A'], {'a': number})
        nodes = collect(node)
    query.returns(nodes)
    cypher = query.render_query()[0]
    expected = [
        PREFIX,
        'MERGE (b0:B {b: 1}) ON CREATE SET b0.dbcreated = time0',
        'WITH * UNWIND $mydata0 as unwound_mydata0',
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
    cypher = query.render_query()[0]
    expected = [
        PREFIX,
        'MERGE (b0:B {b: 1}) ON CREATE SET b0.dbcreated = time0',
        'WITH * UNWIND $mydata0 as unwound_mydata0',
            'WITH * UNWIND $mydata0 as unwound_mydata1',
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
    cypher = query.render_query()[0].split('\n')
    assert cypher[-2] == 'WITH time0, b0'


def test_closed_unwind_context_not_accessible():
    with CypherQuery() as query:
        data = CypherData([1,2,3], 'mydata')
        merge_node(['B'], {'b': 1})
        with unwind(data) as number:
            node = merge_node(['A'], {'a': number})
        with pytest.raises(ValueError):
            merge_node(['B'], {'b': number})


def test_unwind_that_returns_multiple_variables():
    with CypherQuery() as query:
        data = CypherData(['1', '2', '3'], 'mydata')
        with unwind(data, data) as (a, b):
            pass
    cypher = query.render_query()[0]
    expected = [
        PREFIX,
        "WITH *, apoc.coll.max([x in [$mydata0,$mydata0] | SIZE(x)])-1 as m",
        "UNWIND range(0, m) as i0 WITH *, $mydata0[i0] as unwound_mydata0, $mydata0[i0] as unwound_mydata1",
        "WITH time0",
        "RETURN time0"
    ]
    assert cypher == '\n'.join(expected)


def test_unwind_that_returns_multiple_variables_with_enumerate():
    with CypherQuery() as query:
        data = CypherData(['1', '2', '3'], 'mydata')
        with unwind(data, enumerated=True) as (d, i):
            pass
        indexes = collect(i)
    cypher = query.render_query()[0]
    expected = [
        PREFIX,
        "WITH *, apoc.coll.max([x in [$mydata0] | SIZE(x)])-1 as m",
        "UNWIND range(0, m) as i0 WITH *, $mydata0[i0] as unwound_mydata0",
        "WITH time0, collect(i0) as is0",
        "RETURN time0"
    ]
    assert cypher == '\n'.join(expected)



def test_unwind_collect_unwind_collect():
    with CypherQuery() as query:
        data = CypherData(['1', '2', '3'], 'mydata')
        with unwind(data, enumerated=True) as (d, i):
            pass
        ds = collect(d)
        with unwind(ds) as d:
            pass
        ds = collect(d)
    cypher = query.render_query()[0]
    expected = [
        PREFIX,
        "WITH *, apoc.coll.max([x in [$mydata0] | SIZE(x)])-1 as m",
        "UNWIND range(0, m) as i0 WITH *, $mydata0[i0] as unwound_mydata0",
        "WITH time0, collect(unwound_mydata0) as unwound_mydatas0",
        "WITH * UNWIND unwound_mydatas0 as unwound_unwound_mydatas0",
        "WITH time0, collect(unwound_unwound_mydatas0) as unwound_unwound_mydatass0",
        "RETURN time0"
    ]
    assert cypher == '\n'.join(expected)


def test_accessing_unwound_variable_after_unwinding_is_not_allowed():
    with CypherQuery() as query:
        data = CypherData(['1', '2', '3'], 'mydata')
        with unwind(data, enumerated=True) as (d, i):
            pass
        ds = collect(d)
        with unwind(ds) as d:
            pass
        with pytest.raises(ValueError):
            groupby(ds, 'property')


def test_groupby_makes_dict(procedure_tag, write_database):
    with CypherQuery() as query:
        data = CypherData(['1', '2', '3'], 'mydata')
        merge_node(['B'], {'b': 1})
        with unwind(data) as number:
            node = merge_node(['A'], {'a': number})
        nodes = collect(node)
        nodedict = groupby(nodes, 'a')
    query.returns(nodedict['1'].a, nodedict['2']['a'], nodedict['3']['a'], nodedict['0']['a'])
    cypher, parameters = query.render_query(procedure_tag)
    result = write_database.neograph.run(cypher, parameters=parameters).to_table()
    assert result[0] == ('1', '2', '3', None)


def test_groupby_fails_if_input_is_not_collection():
    with CypherQuery() as query:
        data = CypherData(['1', '2', '3'], 'mydata')
        b = merge_node(['B'], {'b': 1})
        with unwind(data) as number:
            node = merge_node(['A'], {'a': number})
        with pytest.raises(TypeError):
            nodedict = groupby(b, 'b')


@pytest.fixture(scope='class')
def bigquery(write_database, procedure_tag):
    with CypherQuery() as query:
        checksums = CypherData(list(range(10)),'checksums')
        runids = CypherData(['runid1', 'runid2', 'runid3'], 'runids')
        with unwind(checksums, enumerated=True) as (checksum, i):
            fibretarget = merge_node(['target'], {'id': i})
        fibretargets = collect(fibretarget)
        with unwind(checksums, fibretargets) as (checksum, fibretarget):
            with unwind(runids) as runid:
                run = merge_node(['run'], {'id': runid})
                raw = merge_node(['raw'], {}, parents={run: 'req'}, versioned_labels=['run'])
                spec = merge_node(['SingleSpectrum'], {}, parents={raw: 'req', fibretarget: 'req'}, versioned_labels=['raw', 'target'])
            specs = collect(spec)
            stack = merge_node(['stack'], {'checksum': checksum}, parents={specs: 'req'})
        stacks = collect(stack)
        merge_node(['StackFile'], {'fname': 'fname'}, parents={stacks: 'req'})
        run = match_node(['run'], {'id': 'runid1'})
        raw = match_node(['raw'], {}, parents={run: 'req'})
        merge_node(['rawfile'], {'fname': 'rawfname'}, parents={raw: 'req'})
    cypher, data = query.render_query(procedure_tag)
    write_database.neograph.run(cypher, parameters=data)
    return cypher, data, write_database


@pytest.mark.usefixtures("bigquery")  # careful, there seems to be a bug with pytest and this database is not separate!
class TestBigQuery:
    def test_spec_has_run_and_raw(self, bigquery):
        cypher, data, database = bigquery
        result = database.neograph.run('match (s:SingleSpectrum) optional match (raw:Raw)-->(s)<--(t:Target) return s, raw, t').to_data_frame()
        assert len(result) == 30
        assert not result['raw'].isnull().any()

    @pytest.mark.parametrize('v,n,version', [('Run', 3, None), ('Raw', 3, 0), ('SingleSpectrum', 30, 0)])
    def test_raw_has_version0(self, v, n, version, bigquery):
        cypher, data, database = bigquery
        result = database.neograph.run(f'match (v:{v}) return v.version').to_data_frame()
        assert len(result) == n
        if version is not None:
            assert (result['v.version'] == version).all()
        else:
            assert (result['v.version'].isnull()).all()

    def test_stack_has_3_single_spectra(self, bigquery):
        cypher, data, database = bigquery
        result = database.neograph.run(f'match (stack:Stack) optional match (stack)<--(single:SingleSpectrum) '
                                       f'with stack, collect(single) as singles return stack, singles').to_data_frame()
        assert len(result) == 10
        assert all(len(s) == 3 for s in result['singles'])

    def test_stackfile_has_all_stacks(self, bigquery):
        cypher, data, database = bigquery
        result = database.neograph.run(f'match (stackfile:StackFile) optional match (stackfile)<--(stack:Stack) '
                                       f'with stackfile, collect(stack.checksum) as stacks return stackfile, stacks').to_data_frame()
        assert len(result) == 1
        assert all(set(s) == set(range(10)) for s in result['stacks'])

    def test_rawfile_is_attached_to_a_single_raw(self, bigquery):
        cypher, data, database = bigquery
        result = database.neograph.run('match (f:Rawfile) optional match (run:Run)-->(raw:Raw)-->(f) return f, raw, run.id').to_data_frame()
        assert len(result) == 1
        assert result['run.id'][0] == 'runid1'


def test_merge_relationship():
    with CypherQuery() as query:
        a = merge_node(['a'], {})
        b = merge_node(['b'], {})
        merge_relationship(a, b, 'rel', {'prop': 'a'})
    cypher = query.render_query()[0]
    assert cypher == '\n'.join([PREFIX,
                                "MERGE (a0:A {}) ON CREATE SET a0.dbcreated = time0",
                                "MERGE (b0:B {}) ON CREATE SET b0.dbcreated = time0",
                                "MERGE (a0)-[rel0:rel {prop: 'a'}]->(b0) ON CREATE SET rel0.dbcreated = time0",
                                "RETURN time0"
                                ])


def test_merge_node_relationship_with_one():
    with CypherQuery() as query:
        a = merge_node(['a'], {})
        merge_node_relationship(['b'], {'prop': 'b'},  [(a, 'rel', {'prop': 'a'})])
    cypher = query.render_query()[0]
    assert cypher == '\n'.join([PREFIX,
                                "MERGE (a0:A {}) ON CREATE SET a0.dbcreated = time0",
                                "MERGE (a0)-[rel0:rel {prop: 'a'}]->(b0:B {prop: 'b'}) "
                                "ON CREATE SET rel0.dbcreated = time0, b0.dbcreated = time0",
                                "RETURN time0"
                                ])


def test_merge_node_relationship_with_two():
    with CypherQuery() as query:
        a = merge_node(['a'], {})
        b = merge_node(['b'], {})
        merge_node_relationship(['c'], {'prop': 'c'},  [(a, 'rel', {'prop': 'a'}),
                                                        (b, 'rel', {'prop': 'b'})])
    cypher = query.render_query()[0]
    assert cypher == '\n'.join([PREFIX,
                                "MERGE (a0:A {}) ON CREATE SET a0.dbcreated = time0",
                                "MERGE (b0:B {}) ON CREATE SET b0.dbcreated = time0",
                                "MERGE (a0)-[rel0:rel {prop: 'a'}]->(c0:C {prop: 'c'})<-[rel1:rel {prop: 'b'}]-(b0) "
                                "ON CREATE SET rel0.dbcreated = time0, c0.dbcreated = time0, rel1.dbcreated = time0",
                                "RETURN time0"
                                ])


def test_versioning(write_database):
    with CypherQuery() as query:
        a = merge_node(['a'], {})
        b = merge_node(['b'], {})
        c1 = merge_node_relationship(['c'], {'prop': 'c1'},  [(a, 'rel', {'prop': 'a'}),
                                                        (b, 'rel', {'prop': 'b'})])
        set_version([a, b], ['rel', 'rel'], 'C', c1)
        c2 = merge_node_relationship(['c'], {'prop': 'c2'}, [(a, 'rel', {'prop': 'a'}),
                                                            (b, 'rel', {'prop': 'b'})])
        set_version([a, b], ['rel', 'rel'], 'C', c2)
        c2dupl = merge_node_relationship(['c'], {'prop': 'c2'}, [(a, 'rel', {'prop': 'a'}),
                                                            (b, 'rel', {'prop': 'b'})])
        set_version([a, b], ['rel', 'rel'], 'C', c2dupl)
    cypher = query.render_query()[0]
    write_database.neograph.run(cypher)
    results = write_database.neograph.run('MATCH (c:C) return c.prop, c.version ORDER BY c.version').to_table()
    assert results[0] == ('c1', 0)
    assert results[1] == ('c2', 1)
    assert len(results) == 2
