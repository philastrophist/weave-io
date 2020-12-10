import pytest
import py2neo


def test_merge_many_to_one_in_isolation(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes
        
        MERGE (b1: B {{id: 1}})
        
        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)
        
        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames
        
        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification
        
        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 1


def test_does_not_duplicate_on_same(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 1



def test_creates_two_on_subset(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 10) as i  // make a subset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 100) as i  // do it again
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 2



def test_creates_two_on_superset(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 200) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 100) as i  // do it again
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 2



def test_creates_two_on_rel_change(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 100) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel_different', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 100) as i  // do it again
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 2




def test_creates_two_on_prop_change(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 100) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai, something_else: 1}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 100) as i  // do it again
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 2


def test_creates_one_to_one_correctly_on_spec_change(database: py2neo.Graph, procedure_tag: str):
    q = f"""
    MERGE (a: A {{id: 1}})
    WITH [[a, 'rel', {{order: 0}}]] as specs
    CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 'match'}}, {{oncreate:1}}, {{oncreate: 1}}) YIELD child
    
    MERGE (a: A {{id: 1}})
    WITH [[a, 'rel', {{order: 0}}]] as specs
    CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 'match'}}, {{oncreate: 2}}, {{oncreate: 2}}) YIELD child

    MERGE (a: A {{id: 2}})
    WITH [[a, 'rel', {{order: 0}}]] as specs
    CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 'match'}}, {{oncreate:3}}, {{oncreate: 3}}) YIELD child

    MATCH (n:MyLabel)
    MATCH (n)<-[r]-(a:A)
    RETURN n.oncreate, r.oncreate
    ORDER BY n.oncreate
    """
    t = database.run(q).to_table()
    assert len(t) == 2
    assert t[0][0] == 1
    assert t[0][1] == 1
    assert t[1][0] == 3
    assert t[1][1] == 3


def test_creates_one_to_one_correctly_on_child_change(database: py2neo.Graph, procedure_tag: str):
    q = f"""
    MERGE (a: A {{id: 1}})
    WITH [[a, 'rel', {{order: 0}}]] as specs
    CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 'match'}}, {{oncreate:1}}, {{oncreate: 1}}) YIELD child
    
    MERGE (a: A {{id: 1}})
    WITH [[a, 'rel', {{order: 0}}]] as specs
    CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 'match'}}, {{oncreate: 2}}, {{oncreate: 2}}) YIELD child

    MERGE (a: A {{id: 1}})
    WITH [[a, 'rel', {{order: 0}}]] as specs
    CALL custom.multimerge{procedure_tag}(specs, ['MyLabel', 'MyOtherLabel'], {{match: 'match'}}, {{oncreate:3}}, {{oncreate: 3}}) YIELD child

    MATCH (n:MyLabel)
    MATCH (n)<-[r]-(a:A)
    RETURN n.oncreate, r.oncreate
    ORDER BY n.oncreate
    """
    t = database.run(q).to_table()
    assert len(t) == 2
    assert t[0][0] == 1
    assert t[0][1] == 1
    assert t[1][0] == 3
    assert t[1][1] == 3


def test_creates_three_on_super_and_subsets(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 200) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 10) as i  // make subset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    assert len(t) == 3


@pytest.mark.parametrize('beforelabels,afterlabels', [
    ('["Label1"]', '["Label1"]'),
    ('["Label1"]', '["Label2"]'),
    ('["Label1"]', '["Label1", "Label2"]'),
    ('["Label1", "Label2"]', '["Label1"]'),
])
def test_creates_new_when_label_changes(database: py2neo.Graph, procedure_tag: str,
                                        beforelabels, afterlabels):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, {beforelabels}, {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 100) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, {afterlabels}, {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        MATCH (n)
        WHERE NOT n:A AND NOT n:B
        RETURN n
        """
    t = database.run(q).to_table()
    if beforelabels == afterlabels:
        assert len(t) == 1
    else:
        assert len(t) == 2


@pytest.mark.parametrize('beforeproperties,afterproperties,diff', [
    ('{a: 1}', '{a: 1}', False),
    ('{a: 1}', '{a: 2}', True),
    ('{a: 1}', '{a: 1, b: 2}', True),  # we merge based on given properties only
    ('{}', '{}', False),
    ('{}', '{a: 1}', True),
    ('{a:1}', '{}', False),  # we merge based on given properties only
    ('{a:1, b: 2}', '{a: 1}', False),  # we merge based on given properties only
])
def test_creates_new_when_properties_change(database: py2neo.Graph, procedure_tag: str,
                                            beforeproperties, afterproperties, diff):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {beforeproperties}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here
        
        UNWIND range(0, 100) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {afterproperties}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        MATCH (n:MyLabel)
        RETURN n
        """
    t = database.run(q).to_table()
    if not diff:
        assert len(t) == 1
    else:
        assert len(t) == 2



def test_createprops_rels_are_created(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 1}}, {{create: 'node1'}}, {{create: 'rel1'}}) YIELD child
        WITH collect(child) as c // just to reset here

        UNWIND range(0, 200) as i  // make a superset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 2}}, {{create: 'node2'}}, {{create: 'rel2'}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 100) as i  // and again
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 1}}, {{create: 'node3'}}, {{create: 'rel3'}}) YIELD child
        WITH collect(child) as c // just to reset here

        MATCH (n:MyLabel)
        MATCH (other)-[r]->(n)
        RETURN n, collect(n.create) as created, collect(n.match) as matched, collect(r.create) as createdrel
        ORDER BY n.match
        """
    t = database.run(q).to_table()
    assert len(t) == 2
    assert all(i == 'node1' for i in t[0][1])  # create node prop
    assert all(i == 1 for i in t[0][2])  # match
    assert all(i == 'rel1' for i in t[0][3])  # create rel prop
    #
    assert all(i == 'node2' for i in t[1][1])  # create node prop
    assert all(i == 2 for i in t[1][2])  # match
    assert all(i == 'rel2' for i in t[1][3])  # create rel prop


def test_leaving_properties_out_returns_all_children_without_modification(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c // just to reset here

        UNWIND range(0, 10) as i  // make a subset
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 1}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c  // just to reset here

        UNWIND range(0, 100) as i  // do it again
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{match: 2}}, {{}}, {{}}) YIELD child
        WITH collect(child) as c, specs // just to reset here

        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        RETURN child.match
        ORDER BY child.match
        """
    t = database.run(q).to_table()
    assert len(t) == 2  # on the ones with 100 A nodes
    assert t[0][0] == 2
    assert t[1][0] is None  # these are

    assert database.run('MATCH (n: MyLabel) RETURN count(n)').to_table()[0][0] == 3