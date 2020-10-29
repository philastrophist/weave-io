"""
Tests to ensure that
Query(predicates, return_properties, return_nodes, conditions, exists)
are written to neo4j correctly.e

I.e.
"""
import pytest
from weaveio.basequery.query import FullQuery, Generator, Path, Condition


def test_matches_must_overlap():
    g = Generator()
    a, b, c, d = g.nodes('A', 'B', 'C', 'D')
    path1 = Path(a, '->', b)
    path2 = Path(c, '->', d)
    with pytest.raises(ValueError):
        FullQuery([path1, path2])


def test_current_node():
    g = Generator()
    a, b, c = g.nodes('A', 'B', 'C')
    path1 = Path(a, '->', b)
    path2 = Path(b, '->', c)
    query = FullQuery([path1, path2])
    assert query.current_node == c


def test_matches_only():
    g = Generator()
    a, b, c = g.nodes('A', 'B', 'C')
    path1 = Path(a, '->', b)
    path2 = Path(b, '->', c)
    query = FullQuery([path1, path2])
    q = query.to_neo4j()
    assert q == f'MATCH (a0:A)->(b0:B)\nMATCH (b0)->(c0:C)\nRETURN'


def test_return_nodes():
    g = Generator()
    a, b, c = g.nodes('A', 'B', 'C')
    path1 = Path(a, '->', b)
    path2 = Path(b, '->', c)
    query = FullQuery([path1, path2], returns=[c, a])
    q = query.to_neo4j()
    assert q == f'MATCH (a0:A)->(b0:B)\nMATCH (b0)->(c0:C)\nRETURN c0, a0'


def test_return_properties():
    g = Generator()
    a, b, c = g.nodes('A', 'B', 'C')
    path1 = Path(a, '->', b)
    path2 = Path(b, '->', c)
    query = FullQuery([path1, path2], returns=[c.property_c, a.property_a])
    q = query.to_neo4j()
    assert q == f'MATCH (a0:A)->(b0:B)\nMATCH (b0)->(c0:C)\nRETURN c0.property_c, a0.property_a'


def test_return_nodes_properties():
    g = Generator()
    a, b, c = g.nodes('A', 'B', 'C')
    path1 = Path(a, '->', b)
    path2 = Path(b, '->', c)
    query = FullQuery([path1, path2], returns=[c.property_c, a.property_a, b, c])
    q = query.to_neo4j()
    assert q == f'MATCH (a0:A)->(b0:B)\nMATCH (b0)->(c0:C)\nRETURN c0.property_c, a0.property_a, b0, c0'


def test_condition_on_property():
    g = Generator()
    a, b, c = g.nodes('A', 'B', 'C')
    path1 = Path(a, '->', b)
    path2 = Path(b, '->', c)
    condition = Condition(a.id, '=', 'idvalue')
    query = FullQuery([path1, path2], condition)
    q = query.to_neo4j()
    assert q == f'MATCH (a0:A)->(b0:B)\nMATCH (b0)->(c0:C)\nWHERE (a0.id = \'idvalue\')\nRETURN'
