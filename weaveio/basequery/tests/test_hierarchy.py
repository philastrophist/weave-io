import pytest
from weaveio.basequery.hierarchy import HomogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery, \
    HeterogeneousHierarchyFrozenQuery
from weaveio.basequery.query import Condition
from weaveio.basequery.tests.example_structures.one2one import HierarchyA, MyData


def test_begin_with_heterogeneous():
    data = MyData('.')
    hetero = data.handler.begin_with_heterogeneous()
    assert isinstance(hetero, HeterogeneousHierarchyFrozenQuery)


def test_get_homogeneous_from_data():
    data = MyData('.')
    hierarchyas = data.hierarchyas
    assert isinstance(hierarchyas, HomogeneousHierarchyFrozenQuery)
    assert hierarchyas._hierarchy is HierarchyA
    query = hierarchyas.query
    assert not query.predicates
    assert not query.returns
    assert not query.exist_branches
    assert not query.conditions
    assert len(query.matches) == 1 and query.matches[0][-1].name == 'hierarchya0'


def test_index_by_single_identifier():
    data = MyData('.')
    single = data.hierarchyas['1']
    query = single.query
    assert isinstance(single, SingleHierarchyFrozenQuery)
    assert single._hierarchy is HierarchyA
    assert not query.predicates
    assert not query.returns
    assert not query.exist_branches
    assert len(query.matches) == 1 and query.matches[0].nodes[-1].name == 'hierarchya0'
    assert query.conditions == Condition(query.matches[0].nodes[-1].id, '=', '1')
