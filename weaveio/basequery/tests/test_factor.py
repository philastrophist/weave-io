import pytest

from weaveio.basequery.factor import SingleFactorFrozenQuery, ColumnFactorFrozenQuery
from weaveio.basequery.query import NodeProperty, Node


def test_single_hierarchy_direct_single_factor(data):
    """get a single factor from the parent hierarchy directly"""
    single = data.hierarchyas['1'].a_factor_a
    query = single.query
    assert isinstance(single, SingleFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyA', name='hierarchya0'), 'a_factor_a')
    assert len(query.returns) == 1   # a_factor_a and no indexer


def test_single_hierarchy_indirect_single_factor(data):
    """get a single factor from a hierarchy above the parent"""
    single = data.hierarchyas['1'].b_factor_b
    query = single.query
    assert isinstance(single, SingleFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_b')
    assert len(query.returns) == 1   # b_factor_b and no indexer


def test_single_hierarchy_fails_with_unknown_name(data):
    with pytest.raises(AttributeError):
        data.hierarchyas['1'].unknown


def test_homogeneous_hierarchy_fails_with_unknown_name(data):
    with pytest.raises(AttributeError):
        data.hierarchyas.unknowns


def test_homogeneous_hierarchy_direct_plural_factor(data):
    single = data.hierarchyas.a_factor_as
    query = single.query
    assert isinstance(single, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyA', name='hierarchya0'), 'a_factor_a')
    assert len(query.returns) == 1   # a_factor_a and no indexer


def test_homogeneous_hierarchy_indirect_plural_factor(data):
    single = data.hierarchyas.b_factor_bs
    query = single.query
    assert isinstance(single, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_b')
    assert len(query.returns) == 1   # b_factor_b and no indexer


def test_identified_homogeneous_hierarchy_fails_with_unknown_name(data):
    with pytest.raises(AttributeError):
        data.hierarchyas[['1', '2']].unknowns


def test_identified_homogeneous_hierarchy_direct_plural_factor(data):
    single = data.hierarchyas[['1', '2']].a_factor_as
    query = single.query
    assert isinstance(single, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyA', name='hierarchya0'), 'a_factor_a')
    assert len(query.returns) == 1   # a_factor_a and no indexer


def test_identified_homogeneous_hierarchy_indirect_plural_factor(data):
    single = data.hierarchyas[['1', '2']].b_factor_bs
    query = single.query
    assert isinstance(single, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_b')
    assert len(query.returns) == 1   # b_factor_b and no indexer