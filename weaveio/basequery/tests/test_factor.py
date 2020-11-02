import pytest

from weaveio.basequery.factor import SingleFactorFrozenQuery, ColumnFactorFrozenQuery, RowFactorFrozenQuery
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


def test_heterogeneous_plural_factor(data):
    factors = data.b_factor_bs
    query = factors.query
    assert isinstance(factors, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_b')
    assert len(query.returns) == 1  # b_factor_b and no indexer


@pytest.mark.parametrize('typ', [tuple, list])
@pytest.mark.parametrize('hiers', [['a'], ['b'], ['a', 'b']])
def test_single_hierarchy_row_of_factors(data, typ, hiers):
    items, hiers = zip(*[(item, h) for h in hiers for item in [f'{h}_factor_{i}' for i in 'ab']])
    row = data.hierarchyas.__getitem__(typ(items))
    assert isinstance(row, RowFactorFrozenQuery)
    for i, (item, hier) in enumerate(zip(items, hiers)):
        prop = row.query.returns[i]
        assert prop.property_name == item
        assert prop.node.label == f'Hierarchy{hier.upper()}'
    if typ is list:
        assert row.return_keys == items
    elif typ is tuple:
        assert row.return_keys is None
    else:
        assert False, "Bad arguments"


def test_homogeneous_hierarchy_table_of_factors(data):
    assert False

def test_single_hierarchy_direct_single_factor_by_getitem(data):
    assert False

def test_single_hierarchy_direct_plural_factor_by_getitem(data):
    assert False