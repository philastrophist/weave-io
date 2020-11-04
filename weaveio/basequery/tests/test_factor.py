import pytest

from weaveio.basequery.factor import SingleFactorFrozenQuery, ColumnFactorFrozenQuery, RowFactorFrozenQuery, TableFactorFrozenQuery
from weaveio.basequery.hierarchy import HomogeneousHierarchyFrozenQuery
from weaveio.basequery.query import NodeProperty, Node
from weaveio.utilities import quote


def test_single_hierarchy_direct_single_factor(data):
    """get a single factor from the parent hierarchy directly"""
    single = data.hierarchyas['1'].a_factor_a
    query = single.query
    assert isinstance(single, SingleFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyA', name='hierarchya0'), 'a_factor_a')
    assert len(query.returns) == 1   # a_factor_a and no indexer


def test_single_hierarchy_indirect_single_factor(data):
    """get a single factor from a hierarchy above the parent"""
    single = data.hierarchyas['1'].b_factor_a
    query = single.query
    assert isinstance(single, SingleFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_a')
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
    single = data.hierarchyas.b_factor_as
    query = single.query
    assert isinstance(single, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_a')
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
    single = data.hierarchyas[['1', '2']].b_factor_as
    query = single.query
    assert isinstance(single, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_a')
    assert len(query.returns) == 1   # b_factor_b and no indexer


def test_heterogeneous_plural_factor(data):
    factors = data.b_factor_as
    query = factors.query
    assert isinstance(factors, ColumnFactorFrozenQuery)
    assert query.returns[0] == NodeProperty(Node(label='HierarchyB', name='hierarchyb0'), 'b_factor_a')
    assert len(query.returns) == 1  # b_factor_b and no indexer


@pytest.mark.parametrize('typ', [tuple, list])
@pytest.mark.parametrize('hiers', [['a'], ['b'], ['a', 'b']])
def test_single_hierarchy_row_of_factors(data, typ, hiers):
    items, hiers = zip(*[(item, h) for h in hiers for item in [f'{h}_factor_{i}' for i in 'ab']])
    items = typ(items)
    row = data.hierarchyas['1'].__getitem__(items)
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


@pytest.mark.parametrize('hiers', [['a'], ['b'], ['a', 'b']])
@pytest.mark.parametrize('factor_intype', [tuple, list])
@pytest.mark.parametrize('factor_names', [['a'], ['b'], ['a', 'b']], ids=lambda x: str(x))
@pytest.mark.parametrize('idfilter', ['1', ['1', '2'], ('1', '2'), None],
ids=lambda v: f'hierarchies[{quote(v)}]'.replace('(', '').replace(')', '') if v is not None else 'hierarchies')
def test_tablelike_factors_by_getitem(data, factor_intype, hiers, factor_names, idfilter):
    items, hiers = zip(*[(item, h) for h in hiers for item in [f'{h}_factor_{i}' for i in factor_names]])
    items = factor_intype(items)

    structure = data.hierarchyas
    if idfilter is not None:
        structure = structure.__getitem__(idfilter)

    if isinstance(idfilter, (list, tuple)) or idfilter is None:
        if isinstance(items, (list, tuple)):
            querytype = TableFactorFrozenQuery
        else:
            querytype = ColumnFactorFrozenQuery
    else:  # scalar, therefore a single
        if isinstance(items, (list, tuple)):
            querytype = RowFactorFrozenQuery
        else:
            assert False, "bad arguments"  # this is never reached in this test

    table = structure.__getitem__(items)
    assert isinstance(table, querytype), f"data.hierarchyas[{idfilter}][{items}]"

    zippable_items = [items] if not isinstance(items, (tuple, list)) else items
    for i, (item, hier) in enumerate(zip(zippable_items, hiers)):
        prop = table.query.returns[i]
        assert prop.property_name == item
        assert prop.node.label == f'Hierarchy{hier.upper()}'
    if isinstance(items, list):
        assert table.return_keys == items
    elif isinstance(items, tuple):
        assert table.return_keys is None
    elif isinstance(items, str):
        pass
    else:
        assert False, "Bad arguments"


@pytest.mark.parametrize('hier', ['a', 'b'])
@pytest.mark.parametrize('idfilter', ['1', ['1', '2'], ('1', '2'), None],
ids=lambda v: f'hierarchies[{quote(v)}]'.replace('(', '').replace(')', '') if v is not None else 'hierarchies')
def test_direct_single_factors_by_getitem(data, idfilter, hier):
    if idfilter is not None:
        structure = data.hierarchyas.__getitem__(idfilter)
    else:
        structure = data.hierarchyas
    if isinstance(idfilter, (list, tuple)) or idfilter is None:
        querytype = ColumnFactorFrozenQuery
    else:  # scalar, therefore a single
        querytype = SingleFactorFrozenQuery
    factor_name = f'{hier}_factor_a'
    result = structure[factor_name]
    assert isinstance(result, querytype), str(result)
    prop = result.query.returns[0]
    assert prop.property_name == factor_name
    assert len(result.query.returns) == 1
    assert prop.node.label == f'Hierarchy{hier.upper()}'
