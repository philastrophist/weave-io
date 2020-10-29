from copy import copy

import pytest
from weaveio.basequery.hierarchy import HomogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery, \
    HeterogeneousHierarchyFrozenQuery
from weaveio.basequery.query import Condition, Generator, AmbiguousPathError, Node
from weaveio.basequery.tests.example_structures.one2one import HierarchyA, MyData


hierarchies = MyData('.').hierarchies

def test_begin_with_heterogeneous(data):
    hetero = data.handler.begin_with_heterogeneous()
    assert isinstance(hetero, HeterogeneousHierarchyFrozenQuery)


@pytest.mark.parametrize('hier', hierarchies)
def test_get_homogeneous_from_data(data, hier):
    hierarchies = data.__getattr__(hier.plural_name)
    assert isinstance(hierarchies, HomogeneousHierarchyFrozenQuery)
    assert hierarchies._hierarchy is hier
    query = hierarchies.query
    assert not query.predicates
    assert not query.returns
    assert not query.exist_branches
    assert not query.conditions
    assert len(query.matches) == 1 and query.matches[0][-1].name == f'{hier.singular_name}0'


@pytest.mark.parametrize('hier', hierarchies)
def test_return_homogeneous_with_indexer(data, hier):
    hierarchies = data.__getattr__(hier.plural_name)
    query = hierarchies._prepare_query()
    indexer = Node(name='none0')
    assert query.returns == [query.current_node, indexer]


def test_index_by_single_identifier(data):
    single = data.hierarchyas['1']
    query = single.query
    assert isinstance(single, SingleHierarchyFrozenQuery)
    assert single._hierarchy is HierarchyA
    assert not query.predicates
    assert not query.returns
    assert not query.exist_branches
    assert len(query.matches) == 1 and query.matches[0].nodes[-1].name == 'hierarchya0'
    assert query.conditions == Condition(query.matches[0].nodes[-1].id, '=', '1')


def test_get_homogeneous_from_homogeneous(data):
    homo = data.hierarchyas.hierarchybs
    assert isinstance(homo, HomogeneousHierarchyFrozenQuery)
    assert len(homo.query.matches) == 2


def test_get_single_from_homogeneous_invalid(data):
    with pytest.raises(AmbiguousPathError):
        data.hierarchyas.hierarchyb


def test_get_single_from_single_ascending(data):
    single = data.hierarchyas['1'].hierarchyb
    assert isinstance(single, SingleHierarchyFrozenQuery)

