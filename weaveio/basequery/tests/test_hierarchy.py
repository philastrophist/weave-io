import pytest
from weaveio.basequery.hierarchy import HomogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery, \
    HeterogeneousHierarchyFrozenQuery
from weaveio.basequery.tests.example_structures import HierarchyA, MyData


def test_begin_with_heterogeneous():
    data = MyData('.')
    hetero = data.handler.begin_with_heterogeneous()
    assert isinstance(hetero, HeterogeneousHierarchyFrozenQuery)


def test_get_homogeneous_from_data():
    data = MyData('.')
    hierarchyas = data.hierarchyas
    assert isinstance(hierarchyas, HomogeneousHierarchyFrozenQuery)
    assert hierarchyas._hierarchy is HierarchyA


def test_index_by_single_identifier():
    data = MyData('.')
    single = data.hierarchyas['idname']
    assert isinstance(single, SingleHierarchyFrozenQuery)
    assert single._hierarchy is HierarchyA


