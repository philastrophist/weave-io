import pytest
from weaveio.basequery.hierarchy import HomogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery
from weaveio.basequery.tests.example_structures import HierarchyA, MyData


def test_get_homogeneous_from_data():
    data = MyData('.')
    hierarchyas = data.hierarchyas
    assert isinstance(hierarchyas, HomogeneousHierarchyFrozenQuery)
    assert hierarchyas.hierarchy is HierarchyA


def test_index_by_single_identifier():
    data = MyData('.')
    hierarchyas = data.hierarchyas
    single = hierarchyas['idname']
    assert isinstance(single, SingleHierarchyFrozenQuery)
    assert single.hierarchy is HierarchyA



