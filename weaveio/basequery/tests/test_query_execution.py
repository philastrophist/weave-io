import pytest

from weaveio.basequery.tests.example_structures.one2one import MyData, HierarchyA, HierarchyB


def test_instantiate_db(database):
    pass


def test_return_homogeneous(database):
    for h in database.hierarchyas():
        assert isinstance(h, HierarchyA)
        assert h.a_factor_a == 'a'
        assert h.a_factor_b == 'b'


def test_return_single(database):
    b = database.hierarchyas['1.fits'].hierarchyb()
    assert isinstance(b, HierarchyB)
    assert b.otherid == '1.fits'


def test_empty_id_raise_keyerror(database):
    with pytest.raises(KeyError, match="nonexistent_id"):
        database.hierarchyas['nonexistent_id'].hierarchyb()
