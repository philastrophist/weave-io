import pytest
import numpy as np

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


def test_uchained_queries(database):
    first = database.hierarchyas
    second = first.hierarchybs
    first()
    second()


def test_multiple_ids(database):
    names = ['1.fits', '2.fits', '1.fits']
    a = database.hierarchyas[names]()
    assert [i.id for i in a] == names


def test_multiple_ids_keyerror(database):
    names = ['1.fits', '2.fits', 'nan']
    with pytest.raises(KeyError, match='nan'):
        database.hierarchyas[names]()


def test_single_factor_is_scalar(database):
    assert database.hierarchyas['1.fits'].a_factor_a() == 'a'


def test_column_factor_is_vector(database):
    np.testing.assert_array_equal(database.hierarchyas['1.fits', '2.fits'].a_factor_as(), ['a', 'a'])


