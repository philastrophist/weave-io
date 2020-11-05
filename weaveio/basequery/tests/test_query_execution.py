import pytest
import numpy as np
from astropy.table import Table

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


@pytest.mark.parametrize('columns,plural,colshape',
                         (['c_factor_as', True, (2, )], [['c_factor_as'], True, (1, 2)],
                          ['a_factor_a', False, tuple()], [['a_factor_a'], False, (1, )]),
                         ids=["['c_factor_as']", "[['c_factor_as']]", "['a_factor_a']", "[['a_factor_a']]"])
@pytest.mark.parametrize('idfilter,idshape', ([None, (5,)], [('1.fits', ), (1, )], [['1.fits'], (1,)]),
                         ids=["", "['1.fits']", "[['1.fits']]"])
def test_table_return_type(database, columns, idfilter, plural, idshape, colshape):
    """
    Test that [[colname]] type getitems always return astropy tables with the correct shape,
    plural colnames should make a list structure within it.
    """
    parent = database.hierarchyas
    if idfilter is not None:
        parent = parent.__getitem__(idfilter)
    structure = parent.__getitem__(columns)
    result = structure()
    if isinstance(columns, list):
        assert isinstance(result, Table)
        result = result.to_pandas().values.tolist()
    else:
        assert isinstance(result, list)
    result = np.asarray(result)
    expected = np.empty(idshape + colshape, dtype=str)
    expected[:] = 'a'
    np.testing.assert_array_equal(result, expected)
