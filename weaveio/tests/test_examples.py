from astropy.table import Table
import pytest
from weaveio import *
from weaveio.opr3 import Data


@pytest.fixture
def data():
    return Data()

def test_example1a(data):
    runid = 1003453
    nsky = sum(data.runs[runid].targuses == 'S')()
    assert nsky == 100

def test_example1b(data):
    nsky = sum(data.runs.targuses == 'S', wrt=data.runs)()  # sum the number of skytargets with respect to their runs
    assert set(nsky) == {100, 160, 198, 200, 299, 360}


def test_example1c(data):
    nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
    query_table = data.runs[['id', nsky]]  # design a table by using the square brackets
    concrete_table = query_table()
    assert len(concrete_table) == 196
    assert set(concrete_table.sum0) == {100, 160, 198, 200, 299, 360}
    assert len(set(concrete_table.id)) == 196


def test_example2(data):
    yesterday = 57811
    runs = data.runs
    is_red = runs.camera == 'red'
    is_yesterday = floor(runs.exposure.mjd) == yesterday  # round to integer, which is the day
    runs = runs[is_red & is_yesterday]  # filter the runs first
    spectra = runs.l1single_spectra
    sky_spectra = spectra[spectra.targuse == 'S']
    table = sky_spectra[['wvl', 'flux']]
    assert count(table)() == 690


def test_example3(data):
    l2s = data.l2stacks
    l2s = l2s[(l2s.ob.mjd >= 57780) & any(l2s.fibre_target.surveys == '/WL.*/', wrt=l2s.fibre_target)]
    l2s = l2s[l2s['ha_6562.80_flux'] > 0]
    table = l2s[['ha_6562.80_flux', 'z']]()  # type: Table
    assert len(table) == 306
    table.sort('z')
    assert table[0]['z'] == -0.0028588480571366923
    assert table[0]['ha_6562.80_flux'] == 150790646.85417017

