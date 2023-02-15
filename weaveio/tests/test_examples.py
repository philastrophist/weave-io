from astropy.io import fits
from astropy.table import Table
import pytest
from weaveio import *

@pytest.fixture(scope='module')
def nsky_expected(data):
    return {sum(Table(fits.open(data.rootdir / p)['FIBTABLE'].data)['TARGUSE'] == 'S') for p in data.raw_files.path()}


def test_example1a(data):
    runid = 1003453
    nsky = sum(data.runs[runid].targuses == 'S')()
    q = data.runs[runid].raw_file.path
    t = Table(fits.open(data.rootdir / q()[0])['FIBTABLE'].data)
    assert nsky == sum(t['TARGUSE'] == 'S')

def test_example1b(data, nsky_expected):
    nsky = set(sum(data.runs.targuses == 'S', wrt=data.runs)())  # sum the number of skytargets with respect to their runs
    assert nsky == nsky_expected


def test_example1c(data, nsky_expected):
    nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
    query_table = data.runs[['id', nsky]]  # design a table by using the square brackets
    concrete_table = query_table()
    expected_nruns = count(data.runs)()
    assert len(concrete_table) == expected_nruns
    assert set(concrete_table.sum) == nsky_expected
    assert len(set(concrete_table.id)) == expected_nruns


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
    assert not any(np.any(t['wvl'].mask) for t in table(limit=10))  # just check the first few


def test_example3(data):
    l2s = data.l2stacks
    l2s = l2s[(l2s.ob.mjd >= 57780) & any(l2s.fibre_target.surveys == '/WL.*/', wrt=l2s.fibre_target)]
    l2s = l2s[l2s['ha_6562.80_flux'] > 0]
    table = l2s[['ha_6562.80_flux', 'z']]()  # type: Table
    assert len(table) == 914
    assert table['z'].min(), table['z'].max() == (-0.004380459951814368, 0.9470194571839268)
    assert table['ha_6562.80_flux'].min(), table['ha_6562.80_flux'].max() == (0.013978097400486398, 2.736550702489109e+18)


def test_example4(data):
    from astropy.table import Table
    import weaveio
    fname = Path(weaveio.__file__).parents[0] / 'tests/my_table.ascii'
    table = Table.read(fname, format='ascii')
    rows, targets = join(table, 'cname', data.weave_targets)
    mjds = targets.exposures.mjd  # get the mjd of the plate exposures for each target
    q = targets['cname', rows['modelMag_i'], {'mjds': mjds, 'nobservations': count(mjds, wrt=targets)}]
    t = q()
    assert len(t) == len(table)
    assert np.all(np.sort(t['cname']) == np.sort(table['cname']))
    assert np.all(t['nobservations'] >= 3)


def test_adjunct(data):
    t = data.l1single_spectra[data.l1single_spectra.camera == 'red']
    t = t[['camera', t.adjunct.camera]](limit=5)
    assert np.all(t['camera0'] == 'red')
    assert np.all(t['camera1'] == 'blue')

