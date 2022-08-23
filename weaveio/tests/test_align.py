import pytest
from weaveio import *
from weaveio.readquery.results import Table


def test_aligned_attribute(data):
    red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
    blue = red.adjunct
    aligned = align(red=red, blue=blue).camera
    table = aligned()
    assert isinstance(table, Table)
    assert table.colnames == ['red', 'blue']
    assert all((table['red'].data == 'red').filled(True))
    assert all((table['blue'].data == 'blue').filled(True))

def test_aligned_column(data):
    red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
    blue = red.adjunct
    aligned = align(red=red, blue=blue)[['camera']]
    table = aligned()
    assert isinstance(table, Table)
    assert table.colnames == ['red_camera', 'blue_camera']
    assert all((table['red_camera'].data == 'red').filled(True))
    assert all((table['blue_camera'].data == 'blue').filled(True))

def test_aligning_to_table_by_from_others(data):
    red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
    blue = red.adjunct
    aligned = align(red=red, blue=blue)
    avg = mean(aligned.l1stack_spectra.snr, wrt=aligned)
    snr = aligned.snr
    q = aligned[{'divide': avg / snr}, snr, avg]
    table = q()
    assert table.colnames == ['red_divide', 'red_snr', 'red_mean', 'blue_divide', 'blue_snr', 'blue_mean']
    np.testing.assert_allclose(table['red_divide'], table['red_mean'] / table['red_snr'])
    np.testing.assert_allclose(table['blue_divide'], table['blue_mean'] / table['blue_snr'])
