from weaveio import *
from weaveio.readquery.results import Row
from astropy.table import vstack
import pytest

def test_as_table(data):
    obs = data.obs
    all_query = count(obs.l1single_spectra, wrt=obs)
    all_r = obs[[all_query, 'id']]
    split_obs = split(obs)
    query = count(split_obs.l1single_spectra, wrt=split_obs)
    r = split_obs[[query, 'id']]

    rows = []
    for index, subquery in r:
        row = subquery()
        assert isinstance(row, Row)
        assert np.all(row['id'] == index[0])
        rows.append(row)

    row_table = vstack(rows)
    row_table.sort('id')
    all_table = all_r(limit=None)
    all_table.sort('id')
    row_table.rename_column('count1', 'count0')
    assert all(row_table == all_table)


def test_as_attribute(data):
    obs = data.obs
    split_obs = split(obs)
    query = count(split_obs.l1single_spectra, wrt=split_obs)
    all_query = count(obs.l1single_spectra, wrt=obs)

    rows = []
    for index, q in query:
        rows.append(q())
    assert sorted(all_query(limit=None)) == sorted(rows)


def test_continue_query(data):
    obs = data.obs
    split_obs = split(obs)
    query = split_obs.l1single_spectra

    for index, q in query:
        count(q)()
        break