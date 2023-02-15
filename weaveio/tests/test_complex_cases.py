import pytest
from weaveio import *


def test_rerun(data):
    l2s = data.l2stacks
    l2s = l2s[any(l2s.fibre_target.surveys == '/WL.*/', wrt=l2s)]
    l2s = l2s[l2s['ha_6562.80_flux'] > 0]
    ratio = l2s['[oiii]_5006.77_flux'] / l2s['ha_6562.80_flux']
    max(ratio)
    ratio = l2s['[oiii]_5006.77_flux'] / l2s['ha_6562.80_flux']
    one_l2 = l2s[max(ratio) == ratio]
    t = one_l2[['[oiii]_5006.77_flux', 'ha_6562.80_flux', 'cname']]
    cname1 = t()['cname'][0]
    l2s = l2s[l2s['[oiii]_5006.77_flux'] > 0]
    ratio = l2s['[oiii]_5006.77_flux'] / l2s['ha_6562.80_flux']
    one_l2 = l2s[max(ratio) == ratio]
    t = one_l2[['[oiii]_5006.77_flux', 'ha_6562.80_flux', 'cname']]
    cname2 = t()['cname'][0]
    assert cname2 == cname1


def test_rerun_with_limit(data):
    obs = data.obs.id
    assert np.all(obs(limit=10) == obs(limit=10))


def test_merge_tables(data):
    def noise_spectra_query(parent, camera, targuse='S', split_into_subqueries=True):
        if split_into_subqueries:
            parent = split(parent)
        stacks = parent.l1stack_spectra[(parent.l1stack_spectra.targuse == targuse) & (parent.l1stack_spectra.camera == camera)]
        singles = stacks.l1single_spectra
        singles_table = singles[['flux', 'ivar']]
        query = stacks[[stacks.ob.id, {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles_table}]]
        return query

    for index, query in noise_spectra_query(data.obs, 'red'):
        t = query(limit=1)
        assert t['id'] == index
        assert len(t['stack_flux'].shape) == 1
        assert len(t['single_flux'].shape) == 2
        break


def test_aggregations_of_aggregations_of_aggregations(data):
    obs = data.obs
    runs = obs.runs
    single = runs.l1single_spectra
    spec = single.l1stack_spectra
    single_mean = mean(spec.snr, wrt=single)
    run_mean = mean(single_mean, wrt=runs)
    ob_mean = mean(run_mean, wrt=obs)
    assert len(ob_mean()) == count(obs)()


def test_noss_access(data):
    parent = data.obs
    stacks = parent.l1stack_spectra[(parent.l1stack_spectra.targuse == 'S') & (parent.l1stack_spectra.camera == 'red')]
    singles = stacks.l1single_spectra  # get single spectra for each stack spectrum
    ss_query = stacks[['ob', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles[['flux', 'ivar', 'wvl']]}]]
    noss_query = stacks.noss[['ob', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles.noss[['flux', 'ivar', 'wvl']]}]]
    ss = ss_query(limit=10)
    noss = noss_query(limit=10)
    assert all(ss['ob'] == noss['ob'])
    assert all(np.max(ss['stack_flux'], axis=1) <= np.max(noss['stack_flux'], axis=1))