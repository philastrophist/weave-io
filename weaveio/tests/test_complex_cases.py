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