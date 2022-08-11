import logging

import inspect

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()
parent = data.obs
groupby = 'ob'
targuse = 'S'
camera = 'red'

split_obs = split(parent.ob)
spec = split_obs.l1single_spectra[split_obs.l1single_spectra.camera == camera]
query = count(spec, wrt=split_obs)
r = split_obs[[query, mean(spec.snr, wrt=split_obs)]]
# groups = count(groups)

print(r())
# for name, group in r:
#     print(name)
#     for row in group:
#         print(row)
    # break
# stacks = groups.l1stack_spectra[(groups.l1stack_spectra.targuse == targuse) & (groups.l1stack_spectra.camera == camera)]
# singles = stacks.l1single_spectra
# query = stacks[['ob.id', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles[['flux', 'ivar']]}]]
#
