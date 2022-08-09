import logging

import inspect

# logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()
parent = data.obs
groupby = 'ob'
targuse = 'S'
camera = 'red'

split_obs = split(parent.ob)
query = count(split_obs.l1single_spectra.snr, wrt=split_obs)

# groups = count(groups)

for name, group in query:
    print(name, group, group._debug_output()[0])
    break
# stacks = groups.l1stack_spectra[(groups.l1stack_spectra.targuse == targuse) & (groups.l1stack_spectra.camera == camera)]
# singles = stacks.l1single_spectra
# query = stacks[['ob.id', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles[['flux', 'ivar']]}]]
#
