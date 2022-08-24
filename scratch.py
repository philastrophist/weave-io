import logging

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()


def noise_spectra_query(parent, camera, noss=False, targuse='S', split_into_subqueries=True):
    if split_into_subqueries:
        parent = split(parent)
    stacks = parent.l1stack_spectra[(parent.l1stack_spectra.targuse == targuse) & (parent.l1stack_spectra.camera == camera)]
    singles = stacks.l1single_spectra  # get single spectra for each stack spectrum
    if noss:
        stacks = stacks.noss
        singles = singles.noss
    singles_table =  singles[['flux', 'ivar']]
    query = stacks[['ob.id', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles_table}]]   # add in the single spectra per stack spectrum
    return query


for index, ob_query in noise_spectra_query(data.obs, 'red', noss=True):
    print(f"OB = {index}")
    print(ob_query())