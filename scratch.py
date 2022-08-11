import logging

import inspect

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()
parent = data.obs
groupby = 'ob'
targuse = 'S'
camera = 'red'

def noise_spectra_query(parent, camera, targuse='S', split_into_subqueries=True):
    if split_into_subqueries:
        parent = split(parent)
    stacks = parent.l1stack_spectra[(parent.l1stack_spectra.targuse == targuse) & (parent.l1stack_spectra.camera == camera)]
    singles = stacks.l1single_spectra
    singles_table =  singles[['flux', 'ivar']]
    query = stacks[['ob.id', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles_table}]]
    return query


for index, query in noise_spectra_query(data.obs, 'red'):
    print(index)
    print(query(limit=10))
    break
