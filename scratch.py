import logging

from weaveio.path_finding import find_paths

logging.basicConfig(level=logging.INFO)
from weaveio import *
from weaveio.opr3.l1 import *
from weaveio.opr3.hierarchy import *
data = Data()
#
parent = data.obs
# stacks = parent.l1stack_spectra[(parent.l1stack_spectra.targuse == 'S') & (parent.l1stack_spectra.camera == 'red')]
# singles = stacks.l1single_spectra  # get single spectra for each stack spectrum
# singles_table =  singles.noss[['flux', 'ivar']]
# query = stacks.noss[['ob.id', {'stack_flux': 'flux', 'stack_ivar': 'ivar'}, 'wvl', {'single_': singles_table}]]   # add in the single spectra per stack spectrum
# print(query(limit=2))
print(parent.nosses.flux._debug_output()[0])


# q = data.l1stack_spectra.wavelength_holder.wvl
# q = data.l1stack_spectra.noss.wvl
# print(q._debug_output()[0])




# r = data.hierarchy_graph.find_paths(L1StackSpectrum, NoSS, True)
# print(r)