from matplotlib import pyplot as plt
from tqdm import tqdm

from weaveio import *
import logging

from weaveio.data import find_children_of
from weaveio.path_finding import find_singular_simple_hierarchy_path

logging.basicConfig(level=logging.INFO)

data = Data()
# lines = data.gandalfs['ha_6562.80_flux']
# q = data.gandalfs[lines > 0].l1single_spectra['runs.id', 'wvl']
# # q = data.l1single_spectra['run.id', 'flux', 'wvl']
# # q = data.runs.id
# print('\n'.join(q._precompile()._to_cypher()[0]))
# print(q(limit=2))
# for row in q:
#     print(row)
#     break

runid = 1003453
nsky = sum(data.runs[runid].targuses == 'S')
print("number of sky targets = {}".format(nsky()))

nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
print(nsky())

nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
query_table = data.runs[['id', nsky]]  # design a table by using the square brackets
concrete_table = query_table()  # make it "real" by executing the query
print(concrete_table)
print(type(concrete_table))


yesterday = 57811

runs = data.runs
is_red = runs.camera == 'red'
is_yesterday = floor(runs.exposure.mjd) == yesterday  # round to integer, which is the day

runs = runs[is_red & is_yesterday]  # filter the runs first
spectra = runs.l1single_spectra
sky_spectra = spectra[spectra.targuse == 'S']

table = sky_spectra[['wvl', 'flux']]


plt.figure()
for row in table:
    plt.plot(row.wvl, row.flux, 'k-', alpha=0.4)
    break
# plt.savefig('sky_spectra.png')

plt.figure()
l2s = data.l2stacks
l2s = l2s[(l2s.ob.mjd >= 57780) & any(l2s.fibre_target.surveys == '/WL.*/', wrt=l2s.fibre_target)]
l2s = l2s[(l2s['ha_6562.80_flux'] > 0) & (l2s['ha_6562.80_flux'] < 1e50)]
table = l2s[['ha_6562.80_flux', 'z']]()
plt.scatter(table['z'], table['ha_6562.80_flux'], s=1)
plt.yscale('log')

# plt.figure()
# l1s = l2s.l1stack_spectra.wvl


plt.show()



#
# plt.figure()
# q = data.l2stacks.l1stack_spectra.wvls
# print('\n'.join(q._precompile()._to_cypher()[0]))
# print(q(limit=1))
# ob = data.obs[3756]
# l2stacks = ob.l2stacks[any(ob.l2stacks.surveys == '/WL.*/', wrt=ob.l2stacks)]
# gband = l2stacks.l1stack_spectra[l2stacks.l1stack_spectra.camera == 'blue'].mean_flux_g  # using "mean_flux_g" instead of "mag_g" will use the weave-observed flux not the input catalogue magnitude
# l2stack = l2stacks[gband == max(gband, wrt=l2stacks)]
# print(gband(limit=1))