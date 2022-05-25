from matplotlib import pyplot as plt
from tqdm import tqdm

from weaveio import *
import logging
logging.basicConfig(level=logging.INFO)

data = Data(dbname='lowleveltest2')

# runid = 1003453
# nsky = sum(data.runs[runid].targuses == 'S')
# print("number of sky targets = {}".format(nsky()))
#
# nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
# print(nsky())
#
# nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
# query_table = data.runs[['id', nsky]]  # design a table by using the square brackets
# concrete_table = query_table()  # make it "real" by executing the query
# print(concrete_table)
# print(type(concrete_table))


# yesterday = 57811
#
# runs = data.runs
# is_red = runs.camera == 'red'
# is_yesterday = floor(runs.exposure.mjd) == yesterday  # round to integer, which is the day
#
# runs = runs[is_red & is_yesterday]  # filter the runs first
# spectra = runs.l1single_spectra
# sky_spectra = spectra[spectra.targuse == 'S']
#
# table = sky_spectra[['wvl', 'flux']]
#
# for row in table:
#     plt.plot(row.wvl, row.flux, 'k-', alpha=0.4)
# plt.savefig('sky_spectra.png')

# obs = data.obs[data.obs.mjd >= 57811]  # pick an OB that started after this date
# fibre_targets = obs.fibre_targets[any(obs.fibre_targets.surveys == '/WL.*/', wrt=obs.fibre_targets)]  # / indicate regex is starting and ending
# l2rows = fibre_targets.l2stacks
# q = l2rows['ha_6562.80_flux']
l2 = data.gandalfs
q = l2['ha_6562.80_flux']
# l2rows = l2[(l2.ob.mjd >= 57811) & any(l2.fibre_target.surveys == '/WL.*/', wrt=l2.fibre_target)]
# q = l2rows['ha_6562.80_flux']
print('\n'.join(q._precompile()._to_cypher()[0]))
print(q(limit=100))


# print(data.find_names('ha_6562_flux'))
#
# import matplotlib.pyplot as plt
# # uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# # %matplotlib
# plt.scatter(table['lineflux_ha_6562'], table['z'])