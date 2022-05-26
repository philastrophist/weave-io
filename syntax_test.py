from matplotlib import pyplot as plt
from tqdm import tqdm

from weaveio import *
import logging

from weaveio.data import make_arrows
from weaveio.path_finding import find_singular_simple_hierarchy_path

logging.basicConfig(level=logging.INFO)

data = Data(dbname='lowleveltest2')


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

plt.figure()
l2s = data.l2stacks
l2s = l2s[(l2s.ob.mjd >= 57780) & any(l2s.fibre_target.surveys == '/WL.*/', wrt=l2s.fibre_target)]
l2s = l2s[l2s['ha_6562.80_flux'] > 0]
table = l2s[['ha_6562.80_flux', 'z']]()
plt.scatter(table['z'], table['ha_6562.80_flux'], s=1)
plt.yscale('log')
plt.show()
# plt.savefig('sky_spectra.png')

# obs = data.obs[data.obs.mjd >= 57811]  # pick an OB that started after this date
# fibre_targets = obs.fibre_targets[any(obs.fibre_targets.surveys == '/WL.*/', wrt=obs.fibre_targets)]  # / indicate regex is starting and ending
# l2rows = fibre_targets.l2stacks
# q = l2rows['ha_6562.80_flux']

# l2 = data.redrocks.l2single
# q = l2.surveys.name
# q = l2['ha_6562.80_flux']
# l2rows = l2[any(l2.surveys == '/WL.*/', wrt=l2)] #&(l2.ob.mjd >= 57811)]
# q = l2rows['ha_6562.80_flux']
# q = l2.targra
# print('\n'.join(q._precompile()._to_cypher()[0]))
# print(q(limit=100))

# print(data.path_to_hierarchy('Redrock', 'Survey', False))
# print(data.redrocks.surveys.name(limit=10))

# import networkx as nx
#
# G = nx.DiGraph()
# G.add_edge('a', 'b', weight=0)
# G.add_edge('b', 'c', weight=0)
# G.add_edge('a', 'c', weight=0)
# G.add_edge('a', 'd', weight=0)
# G.add_edge('d', 'e', weight=0)
# G.add_edge('e', 'c', weight=0)
# for path in nx.shortest_simple_paths(G, 'a', 'c', 'weight'):
#     print(path)

# g = data.relation_graphs[2]
# a, b = data.class_hierarchies['CombinedIngestedSpectrum'], data.class_hierarchies['CombinedModelSpectrum']
#
# for (start, end) in [(a, b), (b, a)]:
#     path = find_singular_simple_hierarchy_path(g, start, end)
#     singular = all(g.edges[(a, b)]['singular'] for a, b in zip(path[:-1], path[1:]))
#     forwards = ['relation' not in g.edges[edge] for edge in zip(path[:-1], path[1:])]
#     arrows = make_arrows(path, [not f for f in forwards])
#     print(start.__name__, arrows, end.__name__)

# print(data.find_names('ha_6562_flux'))
#
# import matplotlib.pyplot as plt
# # uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# # %matplotlib
