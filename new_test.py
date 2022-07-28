from weaveio import *
from weaveio.opr3 import Data
from astropy.table import Table

from weaveio.readquery.uploads import join

data = Data()

table = Table.read('weaveio/tests/my_table.ascii', format='ascii')
rows, targets = join(table, 'cname', data.weave_targets)
q = targets.cname
mjds = targets.exposures.mjd
q = targets['cname', {'mjds': mjds, 'nobservations': count(mjds, wrt=targets)}]
exposures = q()
print(exposures)

q = targets.l1single_spectra[['cname', rows['modelMag_i'], 'wvl', 'flux']]
table = q()
mean_fluxes = []
for row in table:
    filt = (row['wvl'] > 4000) & (row['wvl'] < 4500)
    mean_fluxes.append(mean(row['flux'][filt]))
table['mean_flux'] = mean_fluxes
print(table['mean_flux'])

table['mean_flux'] = apply(table, lambda row: mean(row['flux'][(row['wvl'] > 4000) & (row['wvl'] < 4500)]))
print(table['mean_flux'])

filt = apply(table['wvl'], lambda x: (x > 4000) & (x < 4500))
table['mean_flux'] = table['flux'].filtered(filt).apply(mean)
print(table['mean_flux'])



#
# targets = targets[~all(targets.redrocks.zwarn != 0, wrt=targets)]
# zs = targets.redrocks[targets.redrocks.zwarn == 0].z
# q = targets[['cname', {'redshifts': zs, 'mean_redshift': mean(zs, wrt=targets)}]]
#
#
#
cypher, params = q._precompile()._to_cypher()
print('\n'.join(cypher))
# t = q(limit=100)
# print(t)
# print(len(t), t.colnames)