from weaveio import *
from weaveio.opr3 import Data
from astropy.table import Table

from weaveio.readquery.uploads import join

data = Data()


table = Table.read('my_table.ascii', format='ascii')

rows, targets = join(table, 'cname', data.weave_targets, 'cname', join_type='left')
# targets = targets[~isnull(rows.cname)]
# q = targets[['cname', rows['type']]]
# find parent, unwind
spectra = targets.l1single_spectra#[targets.l1single_spectra.mag_i > rows['modelMag_i']]
spectra = spectra[rows['modelMag_i'] > spectra.snr]
q = spectra[['cname', 'camera', 'exposure.mjd', rows['modelMag_i'], spectra.snr]]


# mjds = spectra.exposure.mjd
# print(targets[['cname', count(mjds, wrt=targets), mjds]]())
# for w, f in zip(spectra[['wvl', 'flux']], table['modelMag_i']):


cypher, params = q._precompile()._to_cypher()  #
print('\n'.join(cypher))
print(q(limit=100))
