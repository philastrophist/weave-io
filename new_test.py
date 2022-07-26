from weaveio import *
from weaveio.opr3 import Data
from astropy.table import Table

data = Data()


table = Table.read('my_table.ascii', format='ascii')
targets = data.runs.weave_targets[table['cname']]


# mjds = spectra.exposure.mjd
# print(targets[['cname', count(mjds, wrt=targets), mjds]]())
# for w, f in zip(spectra[['wvl', 'flux']], table['modelMag_i']):

q = targets[['run.id', 'cname']]

cypher, params = q._precompile()._to_cypher()
print('\n'.join(cypher))
print(q())
