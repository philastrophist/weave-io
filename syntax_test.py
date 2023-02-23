from astropy.table import Table

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data()
runid = 1003453
nsky = sum(data.runs[runid].targuses == 'S')
print("number of sky targets = {}".format(nsky()))
run = data.runs[runid]
condition = run.targuses == 'S'
sky_spectra = run.l1single_spectra[condition]
table = sky_spectra[['wvl', 'flux']]
print(count(table)())