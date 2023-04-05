from astropy.table import Table

from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(rootdir='/beegfs/weavelofar/weaveio')
lines = ['HeII_3203.15', '[NeV]_3345.81', '[NeV]_3425.81', '[OII]_3726.03', '[OII]_3728.73']
print(data.l2stacks.gandalfs[[f"{l}_flux".lower() for l in lines]](limit=10))
# print(data.l2stacks.gandalfs[[f"{l}_flux".lower() for l in lines]]._debug_output()[1])