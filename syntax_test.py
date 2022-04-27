import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
obs = data.obs[data.obs.obstartmjd >= 57787]  # pick an OB that started after this date
# fibretargets = obs.fibretargets[obs.fibretargets.targras == 0]
fibretargets = obs.fibretargets[any(obs.fibretargets.l1singlespectra.snr == 0)]
q = l2rows = fibretargets['fibretarget.targra']
# table = chosen['wvl', 'flux'](limit=10)


q._G.export('parser')
lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)