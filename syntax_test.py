import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
# obs = data.obs[data.obs.obstartmjd >= 57787]  # pick an OB that started after this date
# fibretargets = obs.fibretargets[obs.fibretargets.targra == 0]
# fibretargets = obs.fibretargets[any(obs.fibretargets.l1singlespectra.snr == 0)]
# fibs = data.fibretargets
# q = fibs['targra', 'targdec', 'targras', 'surveys.name']

# data.obs.mean_fluxes.g
q = data.obs.l1singlespectra.noss.flux

#
data.query._G.export('parser')
lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)
data.query._G.export('parser')
