from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
runs = data.obs.runs
specs = runs.l1singlespectra
# ra = specs.targras
# fibretarget = specs.fibretarget
q = runs[['runid', count(specs, wrt=runs), runs.runid * 2 * mean(specs.snr, wrt=runs)]]


q._G.export('parser')
lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)