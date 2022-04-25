from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
specs = data.obs.runs.l1singlespectra
ra = specs.targra
fibretarget = specs.fibretarget
q = fibretarget.targra#[['targra', 'surveytarget.targra', 'cname', specs.runs.runid, ra]]

q._G.export('parser')
lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)