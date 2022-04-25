from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
runs = data.obs.runs
specs = runs.l1singlespectra
red_specs = specs[specs.camera == 'red']
blue_specs = specs[specs.camera == 'blue']
q = runs[['runid', count(red_specs, wrt=runs), count(blue_specs, wrt=runs), runs.runid * 2 * mean(specs.snr, wrt=runs)]]
# q = red_specs.camera

q._G.export('parser')
lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)