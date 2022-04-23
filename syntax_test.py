from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
fib = data.obs.runs.l1singlespectra.fibretarget
ra = fib.targras
# q = spec[['ra', 'run.runid', 'runid', spec.runs.runid, ra]]
# q = fib.runids
q = fib[['targras', 'surveytarget.targras', 'cname', fib.runs.runid, ra]]

# data.obs.runids should return a list of lists

lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)