from weaveio.opr3 import Data
from weaveio.readquery import *

data = Data()
# fib = data.obs.runs.l1singlespectra.fibretarget
# ra = fib.targras
# q = spec[['ra', 'run.runid', 'runid', spec.runs.runid, ra]]
# q = fib.runids
# q = fib[['targras', 'surveytarget.targras', 'cname', fib.runs.runid, ra]]

# q = data.obs['obid', 'runids', 'obids']
q = sum(data.obs.obids)
# q = data.obs.obids
# q = data.runs['obid', 'obids', 'runids', 'runid']
# q = data.obs['obid', 'obids', 'runids']
# q = data.obs.obids

# unravel everything until the last moment

# data.runs.obids -> [[0], [1], [2]] when its compiled
#                 -> [0, 1, 2] when its in memory (since there is no point

# data.obs.runids -> [[1,2,3,4], [...]...] when its compiled
#                   -> []



# data.obs.runids should return a list of lists
q._G.export('parser')
lines, params, names = q._compile()
for line in lines:
    print(line)
print(params)
print(names)