import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data()
# q = data.obs.l1single_spectra.l1stack_spectra['snr']
print(data.path_to_hierarchy('l1single_spectra', 'l1stack_spectra', True))

# data.query._G.export('parser')
# lines, params, names = q._compile()
# for line in lines:
#     print(line)
# print(params)
# print(names)
# data.query._G.export('parser')
