from weaveio.syntax import *


q = Query()
spec = q._start_at_object_index('OB', 1)._traverse_by_object_index('Run', 123)._traverse_to_specific_object('L1SingleSpectrum')
id = spec._select_attribute('id')
q = spec._filter_by_mask(id._perform_arithmetic('{0} * 2', '*')._perform_arithmetic('{0} > 0', '>'))._select_attribute('id')._perform_arithmetic('{0} * 2', '*')

q._G.export('parser')
lines, params = q._compile()
for line in lines:
    print(line)
print(params)