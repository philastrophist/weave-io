from weaveio.syntax import *


q = Query()
lines, params = q._start_at_object('OB')._traverse_by_object_index('Run', 123)._select_attribute('id')\
    ._compile()
for line in lines:
    print(line)
print(params)