import logging

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data()

for arrows, singular, path in data.paths_to_hierarchy('run', 'noss', False):
    print(path)

# There are quicker traversals in this graph between run and noss though