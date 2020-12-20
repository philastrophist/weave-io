import time

from weaveio.opr3 import OurData
import numpy as np
from weaveio.opr3.file import RawFile, L1SingleFile
import pandas as pd

data = OurData('data', port=7687, write=True)

# deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
# assert np.all(deletion == 0)

# data.drop_all_constraints()
# data.apply_constraints()

print('start write')
with data.write() as query:
    # L1SingleFile.read(data.rootdir, 'single_1002813.fit')
    RawFile.read(data.rootdir, 'r1002813.fit')
    cypher, params = query.render_query()

r = data.graph.execute(cypher, **params)

print(r.stats())