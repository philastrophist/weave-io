import time
from weaveio.opr3 import OurData, L1SingleFile, RawFile, L1StackFile
import numpy as np

data = OurData('data', port=7687, write=True)
data.plot_relations(False)

#
# deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
# assert np.all(deletion == 0)
#
# data.drop_all_constraints()
# data.apply_constraints()
#
#
# print('start write')
# with data.write('track&flag') as query:
#     L1SingleFile.read(data.rootdir, 'single_1002813.fit')
#     cypher, params = query.render_query()
#
# for i in range(1):
#     start = time.time()
#     r = data.graph.execute(cypher, **params)
#     print(f"{time.time() - start:.2f} seconds elapsed for query execution")
# print(r.stats())
#
#
# print('start write')
# with data.write('track&flag') as query:
#     RawFile.read(data.rootdir, 'r1002813.fit')
#     cypher, params = query.render_query()
#
# for i in range(1):
#     start = time.time()
#     r = data.graph.execute(cypher, **params)
#     print(f"{time.time() - start:.2f} seconds elapsed for query execution")
# print(r.stats())

print('start write')
with data.write('track&flag') as query:
    L1StackFile.read(data.rootdir, 'stacked_1002813.fit')
    cypher, params = query.render_query()

# for i in range(1):
#     start = time.time()
#     r = data.graph.execute(cypher, **params)
#     print(f"{time.time() - start:.2f} seconds elapsed for query execution")
# print(r.stats())