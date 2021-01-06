from weaveio.opr3 import OurData
import numpy as np

data = OurData('data', port=7687, write=True)
#
# deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
# assert np.all(deletion == 0)
#
# data.drop_all_constraints()
# data.apply_constraints()
#
# report = data.read_directory()
# print(report)

report = data.read_files('data/stacked_1002814.fit', batch_size=1000)
# report = data.read_files('data/stacked_1002082_1002081.aps.fits')

import matplotlib.pyplot as plt
plt.plot(report.elapsed_time.values)
plt.savefig('times.png')
