import time

from tqdm import tqdm

from weaveio.opr3 import OurData, L1SingleFile, RawFile, L1StackFile
import numpy as np

data = OurData('data', port=7687, write=True)
data.plot_relations(False)

# deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
# assert np.all(deletion == 0)
#
# data.drop_all_constraints()
# data.apply_constraints()


times = []
basefiles = []
for typ in [L1StackFile]:
    for f in data.rootdir.glob(typ.match_pattern):
        basefiles.append([typ, f])
files = []
batch_size = 100
for reader, fname in basefiles:
    if reader == L1StackFile:
        length = len(reader.hash_spectra(fname, fname.relative_to(data.rootdir)))
        for start in range(0, length + 1, batch_size):
            slc = slice(start, start + batch_size)
            files.append((reader, fname, slc))
    else:
        files.append((reader, fname, slice(None)))

bar = tqdm(files, smoothing=1)
for reader, fname, slc in bar:
    if fname.name not in ['stacked_1002081.fit']:#, 'single_1002083.fit', 'single_1002085.fit']:
        continue
    bar.set_description(str(fname))
    with data.write('ignore') as query:
        reader.read(data.rootdir, fname.relative_to(data.rootdir), slc)
        cypher, params = query.render_query()
    start = time.time()
    # ls = cypher.split('\n')
    # cypher = '\n'.join(ls[:-2] + ['RETURN unwound_fibretargets0, count(l1singlespectrum0),count(l1singlespectrum1),count(l1singlespectrum2)'])
    # param_statement = data.graph.output_for_debug(**params)
    results = data.graph.execute(cypher, **params)
    times.append(time.time() - start)
print(times)

import matplotlib.pyplot as plt
plt.plot(times)
plt.savefig('times.png')


