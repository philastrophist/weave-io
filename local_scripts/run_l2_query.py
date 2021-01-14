import time
from pathlib import Path

from tqdm import tqdm

from weaveio.opr3 import OurData
from weaveio.opr3.l2files import StackL2File

data = OurData('../data', port=7687, write=True)
# data.plot_relations(False)

times = []
for i in tqdm(range(0, 1000, 100)):
    with data.write('ignore') as query:
        StackL2File.read(data.rootdir,
                         Path('../data/stacked_1002814_1002813.aps.fits').relative_to(data.rootdir),
                         slice(i, i+100))
        cypher, params = query.render_query()
    start = time.time()
    results = data.graph.execute(cypher, **params)
    times.append(time.time() - start)

print(times)
import matplotlib.pyplot as plt
plt.plot(times)
plt.savefig('times.png')
