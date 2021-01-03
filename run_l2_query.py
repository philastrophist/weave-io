import time
from pathlib import Path

from weaveio.opr3 import OurData
from weaveio.opr3.l2files import StackL2File

data = OurData('data', port=7687, write=True)
# data.plot_relations(False)

with data.write('ignore') as query:
    StackL2File.read(data.rootdir, Path('data/stacked_1002082_1002081.aps.fits').relative_to(data.rootdir))
    cypher, params = query.render_query()
    # ls = cypher.split('\n')[:-3]
    # ls += ['return l1spectrumrow1']
    # cypher = '\n'.join(ls)
    print(data.graph.output_for_debug(**params))
    # print(cypher)
# start = time.time()
# results = data.graph.execute(cypher, **params)
# print(time.time() - start)
