from pathlib import Path

from weaveio.opr3 import OurData
from weaveio.opr3.l2files import StackL2File

data = OurData('data', port=7687, write=True)
# data.plot_relations(False)

with data.write('ignore') as query:
    StackL2File.read(data.rootdir, Path('data/stacked_1002082_1002081.aps.fits').relative_to(data.rootdir))
    cypher, params = query.render_query()
