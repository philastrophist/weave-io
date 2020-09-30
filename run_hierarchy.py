import logging
import pandas as pd
from weaveio.product import get_product

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData, BasicQuery
from weaveio.address import Address

# data[camera=red].runs[runid].OBspec.runs[vph=1]()
# q = BasicQuery().index_by_address(Address(camera='red', mode='MOS')).index_by_hierarchy_name('Run', 'below').\
#     index_by_id('1002793').index_by_hierarchy_name('OBSpec', 'above').\
#     index_by_hierarchy_name('Run', 'below').index_by_address(Address(vph=1)).index_by_hierarchy_name('vph', 'above')
# cypher = q.make(branch=False)
# print(cypher)
#
# print('=====')

data = OurData('data/', port=11007)
# data.directory_to_neo4j()

query = data[Address(camera='red', mode='MOS')].runs['1002793'].obspec.l1singles[Address(vph=1)]
print(query.query.make(True))
files = query()



index = pd.DataFrame({'cname': ['WVE_01095017+3704301']})
product = get_product(files, 'fibtable', index)


print(product)