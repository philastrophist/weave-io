import logging

from weaveio.address import Address

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData

data = OurData('data/', port=11007)
# data.directory_to_neo4j()
# print('validating...')
# data.validate()
base = data[Address(camera='red', vph=1, resolution='LowRes', mode='MOS')].runs.raws
query = base.fibtable
print(query())

