import logging

from weaveio.address import Address

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData

data = OurData('data/', port=11007)
# data.directory_to_neo4j()
# print('validating...')
# data.validate()
base = data[Address(camera='red', vph=1, resolution='LowRes')]
# query = base.runs.l1singles.fibtable[["WVE_02154662-0333198", "WVE_02181009-0401078"]]
# print(query())
# print(data.runs.runids())
# print(data.l1singles.flux.query.make())
print(query())
# for run in query:
#     print(run())
