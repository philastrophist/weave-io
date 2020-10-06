import logging

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData

data = OurData('data/', port=11007)
# data.directory_to_neo4j()
# print('validating...')
# data.validate()

query = data.runs['1002767'].l1single.fibtable[{'cname': ["WVE_21454452+1353047","WVE_21483305+1440028","WVE_21443608+1345451"]}]
print(query())