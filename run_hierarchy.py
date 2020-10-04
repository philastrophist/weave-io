import logging
import pandas as pd
from astropy.table import Table

from weaveio.file import L1Single, Raw
from weaveio.product import get_product

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData, BasicQuery
from weaveio.address import Address



data = OurData('data/', port=11007)
# data.drop_constraints()
# data.directory_to_neo4j()


# print("|select a run from the database")
# query = data[Address(camera='red', mode='MOS')].runs['1002793']
# run = query()
# print(f"Runid={run.runid}, exposure-mjd={run.expmjd}")
#
# print("|Get the l1single file from that run")
# l1single_file_query = query.l1single
# l1single_file = l1single_file_query()
# print(f"{l1single_file.fname}")
#
# print("|get data from that file")
# fibtable_query = l1single_file_query.fibtable[['WVE_01094616+3709096', 'WVE_01095665+3704008']]
# print(fibtable_query())
#
# print("|get the same targets from the many l1singles")
# query = data[Address(camera='red', mode='MOS')].l1singles.fibtable[['WVE_01094616+3709096', 'WVE_01095665+3704008']]
# print(query())