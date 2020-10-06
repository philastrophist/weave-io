import logging
import pandas as pd
from astropy.table import Table

from weaveio.file import L1Single, Raw
from weaveio.product import get_product

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData, BasicQuery
from weaveio.address import Address



data = OurData('data/', port=11007)
# data.directory_to_neo4j()
# print('validating...')
# data.validate()

base_query = data

print(len(base_query.obrealisations.obids()))
print(len(base_query.runs.obids()))


print(len(base_query.weavetargets.cnames()))
print(len(base_query.obrealisations.cnames()))
print(len(base_query.runs.cnames()))