from weaveio.readquery import *
from pprint import pprint

from weaveio.opr3 import *
import logging
logging.basicConfig(level=logging.INFO)

data = Data(dbname='playground', user='neo4j', password='NF49rfTY', write=True)
# data.drop_all_constraints()
data.read_directory('l1single_file', dryrun=False)
