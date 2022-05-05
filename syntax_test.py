import logging

from pprint import pprint

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import *

data = Data(dbname='playground', user='neo4j', password='NF49rfTY', write=True)
data.read_directory('raw_file')
