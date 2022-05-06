from weaveio.readquery import *
from pprint import pprint

from weaveio.opr3 import *
import logging
logging.basicConfig(level=logging.INFO)

data = Data(dbname='test', user='neo4j', password='NF49rfTY', write=False)
# data.drop_all_constraints()
# files = data.find_files('raw_file')
# data.read_files(*files[:5])
count = count(data.raw_files.raw_spectra, wrt=data.raw_files)
q = data.raw_files['fname', count, data.raw_files.ob.id, data.raw_files.raw_spectrum.counts1]
print('\n'.join(q._to_cypher()[0]))
print(q())
#
# print(data.path_to_hierarchy('Run', 'OB', False))
# print(data.path_to_hierarchy('Run', 'RawFile', False))