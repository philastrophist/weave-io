import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

# data = Data(dbname='playground', user='neo4j', password='NF49rfTY', write=True)
# data.read_directory('raw_file')
data = Data(dbname='playground')
q = data.raw_spectra.raw_file.fname

cypher, params, names = q._compile()
for line in cypher:
    print(line)
