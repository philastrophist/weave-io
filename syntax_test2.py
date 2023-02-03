from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')
q = data.redrocks.surveys.name
print(q._debug_output()[0])