from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')
# data.validate()
data._validate_one_required('l2single')