from weaveio.path_finding import find_forking_path, find_paths

from weaveio import *
from weaveio.opr3 import Data
from weaveio.opr3.l2files import L2File

logging.basicConfig(level=logging.INFO)

data = Data(dbname='opr3btestordering')


from weaveio.opr3.l1 import *
from weaveio.opr3.hierarchy import *

graph = data.hierarchy_graph.parents_and_inheritance
