import logging


logging.basicConfig(level=logging.INFO)
from weaveio import *
from weaveio.opr3.l1 import *
from weaveio.opr3.l2 import *
from weaveio.opr3.hierarchy import *
data = Data()

data.hierarchy_graph.find_paths(FibreTarget, OB, True)
# data.l1single_spectra.ob
# noss = data.l1single_spectra.noss
# noss._get_path_to_object(OB, True)