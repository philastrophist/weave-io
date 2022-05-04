import logging
from itertools import zip_longest

from pprint import pprint

import networkx as nx
from weaveio.opr3.l2files import L2SupertargetFile

from weaveio.data import hierarchies_from_files
from weaveio.opr3.hierarchy import *
from weaveio.opr3.l1 import *
from weaveio.opr3.l2 import *
from weaveio.readquery.base import AmbiguousPathError

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import *

# data = Data(dbname='playground', user='neo4j', password='NF49rfTY', write=True)
# data.read_directory('raw_file')
data = Data(dbname='playground')

# pprint(data.get_possible_intermediate_paths(L1SingleSpectrum, NoSS))

# q = data.l1single_spectra.l2singles.galaxy_redshifts
#
# lines, params, names = q._compile()
# for line in lines:
#     print(line)
# print(params, names)
#
#

#
pprint(data.path_to_hierarchy('L1SupertargetSpectrum', 'WeaveTarget', False,
                                                  descriptor='is_required_by',
                                                  return_objs=True))
