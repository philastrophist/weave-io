from copy import copy, deepcopy

from weaveio.basequery.query import Node, Path, Generator, Branch, Predicate, FullQuery


# data.runs[runid].exposure.runs.vphs

from weaveio.basequery.query import NodeProperty

from weaveio.data import OurData

data = OurData('data', port=11007)
thing = data.runs.exposures.runs['1002813', '1002813']
exposures = thing.exposures
print(thing())
print(exposures())
