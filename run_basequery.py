from copy import copy

from weaveio.basequery.query import Node, Path, Generator, Branch, Predicate, FullQuery


# data.runs[runid].exposure.runs.vphs

from weaveio.data import OurData

data = OurData('data', port=11007)
thing = data.runs['1002813'].exposure.runs
print(thing)
query = thing._prepare_query().copy()
print(query.to_neo4j())
print(thing())
