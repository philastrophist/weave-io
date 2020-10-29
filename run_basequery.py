from weaveio.basequery.query import Node, Path, Generator, Branch, Predicate, FullQuery


# data.runs[runid].exposure.runs.vphs

from weaveio.data import OurData

data = OurData('data')
thing = data.runs['1002813'].exposure
print(thing)
print(thing.query.to_neo4j())
# multiplicity, direction, path = data.node_implies_plurality_of('run', 'weavetarget')
# print(multiplicity, direction, path)