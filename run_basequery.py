from weaveio.opr3 import OurData
import numpy as np
from weaveio.opr3.hierarchy import FibreTarget

data = OurData('data', port=7687)
deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
assert np.all(deletion == 0)
data.drop_all_constraints()
print(data.make_constraints_cypher())
data.directory_to_neo4j('Raw')

# data.apply_constraints()
# data.plot_relations()
# data.validate()
# thing = data.exposures.runs.exposures.runs[['1002814']]['runids', 'expmjd', 'cnames']
# thing = data.runs[['expmjds', 'runid', 'cnames']]
# print(thing)
# print(thing.query.to_neo4j()[0])
# result = thing()
# print(type(result))
# print(result)