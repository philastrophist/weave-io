from weaveio.data import OurData
from weaveio.hierarchy import FibreTarget

data = OurData('data', port=7687)
print(data.make_constraints_cypher())
# data.drop_all_constraints()
# data.apply_constraints()
# data.plot_relations()
# data.directory_to_neo4j('L1SingleFile')
# data.validate()
# thing = data.exposures.runs.exposures.runs[['1002814']]['runids', 'expmjd', 'cnames']
# thing = data.runs[['expmjds', 'runid', 'cnames']]
# print(thing)
# print(thing.query.to_neo4j()[0])
# result = thing()
# print(type(result))
# print(result)