from weaveio.opr3 import OurData
import numpy as np
from weaveio.opr3.file import RawFile
import pandas as pd

data = OurData('data', port=7687, write=True)
# print(data.graph.execute('unwind $data as d return d.a', data=pd.DataFrame({'a': [np.nan, 1, 2,np.nan]})).to_table())

deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
assert np.all(deletion == 0)
data.drop_all_constraints()

print('start write')
with data.write() as query:
    fibinfo = RawFile.read_fibinfo_dataframe('data/r1002813.fit')
    srvyinfo = RawFile.read_surveyinfo(fibinfo)
    RawFile.make_fibretargets(srvyinfo, fibinfo)
cypher, params = query.render_query()
r = data.graph.execute(cypher, **params)
print(r.stats())
# data.directory_to_neo4j('Raw')

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