from weaveio.opr3 import OurData
import numpy as np
from weaveio.opr3.file import RawFile, L1SingleFile
import pandas as pd

data = OurData('data', port=7687, write=True)

deletion = data.graph.execute('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
assert np.all(deletion == 0)

t = data.graph.execute("""
WITH *, {fibreid: 1 , xml: 'xml2' , fibrera: 0, other: 'nan', collide: 1} as oldprops
WITH *, {fibreid: 1 , xml: 'xml2' , fibrera: 0 , fibredec: 0 , targuse: 'S', collide: 2} as newprops


// setup the "old" node in this experiment
WITH oldprops, newprops, timestamp() as time0
MERGE (old: N {targid: 0 , targra: 0 , targdec: 0}) 
	ON CREATE SET old += oldprops
	SET old._dbcreated = time0, old._dbupdated = time0

WITH newprops, timestamp() as time0  // reset time for this experiment
MERGE (new: N {targid: 0 , targra: 0 , targdec: 0})
	ON MATCH SET new = apoc.map.merge(newprops, properties(new))   // update, keeping the old colliding properties
	ON CREATE SET new._dbcreated = time0,  new._dbupdated = time0  // setup the node as standard
SET new._dbupdated = time0  // always set updated time 
// now get colliding properties
with *, [x in apoc.coll.intersection(keys(newprops), keys(properties(new))) where newprops[x] <> new[x]] as colliding_keys
WITH *, apoc.map.fromLists(colliding_keys, apoc.map.values(newprops, colliding_keys)) as collisions
CALL apoc.do.when(size(colliding_keys) > 0, 
	"WITH $innode as innode CREATE (c:_Collision)-[:COLLIDES]->(innode) SET c = $collisions SET c._dbcreated = $time", 
	"RETURN $time",
	{innode: new, collisions: collisions, time:time0}) yield value

return collisions
""").to_table()
print(t)
# data.drop_all_constraints()
# data.apply_constraints()

# print('start write')
# with data.write() as query:
#     # L1SingleFile.read(data.rootdir, 'single_1002813.fit')
#     RawFile.read(data.rootdir, 'r1002813.fit')
# cypher, params = query.render_query()
# r = data.graph.execute(cypher, **params)
# print(r.stats())