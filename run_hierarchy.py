from weaveio.data import OurData, BasicQuery, Address

# data[camera=red].runs[runid].OBspec.runs[vph=1]()
q = BasicQuery().index_by_address(Address(camera='red', mode='MOS')).index_by_hierarchy_name('Run', 'below').\
    index_by_id('1002793').index_by_hierarchy_name('OBSpec', 'above').\
    index_by_hierarchy_name('Run', 'below').index_by_address(Address(vph=1))
cypher = q.make(branch=False)
print(cypher)

print('=====')

data = OurData('.', port=11002)
query = data[Address(camera='red', mode='MOS')].runs['1002793'].obspec.runs[Address(vph=1)]
print(query.query.make())

print(query())