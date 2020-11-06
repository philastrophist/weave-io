from weaveio.data import OurData

data = OurData('data', port=11007)
# thing = data.exposures.runs.exposures.runs[['1002814']]['runids', 'expmjd', 'cnames']
thing = data.runs[['expmjds', 'runid', 'cnames']]
print(thing)
print(thing.query.to_neo4j()[0])
result = thing()
print(type(result))
print(result)