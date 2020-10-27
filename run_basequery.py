from weaveio.basequery.query import Node, Path, Generator, Branch, Predicate, FullQuery


# data.runs[runid].exposure.runs.vphs

query = FullQuery()
gen = Generator()



query = query.append_to_root('<--', Path(gen.node('Run')))  # data.runs
query = query.identify_in_root('1002813')  # data.runs[runid]
query = query.append_to_root('<--', Path(gen.node('Exposure')))  # data.runs[runid].exposure
query = FullQuery(Path(gen.node('Run')), exist_branches=query.to_branches())

query = query.merge_into_branches(Path(query.matches.nodes[-2], '<-', gen.node('InstrumentConfiguration')))


# query = query.return_property(query.root.nodes[-2], 'runid')
query = query.return_property(gen.node('InstrumentConfiguration', 'instrumentconfiguration0'), 'vph')
# query = query.return_node(query.root.nodes[-2])
print(query.to_neo4j(gen))

