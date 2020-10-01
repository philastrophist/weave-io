import logging
import pandas as pd
from astropy.table import Table

from weaveio.product import get_product

logging.basicConfig(level=logging.INFO)

from weaveio.data import OurData, BasicQuery
from weaveio.address import Address



data = OurData('data/', port=11007)
graph = data.graph.neograph

from astropy.io import fits
df = Table(fits.open('data/single_1002081.fit')[-1].data).to_pandas()


statement = """
          UNWIND $rows as row
          MERGE (c:TargetSet {name: "targetset"})
          MERGE (t:Target {cname:row.CNAME, ra:row.TARGRA, dec:row.TARGDEC})
          MERGE (f:Fibre {fibid:row.FIBREID})
          MERGE (p:Position {nspec: row.Nspec, fibra:row.FIBRERA, fibdec:row.FIBREDEC, status:row.STATUS, plate: "plate"})
          MERGE (: OtherThing)

          MERGE (t)-[:IS_REQUIRED_BY]->(p)
          MERGE (f)-[:IS_REQUIRED_BY]->(p)
          MERGE (p)-[:IS_REQUIRED_BY]->(c)
          """

# tx = graph.auto()
# params = []
# for index, row in df.iterrows():
#     params.append(row.to_dict())
#     if index % 10000 == 0:
#         tx.evaluate(statement, rows=params)
#         tx = graph.auto()
#         params = []
# tx.evaluate(statement, rows=params)
G = data.graph
G.begin()
targetset = G.add_node('TargetSet', name='targetset')
table = G.add_table_as_name(df, 'fibrow')
t = G.add_node('Target', tables=table[['CNAME', 'TARGRA', 'TARGDEC']].rename(targra='ra', targdec='dec'))
f = G.add_node('Fibre', tables=table[['FIBREID']])
p = G.add_node('Position', tables=table[['Nspec', 'FIBRERA', 'FIBREDEC', 'STATUS']], plate='plate')
o = G.add_node('OtherThing')
G.add_relationship(t, p, 'IS_REQUIRED_BY')
G.add_relationship(f, p, 'IS_REQUIRED_BY')
G.add_relationship(p, targetset, 'IS_REQUIRED_BY')
print(G.make_statement())
G.commit()

# # data.directory_to_neo4j()
#
# print("|select a run from the database")
# query = data[Address(camera='red', mode='MOS')].runs['1002793']
# run = query()
# print(f"Runid={run.runid}, exposure-mjd={run.expmjd}")
#
# print("|Get the l1single file from that run")
# l1single_file_query = query.l1single
# l1single_file = l1single_file_query()
# print(f"{l1single_file.fname}")
#
# print("|get data from that file")
# fibtable_query = l1single_file_query.fibtable[['WVE_01094616+3709096', 'WVE_01095665+3704008']]
# print(fibtable_query())
#
# print("|get the same targets from the many l1singles")
# query = data[Address(camera='red', mode='MOS')].l1singles.fibtable[['WVE_01094616+3709096', 'WVE_01095665+3704008']]
# print(query())