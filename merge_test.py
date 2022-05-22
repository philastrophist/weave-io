from textwrap import dedent

from weaveio.data import Data
from weaveio.hierarchy import Hierarchy, Multiple


class CASU(Hierarchy):
    idname = 'id'

class Run(Hierarchy):
    idname = 'id'

class RawSpectrum(Hierarchy):
    parents = [CASU, Run]
    identifier_builder = ['casu', 'run']



data = Data(dbname='lowleveltest')
data.hierarchies = {CASU, Run, RawSpectrum}
with data.write:
    # data.drop_all_constraints(indexes=True)
    # data.apply_constraints()  # needed here because we're doing it ourselves
    with data.write_cypher('ignore') as query:
        casu = CASU(id=1)
        run = Run(id=1)
        raw = RawSpectrum(casu=casu, run=run)

    cypher, params = query.render_query()
    print(dedent(cypher))
    # results = data.graph.execute(cypher, **params)