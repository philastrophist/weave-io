from weaveio.readquery.writing.statements import MergeNode
from weaveio import *

data = Data()

node = MergeNode(['OB'], {'id': 1}, {'a': 1, 'b':'d'}, on_collision='raise', graph=data.query._G)
print(node.make_cypher([]))