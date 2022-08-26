from weaveio.readquery.writing.statements import MergeNode, MergeSimpleNodeAndRelationships
from weaveio import *

data = Data()
ids = data.query._G.add_parameter({'id': 1}, 'p')
other = data.query._G.add_parameter({'a': 1, 'b': 2}, 'p')

# node = MergeNode(['OB'], {'id': data.query._G.add_variable_getitem(ids, 'id')}, other, on_collision='rewrite', graph=data.query._G)
# print(node.make_cypher([]))


node = MergeSimpleNodeAndRelationships(['OB'], {'id': data.query._G.add_variable_getitem(ids, 'id')}, other,
                                ['a', 'b'], [{'i':0}, {'i':0}], [{'other':0}, {'other':0}],
                                ['is_required', 'is_required'], 'raise', data.query._G)
print(node.make_cypher([]))