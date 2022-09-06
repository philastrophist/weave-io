from weaveio.readquery.writing.statements import MergeNode, MergeSimpleNodeAndRelationships, AdvancedMergeNodeAndRelationships
from weaveio import *

data = Data()
ids = data.query._G.add_parameter({'id': 1}, 'p')
other = data.query._G.add_parameter({'a': 1, 'b': 2}, 'p')

# node = MergeNode(['OB'], {'id': data.query._G.add_variable_getitem(ids, 'id')}, other, on_collision='rewrite', graph=data.query._G)
# print(node.make_cypher([]))


# node = MergeSimpleNodeAndRelationships(['OB'], {'id': data.query._G.add_variable_getitem(ids, 'id')}, other,
#                                 ['a', 'b'], [{'i':0}, {'i':0}], [{'other':0}, {'other':0}],
#                                 ['is_required', 'is_required'], 'raise', data.query._G)
# print(node.make_cypher([]))

# node = AdvancedMergeNodeAndRelationships(['OB'], {'id': data.query._G.add_variable_getitem(ids, 'id')}, other,
#                                          {'parents0': ({'ids': 'ids0'}, {'a': 'a0'}), 'parents1': ({'ids': 'ids1'}, {'a': 'a1'})},
#                                          'is_required', True, 'raise', data.query._G)
# print(node.make_cypher([]))


node = AdvancedMergeNodeAndRelationships(['Child'], {'id': 100}, {},
                                         {'parents0': ({}, {})},
                                         'is_required', True, 'leavealone', data.query._G)
print(node.make_cypher([]))
