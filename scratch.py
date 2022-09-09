from weaveio.readquery.writing.statements import MergeNode, MergeSimpleNodeAndRelationships, AdvancedMergeNodeAndRelationships
from weaveio import *

data = Data()
id = data.query._G.add_parameter(1, 'id')

# node = MergeNode(['OB'], {'id': id}, {'a': id, 'b': id}, on_collision='rewrite', graph=data.query._G)
# print(node.make_cypher([]))


# node = MergeSimpleNodeAndRelationships(['OB'], {'id':id}, {'other': id},
#                                 ['a', 'b'], [{'i':0}, {'i':0}], [{'other':0}, {'other':0}],
#                                 ['is_required', 'is_required'], 'raise', data.query._G)
# print(node.make_cypher([]))

# node = AdvancedMergeNodeAndRelationships(['OB'], {'id': data.query._G.add_variable_getitem(ids, 'id')}, other,
#                                          {'parents0': ({'ids': 'ids0'}, {'a': 'a0'}), 'parents1': ({'ids': 'ids1'}, {'a': 'a1'})},
#                                          'is_required', True, 'raise', data.query._G)
# print(node.make_cypher([]))


id = data.query._G.add_parameter(100)
node = AdvancedMergeNodeAndRelationships(['Child'], {'id': id}, {},
                                         {'parents0': ({}, {})},
                                         'is_required', True, 'leavealone', data.query._G)
print(node.ident_properties)
print(node.make_cypher([]))
