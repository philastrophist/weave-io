from collections import namedtuple
from functools import reduce

from ..statements import Statement


def quote_string(x):
    if isinstance(x, str):
        return f'"{x}"'
    return x

def to_cypher_dict_or_variable(x):
    if isinstance(x, dict):
        return "{" + ', '.join([f"{k}:{quote_string(v)}" for k, v in x.items()]) + "}"
    else:
        return x


class CollisionManager:
    def __init__(self, on_create=None, on_match=None, always=None, after=None):
        self.on_create = on_create
        self.on_match = on_match
        self.always = always
        self.after = after

    @classmethod
    def from_neo_object(cls, neo_object, properties, on_collision):
        on_create, on_match, always, after = "", "", "", ""
        if on_collision == 'overwrite':
            always = f"{neo_object} = {properties}"
        elif on_collision == 'preferold':
            on_match = f"{neo_object} = apoc.map.merge({properties}, properties({neo_object}))"
            on_create = f"{neo_object} = {properties}"
        elif on_collision == 'prefernew':
            always = f"{neo_object} += {properties}"
        elif on_collision == 'leavealone':
            on_create = f"{neo_object} = {properties}"
        elif on_collision == 'raise':
            on_create = f"{neo_object} = {properties}, {neo_object}._collisions = []"
            on_match = f"{neo_object}._collisions = [x in apoc.coll.intersection(keys(properties({neo_object})), " \
                       f"keys({properties})) where {neo_object}[x] <> {properties}[x]], {neo_object} += {properties}"
            after = f"WITH * CALL apoc.util.validate(not isempty({neo_object}._collisions)" \
                    f", 'properties of {neo_object} collided with new: %s', [apoc.map.fromPairs([x in {neo_object}._collisions | [x, {properties}[x]]])])"
        else:
            raise ValueError(f"Unknown on_collision `{on_collision}`. Accepted values are 'overwrite', 'prefernew', 'prefer', 'leavealone', 'raise'")
        return CollisionManager(on_create, on_match, always, after)

    def extend(self, x: 'CollisionManager') -> 'CollisionManager':
        return CollisionManager(', \n'.join([self.on_create, x.on_create]),
                                ', \n'.join([self.on_match, x.on_match]),
                                ', \n'.join([self.always, x.always]),
                                '\n'.join([self.after, x.after])
                                )

    def __iadd__(self, other):
        return reduce(lambda x: x.extend, [self]+list(other))

    def __str__(self):
        on_create = f"\tON CREATE SET {self.on_create}" if self.on_create else None
        on_match = f"\tON MATCH SET {self.on_match}" if self.on_match else None
        after = f"\t{self.after}" if self.after else None
        always = f"\tSET {self.always}" if self.always else None
        return '\n'.join([i for i in [on_create, on_match, always, after] if i])


class Merge(Statement):
    pass


class MergeNode(Merge):
    """
    Merge a node only on class and properties
    on_collision can be either `overwrite` or `ignore`
    """
    ids = ['labels', 'on_collision']

    def __init__(self, labels, ident_properties, other_properties, on_collision, graph, parameters=None):
        ident_properties = to_cypher_dict_or_variable(ident_properties)
        other_properties = to_cypher_dict_or_variable(other_properties)
        super(MergeNode, self).__init__([ident_properties, other_properties], graph, parameters)
        self.ident_properties = ident_properties
        self.other_properties = other_properties
        self.to_node = self.make_variable(labels[0])
        self.labels = ':'.join(labels)
        self.on_collision = on_collision

    def make_cypher(self, ordering: list) -> str:
        merge = f"MERGE ({self.to_node}:{self.labels} {self.ident_properties})"
        if self.other_properties:
            collision = CollisionManager.from_neo_object(self.to_node, self.other_properties, self.on_collision)
            return f"{merge}\n{collision}\nWITH *"
        return f"{merge}\nWITH *"


class MergeSimpleNodeAndRelationships(MergeNode):
    ids = ['labels', 'on_collision', 'parent_rel_types']

    def __init__(self, labels, ident_properties, other_properties, parent_nodes,
                 parent_rel_ident_properties, parent_rel_other_properties,
                 parent_rel_types,
                 on_collision, graph,
                 parameters=None):
        assert len(parent_rel_other_properties) == len(parent_rel_ident_properties) == len(parent_nodes)
        self.parent_rel_ident_properties = [to_cypher_dict_or_variable(x) for x in parent_rel_ident_properties]
        self.parent_rel_other_properties = [to_cypher_dict_or_variable(x) for x in parent_rel_other_properties]
        self.parent_rel_types = parent_rel_types
        super().__init__(labels, ident_properties, other_properties, on_collision, graph, parameters)
        self.inputs += self.parent_rel_ident_properties + self.parent_rel_other_properties + parent_nodes
        self.r1, self.r2 = None, None
        if len(parent_nodes):
            self.r1 = self.make_variable('r1')
        if len(parent_nodes) > 1:
            self.r2 = self.make_variable('r2')
        if len(parent_nodes) > 2:
            raise ValueError(f"Cannot merge more than two node and relationship nodes using MergeSimpleNodeAndRelationships")
        self.parent_nodes = parent_nodes
        self.rels = [self.r1, self.r2]

    def make_cypher(self, ordering: list) -> str:
        merge = f"({self.to_node}:{self.labels} {self.ident_properties})"
        if self.r1:
            merge = f"({self.parent_nodes[0]})-[{self.r1}:{self.parent_rel_types[0]} {self.parent_rel_ident_properties[0]}]->{merge}"
        if self.r2:
            merge = f"{merge}<-[{self.r2}:{self.parent_rel_types[1]} {self.parent_rel_ident_properties[1]}]-({self.parent_nodes[1]})"
        parent_collisions = [CollisionManager.from_neo_object(r, rp, self.on_collision) for r, rp in zip(self.rels, self.parent_rel_other_properties)]
        if self.other_properties:
            collision = CollisionManager.from_neo_object(self.to_node, self.other_properties, self.on_collision)
            collision += parent_collisions
            return f"{merge}\n{collision}\nWITH *"
        elif parent_collisions:
            parent_collisions[0] += parent_collisions[1:]
            return f"{merge}\n{parent_collisions[0]}\nWITH *"
        else:
            return f"{merge}\nWITH *"


class AdvancedMergeNodeAndRelationships(MergeSimpleNodeAndRelationships):
    pass