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


def collision_logic(merge, neo_object, properties, on_collision):
    before, on_create, on_match, always, after = "", "", "", "", ""
    if properties:
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
        if on_create:
            on_create = f"\tON CREATE SET {on_create}"
        if on_match:
            on_match = f"\tON MATCH SET {on_match}"
        if after:
            after = f"\t{after}"
        if always:
            always = f"\tSET {always}"
        return '\n'.join([i for i in [before, merge, on_create, on_match, always, after, 'WITH *'] if i])

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
        return collision_logic(merge, self.to_node, self.other_properties, self.on_collision)


class MergeNodeAndRelationships(MergeNode):
    def __init__(self, labels, ident_properties, other_properties, parent_nodes,
                 parent_rel_ident_properties, parent_rel_other_properties, on_collision, graph,
                 parameters=None):
        self.parent_rel_ident_properties = [to_cypher_dict_or_variable(x) for x in parent_rel_ident_properties]
        self.parent_rel_other_properties = [to_cypher_dict_or_variable(x) for x in parent_rel_other_properties]
        super().__init__(labels, ident_properties, other_properties, on_collision, graph, parameters)
        self.inputs += self.parent_rel_ident_properties + self.parent_rel_other_properties + parent_nodes

    def make_cypher(self, ordering: list) -> str:
        pass
