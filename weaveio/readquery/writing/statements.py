from collections import namedtuple
from functools import reduce

from ..statements import Statement


def to_cypher_dict_or_variable(x):
    if isinstance(x, dict):
        return "{" + ', '.join([f"{k}: {v}" for k, v in x.items()]) + "}"
    else:
        return x


def nonempty(l):
    return [i for i in l if i]


class CollisionManager:
    def __init__(self, on_create=None, on_match=None, on_always=None, on_after=None, prefix=True):
        self.on_create = on_create
        self.on_match = on_match
        self.on_always = on_always
        self.on_after = on_after
        self.prefix = prefix

    @classmethod
    def from_neo_object(cls, obj, properties, id_properties, on_collision, prefix):
        on_create, on_match, always, after = "", "", "", ""
        set_collision = f"{obj}._collisions = [x in apoc.coll.intersection(keys(properties({obj})), " \
                       f"keys({properties})) where {obj}[x] <> {properties}[x]]"
        set_modified = f"{obj}._dbupdated = CASE WHEN apoc.map.submap(properties({obj}), keys({properties})) = {properties} THEN {obj}._dbupdated ELSE time0 END"
        if on_collision == 'rewrite':
            always = f"{obj} = {id_properties}, {obj} += {properties}"
        elif on_collision == 'preferold':
            on_match = f"{obj} = apoc.map.merge({properties}, properties({obj}))"
            on_create = f"{obj} += {properties}"
        elif on_collision == 'prefernew':
            always = f"{obj} += {properties}"
        elif on_collision == 'leavealone':
            on_create = f"{obj} = {properties}"
        elif on_collision == 'raise':
            on_create = f"{obj} += {properties}"
            on_match = f"{obj} = apoc.map.merge({properties}, properties({obj}))"
            after = f"CALL apoc.util.validate(not isempty({obj}._collisions)" \
                    f", 'properties of {obj} collided with new: %s != %s'," \
                    f" [apoc.map.fromPairs([x in {obj}._collisions | [x, {properties}[x]]]), apoc.map.fromPairs([x in {obj}._collisions | [x, {obj}[x]]])])"
        else:
            raise ValueError(f"Unknown on_collision `{on_collision}`. Accepted values are 'rewrite', 'prefernew', 'prefer', 'leavealone', 'raise'")
        on_create = ', \n\t\t'.join(nonempty([on_create, f'{obj}._dbcreated = time0, {obj}._dbupdated = time0, {obj}._collisions = []']))
        on_match = ", \n\t\t".join(nonempty([set_collision, on_match]))
        always = ", \n\t\t".join(nonempty([always, set_modified]))
        return CollisionManager(on_create, on_match, always, after, prefix)

    def extend(self, x: 'CollisionManager') -> 'CollisionManager':
        return CollisionManager(', \n\t\t'.join([self.on_create, x.on_create]).strip('\t\n, '),
                                ', \n\t\t'.join([self.on_match, x.on_match]).strip('\t\n, '),
                                ', \n\t\t'.join([self.on_always, x.on_always]).strip('\t\n, '),
                                '\n\t'.join([self.on_after, x.on_after]).strip('\t\n '),
                                self.prefix)

    def __iadd__(self, other):
        return reduce(lambda x, y: x.extend(y), [self]+list(other))

    @property
    def created(self):
        prefix = 'ON CREATE ' if self.prefix else ''
        return f"\t{prefix}SET {self.on_create}" if self.on_create else None

    @property
    def matched(self):
        prefix = 'ON MATCH ' if self.prefix else ''
        return f"\t{prefix}SET {self.on_match}" if self.on_match else None

    @property
    def after(self):
        return f"\tWITH * \n\t{self.on_after}" if self.on_after else None

    @property
    def always(self):
        return f"\tSET {self.on_always}" if self.on_always else None

    def __str__(self):
        return '\n'.join([i for i in [self.created, self.matched, self.always, self.after] if i])


class Merge(Statement):
    pass


class MergeNode(Merge):
    """
    Merge a node only on class and properties
    on_collision can be either `overwrite` or `ignore`
    """
    ids = ['labels', 'on_collision']

    def __init__(self, labels, ident_properties, other_properties, on_collision, graph, parameters=None):
        self.ident_properties = to_cypher_dict_or_variable(ident_properties)
        self.other_properties = to_cypher_dict_or_variable(other_properties)
        super(MergeNode, self).__init__([self.ident_properties, self.other_properties], graph, parameters)
        self.to_node = self.make_variable(labels[0])
        self.labels = ':'.join(labels)
        self.on_collision = on_collision

    def make_cypher(self, ordering: list) -> str:
        merge = f"MERGE ({self.to_node}:{self.labels} {self.ident_properties})"
        if self.other_properties:
            collision = CollisionManager.from_neo_object(self.to_node, self.other_properties, self.ident_properties, self.on_collision)
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
            self.r1 = self.make_variable('r')
        if len(parent_nodes) > 1:
            self.r2 = self.make_variable('r')
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
        parent_collisions = [CollisionManager.from_neo_object(r, rp, ri, self.on_collision) for r, rp, ri in
                             zip(self.rels, self.parent_rel_other_properties, self.parent_rel_ident_properties)]
        merge = f'MERGE {merge}'
        if self.other_properties:
            collision = CollisionManager.from_neo_object(self.to_node, self.other_properties, self.ident_properties, self.on_collision)
            collision += parent_collisions
            return f"{merge}\n{collision}\nWITH *"
        elif parent_collisions:
            parent_collisions[0] += parent_collisions[1:]
            return f"{merge}\n{parent_collisions[0]}\nWITH *"
        else:
            return f"{merge}\nWITH *"


class AdvancedMergeNodeAndRelationships(MergeSimpleNodeAndRelationships):
    """
    A complex relationship of many parents and one child
    If the entire pattern exists use on_match, otherwise create it and use on_create
    Merge node only on ids and
    Merge each rel independently on ids
    pattern now is guaranteed to exist

    if all parts were created now:
        use on_create
    if no parts were created now:
        use on_match
    if node was created but not rels:
        not possible
    if rels where created but not node:
        this should never happen because node must be created along with the rels since its uniqueness depends on them
        raise error

    In cypher we do this:
        0. unwind collections
        1. merge each part only on ids
        2. collect with created flag & validate the merge holistically
        3. make two lists from parts (creaed/matched). one should be always empty
        4. do foreach set stuff on each list
        5. do the always and after
    """

    def make_cypher(self, ordering: list) -> str:
        merge_node = MergeNode(self.labels, self.ident_properties, None, self.on_collision, self.graph, self.parameters)
        merge_rels = []
        for r, ri, rp, rt in zip(self.rels, self.parent_rel_ident_properties, self.parent_rel_other_properties, self.parent_rel_types):
            merge_rel = MergeRel(ri, None, rt, self.on_collision, self.graph, self.parameters)
            merge_rels.append(merge_rel)
        collect_parts_and_flags = ""
        validate_merge = ""
        collision = CollisionManager.from_neo_object('part[0]', 'part[1]', 'part[2]', self.on_collision, False)
        matched = f"FOREACH ( part in {matched_parts} | {collision.matched})"
        created = f"FOREACH ( part in {created_parts} | {collision.created})"
        called = f"CALL {{ WITH {all_parts} WITH {all_parts} UNWIND {all_parts} as part\n{collision.always}\n{collision.after} }}"
        