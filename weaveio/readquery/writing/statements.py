from collections import namedtuple
from functools import reduce

from typing import List, Dict, Tuple

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
    ids = ['labels', 'on_collision', 'ident_properties', 'other_properties']

    def __init__(self, labels, ident_properties, other_properties, on_collision, graph, parameters=None):
        self.ident_properties = to_cypher_dict_or_variable(ident_properties)
        self.other_properties = to_cypher_dict_or_variable(other_properties)
        super(Merge, self).__init__([self.ident_properties, self.other_properties], graph, parameters)
        self.labels = ':'.join(labels)
        self.on_collision = on_collision


class MergeNode(Merge):
    """
    Merge a node only on class and properties
    """

    def __init__(self, labels, ident_properties, other_properties, on_collision, graph, parameters=None):
        super().__init__(labels, ident_properties, other_properties, on_collision, graph, parameters)
        self.to_node = self.make_variable(labels[0])

    def make_cypher(self, ordering: list) -> str:
        merge = f"MERGE ({self.to_node}:{self.labels} {self.ident_properties})"
        if self.other_properties:
            collision = CollisionManager.from_neo_object(self.to_node, self.other_properties, self.ident_properties, self.on_collision)
            return f"{merge}\n{collision}\nWITH *"
        return f"{merge}\nWITH *"


class MergeRel(Merge):
    def __init__(self, from_node, to_node, labels, ident_properties, other_properties, on_collision, graph, parameters=None):
        super().__init__(labels, ident_properties, other_properties, on_collision, graph, parameters)
        self.from_node = from_node
        self.to_node = to_node
        self.inputs += [self.from_node, self.to_node]
        self.rel = self.make_variable('rel')

    def make_cypher(self, ordering: list) -> str:
        merge = f"MERGE ({self.from_node})-[{self.rel}:{self.labels} {self.ident_properties}]->({self.to_node})"
        if self.other_properties:
            collision = CollisionManager.from_neo_object(self.rel, self.other_properties, self.ident_properties, self.on_collision)
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


class AdvancedMergeNodeAndRelationships(MergeNode):
    """
    A complex relationship of many parents and one child
    If the entire pattern exists use on_match, otherwise create it and use on_create

    Parents are already known, just the rels and child need finding
        - # children/parent is always less than or equal to # globally unique children indexed by id:
            Expand from parent and filter to child

    Input parent_nodes etc are expected to be in a collection
    Input properties are expected to be in individual collections
    e.g.
    >>> AdvancedMergeNodeAndRelationships(['OB'], {'id': 'id1'}, {}, {'fibre_targets1': [{'a': 'a0'}, {}}, 'is_required_by', True, 'raise', graph)
    """
    ids = Merge.ids + ['rel_type, rel_ident_properties_collection, rel_other_properties_collection', 'ordered']

    def __init__(self, labels: List[str], ident_properties: Dict[str], other_properties: Dict[str],
                 rels: Dict[str, Tuple[Dict[str, str], Dict[str, str]]],
                 rel_type: str,
                 ordered, on_collision, graph, parameters=None):
        super().__init__(labels, ident_properties, other_properties, on_collision, graph, parameters)
        self.ordered = ordered
        self.rel_type = rel_type
        self.rels = rels
        for parent, (ids, others) in self.rels.items():
            if self.ordered:
                ids['_order'] = self.make_variable('_order')
            self.inputs.append(parent)
            for d in [ids, others]:
                self.inputs += list(d.values())


    def make_cypher(self, ordering: list) -> str:
        c = ""
        if self.ordered:
            for parent, (ids, others) in self.rels.items():
                c += '// auto-enumerate\n'
                c += f"WITH *, range(0, size({parent})-1) as {ids['_order']}\n"
        # do call signature
        c += "CALL {with "
        for parent, (ids, others) in self.rels.items():
            c += f", {parent}"
            for v in list(ids.values()) + list(others.values()):
                c += f", {v}"
        c += '\nWITH '
        # turn collections into maps
        for parent, (ids, others) in self.rels.items():
            c += f'{parent}, '
        for v in list(ids.values()) + list(others.values()):
            c += f"apoc.map.fromLists([x in {parent} | toString(id(x))], {v}) as {v}, "
        for parent in self.rels:
            c += f"\nCALL apoc.lock.nodes({parent})\n"  # lock nodes
        c += f"WITH *, head({parent}) as first\n" # take a parent to act as the first rel used
        first_ident = ', '.join([f'{k}: {v}[toString(id(first))].{k}]' for ids, _ in self.rels[parent] for k, v in ids.items()])
        c += f"OPTIONAL MATCH (first)-[:{self.rel_type} {{{first_ident}}}]->(d:{':'.join(self.labels)})\n"
        tail_idents = []
        for parent, (ids, others) in self.rels.items():
            i = ', '.join([f'{k}: {v}[toString(id(x))].{k}]' for k, v in ids.items()])
            tail_idents.append(f"WHERE all(x in tail({parent}) WHERE (x)-[:{self.rel_type} {{{i}}}]->(d))")
        tail_idents = "\n\tand ".join(tail_idents)
        c += f'WHERE {tail_idents}\n'
        c += f"""CALL apoc.do.where(d is null,
                "CREATE (dd:{self.labels}) FOREACH (x in ${self.parent_nodes} | CREATE (x)-[r:REL {id: ${self.parent_ids}[toString(id(x))].id}]->(dd) ) RETURN dd as d", 
                "RETURN d as d", {d:d, {self.parent_nodes}:{self.parent_nodes}, {self.parent_ids}:{self.parent_ids}}) yield value
            RETURN value.d as d
        }}
        """
