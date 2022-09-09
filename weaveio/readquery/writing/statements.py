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
    def __init__(self, on_create=None, on_match=None, on_always=None, on_after=None, prefix=True, indent=0):
        self.on_create = on_create
        self.on_match = on_match
        self.on_always = on_always
        self.on_after = on_after
        self.prefix = prefix
        self.indent = indent

    @classmethod
    def from_neo_object(cls, obj: str, properties: Dict[str, str], id_properties: Dict[str, str], on_collision: str, prefix=True, indent=0):
        properties = to_cypher_dict_or_variable(properties)
        id_properties = to_cypher_dict_or_variable(id_properties)
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
                    f", 'properties of {obj} collided with new: %s'," \
                    f" [{obj}._collisions])"
        else:
            raise ValueError(f"Unknown on_collision `{on_collision}`. Accepted values are 'rewrite', 'prefernew', 'prefer', 'leavealone', 'raise'")
        on_create = ', \n\t\t'.join(nonempty([on_create, f'{obj}._dbcreated = time0, {obj}._dbupdated = time0, {obj}._collisions = []']))
        on_match = ", \n\t\t".join(nonempty([set_collision, on_match]))
        always = ", \n\t\t".join(nonempty([always, set_modified]))
        return CollisionManager(on_create, on_match, always, after, prefix, indent)

    def extend(self, x: 'CollisionManager') -> 'CollisionManager':
        return CollisionManager(', \n\t\t'.join([self.on_create, x.on_create]).strip('\t\n, '),
                                ', \n\t\t'.join([self.on_match, x.on_match]).strip('\t\n, '),
                                ', \n\t\t'.join([self.on_always, x.on_always]).strip('\t\n, '),
                                '\n\t'.join([self.on_after, x.on_after]).strip('\t\n '),
                                self.prefix, self.indent)

    def __iadd__(self, other):
        return reduce(lambda x, y: x.extend(y), [self]+list(other))

    @property
    def indents(self):
        return '\t' * self.indent

    @property
    def created(self):
        prefix = 'ON CREATE ' if self.prefix else ''
        return f"{self.indents}\t{prefix}SET {self.on_create}" if self.on_create else ""

    @property
    def matched(self):
        prefix = 'ON MATCH ' if self.prefix else ''
        return f"{self.indents}\t{prefix}SET {self.on_match}" if self.on_match else ""

    @property
    def after(self):
        return f"{self.indents}\tWITH * \n\t{self.on_after}" if self.on_after else ""

    @property
    def always(self):
        return f"{self.indents}\tSET {self.on_always}" if self.on_always else ""

    def __str__(self):
        return '\n'.join([i for i in [self.created, self.matched, self.always, self.after] if i])


class Merge(Statement):
    """
    Merges a node based on its `ident_properties` and then updates its `other_properties` by the
    given `on_collision` method
    property variables are assumed to be references to cypher variables
    """

    ids = ['labels', 'on_collision', 'ident_properties', 'other_properties']

    def __init__(self, labels: List[str], ident_properties: Dict[str,str], other_properties: Dict[str,str],
                 on_collision: str, graph, parameters=None):
        self.ident_properties = ident_properties
        self.other_properties = other_properties
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
        merge = f"MERGE ({self.to_node}:{self.labels} {to_cypher_dict_or_variable(self.ident_properties)})"
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
        merge = f"MERGE ({self.from_node})-[{self.rel}:{self.labels} {to_cypher_dict_or_variable(self.ident_properties)}]->({self.to_node})"
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
        self.parent_rel_ident_properties = parent_rel_ident_properties
        self.parent_rel_other_properties = parent_rel_other_properties
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
        parent_rel_ident_properties = [to_cypher_dict_or_variable(x) for x in self.parent_rel_ident_properties]
        parent_rel_other_properties = [to_cypher_dict_or_variable(x) for x in self.parent_rel_other_properties]

        merge = f"({self.to_node}:{self.labels} {to_cypher_dict_or_variable(self.ident_properties)})"
        if self.r1:
            merge = f"({self.parent_nodes[0]})-[{self.r1}:{self.parent_rel_types[0]} {parent_rel_ident_properties[0]}]->{merge}"
        if self.r2:
            merge = f"{merge}<-[{self.r2}:{self.parent_rel_types[1]} {parent_rel_ident_properties[1]}]-({self.parent_nodes[1]})"
        parent_collisions = [CollisionManager.from_neo_object(r, rp, ri, self.on_collision) for r, rp, ri in
                             zip(self.rels, parent_rel_other_properties, parent_rel_ident_properties)]
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

    def __init__(self, labels: List[str], ident_properties: Dict[str, str], other_properties: Dict[str, str],
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
            c += '// auto-enumerate\n'
            for parents, (ids, others) in self.rels.items():
                c += f"WITH *, range(0, size({parents})-1) as {ids['_order']}\n"
        # do call signature
        c += "CALL {with "
        for parents, (ids, others) in self.rels.items():
            c += f"{parents}, "
            for v in list(ids.values()) + list(others.values()):
                c += f"{v}, "
        c += 'time0'
        c = c.strip(', ')
        c += '\nWITH '
        # turn collections into maps i.e. [a0, a1, ...] -> {nodeid0: a0, ...}
        for parents, (ids, others) in self.rels.items():
            c += f'{parents}, '
            for v in list(ids.values()) + list(others.values()):
                c += f"apoc.map.fromLists([x in {parents} | toString(id(x))], {v}) as {v}, "
        c += 'time0'
        c = c.strip(', ') + '\n'
        # so now self.rels looks like {parent_node_collection_var: [{'key': nodeid2prop_map}, ...]}
        for parents in self.rels:
            c += f"CALL apoc.lock.nodes({parents})\n"  # lock nodes
        c += f"WITH *, head({parents}) as first\n" # take a parent to act as the first rel used
        first_identity = ', '.join([f'{k}: {mapping}[toString(id(first))]' for k, mapping in self.rels[parents][0].items()])
        c += f"OPTIONAL MATCH (first)-[:{self.rel_type} {{{first_identity}}}]->(d:{self.labels} {self.ident_properties})\n"
        identities = []
        for parents, (ids, others) in self.rels.items():
            i = ', '.join([f'{k}: {v}[toString(id(x))]' for k, v in ids.items()])
            identities.append(f"all(x in {parents} where (x)-[:{self.rel_type} {{{i}}}]->(d))")
        identities = "\n\tand ".join(identities)
        c += f'WHERE {identities}\n'

        creates = []
        matches = []
        for parents, (ids, others) in self.rels.items():
            i = ', '.join([f'{k}: ${v}[toString(id(x))]' for k, v in ids.items()])
            o = ', '.join([f'{k}: ${v}[toString(id(x))]' for k, v in others.items()])
            setup = f'WITH *, {{{i}}} as ids, {{{o}}} as others, $time0 as time0'
            collision = CollisionManager.from_neo_object('r', 'others', 'ids', self.on_collision, False)
            creates.append(f"\tUNWIND ${parents} as x \n\t\t{setup}\n\t\tCREATE (x)-[r:{self.rel_type}]->(dd)\n\t\t{collision.created}\n\t\t{collision.always}\n\t\t{collision.after}")
            matches.append(f"\tUNWIND ${parents} as x \n\t\tMATCH (x)-[r:{self.rel_type} {{{i}}}]->(d)\n\t\t{setup}, d as dd\n\t\t{collision.matched}\n\t\t{collision.always}\n\t\t{collision.after}")
        node_collision = CollisionManager.from_neo_object('dd', self.other_properties, self.ident_properties, self.on_collision, False)
        creates.append(f"{node_collision.created}\n{node_collision.always}\n{node_collision.after}")
        matches.append(f"{node_collision.matched}\n{node_collision.always}\n{node_collision.after}")
        create = '\n'.join(creates)
        match = '\n'.join(matches)

        var = ', '.join([f"{x}:{x}" for parents, (ids, others) in self.rels.items() for x in ['d', 'time0', parents]+list(ids.values())+list(others.values())])
        var_import = f"{{{var}}}"
        c += f'CALL apoc.do.when(d is null, "CREATE (dd:{self.labels} {self.ident_properties}) WITH *\n\t{create} RETURN dd", \n\t"{match} RETURN d as dd", {var_import})\n yield value\n'
        c += f" RETURN value.dd as {self.to_node}}} RETURN {self.to_node}"
        return c
