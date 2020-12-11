from textwrap import dedent
from typing import List, Dict, Union, Tuple, Optional

from . import CypherQuery
from .base import camelcase, Varname, Statement, CypherVariable


class MatchNode(Statement):
    keyword = 'MATCH'

    def __init__(self, labels: List[str], properties: dict):
        self.labels = [camelcase(l) for l in labels]
        self.properties = {Varname(k): v for k, v in properties.items()}
        self.out = CypherVariable(labels[-1])
        inputs = [v for v in properties.values() if isinstance(v, CypherVariable)]
        super(MatchNode, self).__init__(inputs, [self.out])

    def to_cypher(self):
        labels = ':'.join(map(str, self.labels))
        return f"{self.keyword} ({self.out}:{labels} {self.properties})"


class MatchRelationship(Statement):
    def __init__(self, parent, child, reltype: str, properties: dict):
        self.parent = parent
        self.child = child
        self.reltype = reltype
        self.properties = {Varname(k): v for k, v in properties.items()}
        inputs = [self.parent, self.child] + [v for v in properties.values() if isinstance(v, CypherVariable)]
        self.out = CypherVariable(reltype)
        super().__init__(inputs, [self.out])

    def to_cypher(self):
        reldata = f'[{self.out}:{self.reltype} {self.properties}]'
        return f"MATCH ({self.parent})-{reldata}->({self.child})"


class CollisionManager(Statement):
    def __init__(self, out, identproperties: Dict[str, Union[str, int, float, CypherVariable]],
                 properties: Dict[str, Union[str, int, float, CypherVariable]], collision_manager='track&flag'):
        self.out = out
        self.properties = {Varname(k): v for k, v in properties.items()}
        self.identproperties = {Varname(k): v for k, v in identproperties.items()}
        self.propvar = CypherVariable('props')
        self.colliding_keys = CypherVariable('colliding_keys')
        self.value = CypherVariable('unnamed')
        inputs = [v for v in properties.values() if isinstance(v, CypherVariable)]
        inputs += [v for v in identproperties.values() if isinstance(v, CypherVariable)]
        outputs = [self.out, self.propvar]
        if collision_manager == 'track&flag':
            outputs += [self.value, self.colliding_keys]
        self.collision_manager = collision_manager
        super().__init__(inputs, outputs)

    @property
    def on_match(self):
        if self.collision_manager == 'overwrite':
            return f"SET {self.out} += {self.propvar}   // overwrite with new colliding properties"
        return f"SET {self.out} = apoc.map.merge({self.propvar}, properties({self.out}))   // update, keeping the old colliding properties"

    @property
    def on_create(self):
        return f"SET {self.out}._dbcreated = time0, {self.out} += {self.propvar}  // setup as standard"

    @property
    def on_run(self):
        return f'SET {self.out}._dbupdated = time0  // always set updated time '

    @property
    def merge_statement(self):
        raise NotImplementedError

    @property
    def collision_record(self):
        raise NotImplementedError

    @property
    def collision_record_input(self):
        raise NotImplementedError

    @property
    def post_merge(self):
        return dedent(f"""
    ON MATCH {self.on_match}
    ON CREATE {self.on_create}
    {self.on_run}""")

    @property
    def pre_merge(self):
        return f"WITH *, {self.properties} as {self.propvar}"

    @property
    def merge_paragraph(self):
        return f"""
        {self.pre_merge}
        {self.merge_statement}
        {self.post_merge}
        """

    def to_cypher(self):
        query = self.merge_paragraph
        if self.collision_manager == 'track&flag':
            query += f"""
            WITH *, [x in apoc.coll.intersection(keys({self.propvar}), keys(properties({self.out}))) where {self.propvar}[x] <> {self.out}[x]] as {self.colliding_keys}
            CALL apoc.do.when(size({self.colliding_keys}) > 0, 
                "{self.collision_record} SET c = $collisions SET c._dbcreated = $time RETURN $time", 
                "RETURN $time",
                {{{self.collision_record_input}, collisions: apoc.map.fromLists({self.colliding_keys}, apoc.map.values({self.propvar}, {self.colliding_keys})), time:time0}}) yield value as {self.value}
            """
        return dedent(query)


class MergeNode(CollisionManager):
    def __init__(self, labels: List[str], identproperties: Dict[str, Union[str, int, float, CypherVariable]],
                 properties: Dict[str, Union[str, int, float, CypherVariable]], collision_manager='track&flag'):
        self.labels = [camelcase(l) for l in labels]
        out = CypherVariable(labels[-1])
        super().__init__(out, identproperties, properties, collision_manager)

    @property
    def merge_statement(self):
        labels = ':'.join(map(str, self.labels))
        return f'MERGE ({self.out}: {labels} {self.identproperties})'

    @property
    def collision_record(self):
        return f"WITH $innode as innode CREATE (c:_Collision)-[:COLLIDES]->(innode)"

    @property
    def collision_record_input(self):
        return f"innode: {self.out}"


class MergeRelationship(CollisionManager):
    def __init__(self, parent, child, reltype: str, identproperties: dict, properties: dict, collision_manager='track&flag'):
        self.parent = parent
        self.child = child
        self.reltype = reltype
        out = CypherVariable(reltype)
        super().__init__(out, identproperties, properties, collision_manager)

    @property
    def merge_statement(self):
        return f'MERGE ({self.parent})-[{self.out}:{self.reltype} {self.identproperties}]->({self.child})'

    @property
    def collision_record(self):
        return f"WITH $a as a, $b as b CREATE (a)-[c:COLLIDES]->(b)"

    @property
    def collision_record_input(self):
        return f"a:{self.parent}, b:{self.child}"


class MergeDependentNode(CollisionManager):
    def __init__(self, labels: List[str], identproperties: Dict[str, Union[str, int, float, CypherVariable]],
                 properties: Dict[str, Union[str, int, float, CypherVariable]],
                 parents: List[CypherVariable],
                 reltypes: List[str],
                 relidentproperties: List[Dict[str, Union[str, int, float, CypherVariable]]],
                 relproperties: List[Dict[str, Union[str, int, float, CypherVariable]]],
                 collision_manager='track&flag'):
        if not (len(parents) == len(reltypes) == len(relproperties) == len(relidentproperties)):
            raise ValueError(f"Parents must have the same length as reltypes, relproperties, relidentproperties")
        self.labels = [camelcase(l) for l in labels]
        self.relidentproperties = relidentproperties
        self.relproperties = relproperties
        self.parents = parents
        self.outnode = CypherVariable(labels[-1])
        self.rels = [CypherVariable(reltype) for reltype in reltypes]
        self.dummy = CypherVariable('dummy')
        self.reltypes = reltypes
        self.relpropsvars = [CypherVariable('rel') for _ in relproperties]
        super().__init__(self.outnode, identproperties, properties, collision_manager)
        self.input_variables += parents
        self.output_variables += self.rels
        self.output_variables.append(self.dummy)

    @property
    def pre_merge(self):
        line = f"WITH *, {self.properties} as {self.propvar}"
        for relprop, relpropsvar in zip(self.relproperties, self.relpropsvars):
            line += f', {relprop} as {relpropsvar}'
        return line

    @property
    def merge_statement(self):
        labels = ':'.join(map(str, self.labels))
        relations = []
        for i, (parent, reltype, relidentprop) in enumerate(zip(self.parents, self.reltypes, self.relidentproperties)):
            rel = f'({parent})-[:{reltype} {relidentprop}]->'
            if i == 0:
                child = f'({self.dummy}: {labels} {self.identproperties})'
            else:
                child = f'({self.dummy})'
            relations.append(rel + child)
        optional_match = 'OPTIONAL MATCH ' + ',\n'.join(relations)
        create = 'CREATE ' + ',\n'.join(relations)
        parent_dict = ','.join([f'{p}:{p}' for p in self.parents])
        parent_alias = ','.join([f'${p} as {p}' for p in self.parents])
        query = f"""
        WITH *, {self.properties} as {self.propvar}
        CALL apoc.lock.nodes([{self.parents}]) // let's lock ahead this time
        {optional_match}
        call apoc.do.when({self.dummy} IS NULL, 
                "WITH {parent_alias}
                {create}
                SET {self.dummy} += ${self.propvar}
                RETURN {self.dummy} as {self.out}",   // created
            "RETURN $d as {self.out}",  // matched 
            {{d:{self.dummy}, {parent_dict}, {self.propvar}:{self.propvar}}}) yield value as {self.value}
        WITH *, {self.value}.{self.out} as {self.out} 
        """
        return dedent(query)

    @property
    def on_match(self):
        if self.collision_manager == 'overwrite':
            query = f"SET {self.out} += {self.propvar}   // overwrite with new colliding properties"
        else:
            query = f"SET {self.out} = apoc.map.merge({self.propvar}, properties({self.out}))   // update, keeping the old colliding properties"
        for r, rprops in zip(self.rels, self.relpropsvars):
            if self.collision_manager == 'overwrite':
                query += f"\nSET {r} += {rprops}"
            else:
                query += f"\nSET {r} = apoc.map.merge({rprops}, properties({r}))"
        return query

    @property
    def on_create(self):
        query = f"SET {self.out}._dbcreated = time0, {self.out} += {self.propvar}  // setup as standard"
        for r, rprops in zip(self.rels, self.relpropsvars):
            query += f'\nSET {r}._dbupdated = time0, {r} += {rprops}'
        return query

    @property
    def on_run(self):
        query = f"SET {self.out}._dbupdated = time0  // always set updated time"
        for r in self.rels:
            query += f'\nSET {r}._dbupdated = time0'
        return query

    @property
    def post_merge(self):
        return dedent(f"""call apoc.do.when({self.dummy} IS NULL, 
            "{self.on_create}",
            "{self.on_match}"
            {self.on_run}""")

    @property
    def collision_record(self):
        return f"WITH $innode as innode CREATE (c:_Collision)-[:COLLIDES]->(innode)"

    @property
    def collision_record_input(self):
        return f"innode: {self.out}"


class SetVersion(Statement):
    def __init__(self, parents: List[CypherVariable], reltypes: List[str], childlabel: str, child: CypherVariable, childproperties: dict):
        if len(reltypes) != len(parents):
            raise ValueError(f"reltypes must be the same length as parents")
        self.parents = parents
        self.reltypes = reltypes
        self.childlabel = camelcase(childlabel)
        self.childproperties = {Varname(k): v for k, v in childproperties.items()}
        self.child = child
        super(SetVersion, self).__init__(self.parents, [])

    def to_cypher(self):
        matches = ', '.join([f'({p})-[:{r}]->(c:{self.childlabel} {self.childproperties})' for p, r in zip(self.parents, self.reltypes)])
        query = [
            f"WITH * CALL {{",
                f"\t WITH {','.join(map(str, self.parents))}, {self.child}",
                f"\t OPTIONAL MATCH {matches}"
                f"\t WHERE id(c) <> id({self.child})",
                f"\t WITH {self.child}, max(c.version) as maxversion",
                f"\t SET {self.child}.version = coalesce({self.child}['version'], maxversion + 1, 0)",
                f"\t RETURN {self.child['version']}",
            f"}}"
        ]
        return '\n'.join(query)


def match_node(labels, properties):
    query = CypherQuery.get_context()  # type: CypherQuery
    statement = MatchNode(labels, properties)
    query.add_statement(statement)
    return statement.out


def match_relationship(parent, child, reltype, properties):
    query = CypherQuery.get_context()  # type: CypherQuery
    statement = MatchRelationship(parent, child, reltype, properties)
    query.add_statement(statement)
    return statement.out


def merge_single_node(labels, identproperties, properties, collision_manager='track&flag'):
    query = CypherQuery.get_context()  # type: CypherQuery
    statement = MergeNode(labels, identproperties, properties, collision_manager)
    query.add_statement(statement)
    return statement.out


def merge_relationship(parent, child, reltype, identproperties, properties, collision_manager='track&flag'):
    query = CypherQuery.get_context()  # type: CypherQuery
    statement = MergeRelationship(parent, child, reltype, identproperties, properties, collision_manager)
    query.add_statement(statement)
    return statement.out


def merge_dependent_node(labels, identproperties, properties, parents, reltypes, relidentproperties, relproperties,
                         collision_manager='track&flag'):
    query = CypherQuery.get_context()  # type: CypherQuery
    statement = MergeDependentNode(labels, identproperties, properties, parents, reltypes, relidentproperties, relproperties,
                                   collision_manager)
    query.add_statement(statement)
    return statement.outnode


def set_version(parents, reltypes, childlabel, child, childproperties):
    query = CypherQuery.get_context()  # type: CypherQuery
    statement = SetVersion(parents, reltypes, childlabel, child, childproperties)
    query.add_statement(statement)


def merge_node(labels, identproperties, properties=None,
               parents: Dict[CypherVariable, Union[Tuple[str, Optional[Dict], Optional[Dict]], str]] = None,
               versioned_label=None,
               versioned_properties=None,
               collision_manager='track&flag'):
    if properties is None:
        properties = {}
    if parents is None:
        parents = {}
    parent_list = []
    reltype_list = []
    relidentproperties_list = []
    relproperties_list = []
    for parent, reldata in parents.items():
        if isinstance(reldata, str):
            reldata = [reldata]
        parent_list.append(parent)
        reltype_list.append(reldata[0])
        if len(reldata) == 2:
            relidentproperties_list.append(reldata[1])
        else:
            relidentproperties_list.append({})
        if len(reldata) == 3:
            relproperties_list.append(reldata[2])
        else:
            relproperties_list.append({})
    if len(parents):
        node = merge_dependent_node(labels, identproperties, properties, parent_list,
                             reltype_list, relidentproperties_list, relproperties_list, collision_manager)
    else:
        node = merge_single_node(labels, identproperties, properties, collision_manager)
    if versioned_label is not None:
        if versioned_properties is None:
            versioned_properties = {}
        set_version(parent_list, reltype_list, versioned_label, node, versioned_properties)
    return node
