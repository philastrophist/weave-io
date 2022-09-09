from functools import reduce
from typing import Dict

from . import to_cypher_dict_or_variable, nonempty


class CollisionManager:
    def __init__(self, on_create=None, on_match=None, on_always=None, on_after=None, prefix=True, indent=0):
        self.on_create = on_create
        self.on_match = on_match
        self.on_always = on_always
        self.on_after = on_after
        self.prefix = prefix
        self.indent = indent

    @classmethod
    def from_neo_object(cls, obj: str, properties, id_properties, on_collision: str, prefix=True, indent=0):
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
        elif on_collision == 'leavealone':  # if matched, dont even add anything
            on_create = f"{obj} = {id_properties}, {obj} += {properties}"
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
