import inspect
from functools import wraps, partial
from typing import Tuple, Dict, Type, Union, List, Optional
from warnings import warn

from . import writequery
from .writequery import CypherQuery, Unwind, Collection, CypherVariable
from .context import ContextError
from .utilities import Varname

def _convert_types_to_node(x):
    if isinstance(x, dict):
        return {_convert_types_to_node(k): _convert_types_to_node(v) for k, v in x.items()}
    elif isinstance(x, (list, set, tuple)):
        return x.__class__([_convert_types_to_node(i) for i in x])
    elif isinstance(x, Graphable):
        return x.node
    else:
        return x

def hierarchy_query_decorator(function):
    @wraps(function)
    def inner(*args, **kwargs):
        args = _convert_types_to_node(args)
        kwargs = _convert_types_to_node(kwargs)
        return function(*args, **kwargs)
    return inner


unwind = hierarchy_query_decorator(writequery.unwind)
merge_node = hierarchy_query_decorator(writequery.merge_node)
match_node = hierarchy_query_decorator(writequery.match_node)
match_pattern_node = hierarchy_query_decorator(writequery.match_pattern_node)
collect = hierarchy_query_decorator(writequery.collect)
merge_relationship = hierarchy_query_decorator(writequery.merge_relationship)
set_version = hierarchy_query_decorator(writequery.set_version)


def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


FORBIDDEN_LABELS = []
FORBIDDEN_PROPERTY_NAMES = []
FORBIDDEN_LABEL_PREFIXES = ['_']
FORBIDDEN_PROPERTY_PREFIXES = ['_']
FORBIDDEN_IDNAMES = ['idname']


class RuleBreakingException(Exception):
    pass


class Multiple:
    def __init__(self, node, minnumber=1, maxnumber=None, constrain=None, idname=None):
        self.node = node
        self.minnumber = minnumber
        self.maxnumber = maxnumber
        self.name = node.plural_name
        self.singular_name = node.singular_name
        self.plural_name = node.plural_name
        self.idname = self.node.idname
        self.relation_idname = idname
        try:
            self.factors =  self.node.factors
        except AttributeError:
            self.factors = []
        try:
            self.parents = self.node.parents
        except AttributeError:
            self.parents = []
        self.constrain = [] if constrain is None else constrain

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}])>"


class One2One(Multiple):
    def __init__(self, node, constrain=None, idname=None):
        super(One2One, self).__init__(node, 1, 1, constrain, idname)

    def __repr__(self):
        return f"<One2One({self.node})>"


class Indexed:
    def __init__(self, hdu_name, column_name=None):
        self.name = hdu_name
        self.column_name = column_name


class GraphableMeta(type):
    def __new__(meta, name: str, bases, _dct):
        dct = {'is_template': False}
        dct.update(_dct)
        dct['aliases'] = dct.get('aliases', [])
        dct['aliases'] += [a for base in bases for a in base.aliases]
        if dct.get('plural_name', None) is None:
            dct['plural_name'] = name.lower() + 's'
        dct['singular_name'] = name.lower()
        if dct['plural_name'] != dct['plural_name'].lower():
            raise RuleBreakingException(f"plural_name must be lowercase")
        if dct['singular_name'] != dct['singular_name'].lower():
            raise RuleBreakingException(f"singular_name must be lowercase")
        idname = dct.get('idname', None)
        if idname in FORBIDDEN_IDNAMES:
            raise RuleBreakingException(f"You may not name an id as one of {FORBIDDEN_IDNAMES}")
        if not (isinstance(idname, str) or idname is None):
            raise RuleBreakingException(f"{name}.idname ({idname}) must be a string or None")
        if name[0] != name.capitalize()[0] or '_' in name:
            raise RuleBreakingException(f"{name} must have `CamelCaseName` style name")
        for factor in dct.get('factors', []) + ['idname'] + [dct['singular_name'], dct['plural_name']]:
            if factor != factor.lower():
                raise RuleBreakingException(f"{name}.{factor} must have `lower_snake_case` style name")
            if factor in FORBIDDEN_PROPERTY_NAMES:
                raise RuleBreakingException(f"The name {factor} is not allowed for class {name}")
            if any(factor.startswith(p) for p in FORBIDDEN_PROPERTY_PREFIXES):
                raise RuleBreakingException(f"The name {factor} may not start with any of {FORBIDDEN_PROPERTY_PREFIXES} for {name}")
        r = super(GraphableMeta, meta).__new__(meta, name, bases, dct)
        return r

    def __init__(cls, name, bases, dct):
        if cls.idname is not None and cls.identifier_builder is not None:
            raise RuleBreakingException(f"You cannot define a separate idname and an identifier_builder at the same time for {name}")
        if cls.indexes and (cls.idname is not None or cls.identifier_builder is not None):
            raise RuleBreakingException(f"You cannot define an index and an id at the same time for {name}")
        nparents_in_id = 0
        parentnames = {}
        for i in cls.parents:
            if isinstance(i, One2One):
                parentnames[i.singular_name] = (1, 1)
            elif isinstance(i, Multiple):
                parentnames[i.plural_name] = (i.minnumber, i.maxnumber)
            else:
                parentnames[i.singular_name] = (1, 1)
        if cls.identifier_builder is not None:
            for p in cls.identifier_builder:
                if p in parentnames:
                    mn, mx = parentnames[p]
                    if mn == 0:
                        raise RuleBreakingException(f"Cannot make an id from an optional (min=0) parent for {name}")
                    if mx != mn:
                        raise RuleBreakingException(f"Cannot make an id from an unbound (max!=min) parent for {name}")
                    nparents_in_id += mx
                elif p in cls.factors:
                    pass
                else:
                    raise RuleBreakingException(f"Unknown identifier source {p} for {name}")
        version_parents = []
        version_factors = []
        for i in cls.version_on:
            parents = [p.node if isinstance(p, One2One) else p for p in cls.parents]
            if i in [p.singular_name if isinstance(p, type) else p.name for p in parents]:
                version_parents.append(i)
            elif i in cls.factors:
                version_factors.append(i)
            else:
                raise RuleBreakingException(f"Unknown {i} to version on for {name}. Must refer to a parent or factor.")
        if len(version_factors) > 1 and len(version_parents) == 0:
            raise RuleBreakingException(f"Cannot build a version relative to nothing. You must version on at least one parent.")
        if not cls.is_template:
            if not (len(cls.indexes) or cls.idname or
                    (cls.identifier_builder is not None and len(cls.identifier_builder) > 0)):
                raise RuleBreakingException(f"{name} must define an indexes, idname, or identifier_builder")
        for i in cls.indexes:
            if i not in cls.parents and i not in cls.factors:
                raise RuleBreakingException(f"index {i} of {name} must be a factor or parent of {name}")
        if len(cls.hdus):
            hduclasses = {}
            for i, (hduname, hdu) in enumerate(cls.hdus.items()):
                if hdu is not None:
                    typename = name+hduname[0].upper()+hduname[1:]
                    typename = typename.replace('_', '')
                    hduclass = type(typename, (hdu, ), {'parents': [One2One(cls)], 'identifier_builder': [cls.singular_name, 'extn', 'name']})
                    hduclasses[hduname] = hduclass
                    if hduname in cls.factors or hduname in [p.singular_name if isinstance(p, type) else p.name for p in cls.parents]:
                        raise RuleBreakingException(f"There is already a factor/parent called {hduname} defined in {name}")
                    for base in bases:
                        if (hduname in base.factors or hduname in base.parents or hasattr(base, hduname)) and hduname not in base.hdus:
                            raise RuleBreakingException(f"There is already a factor/parent called {hduname} defined in {base}->{name}")
                    setattr(cls, hduname, hduclass)  # add as an attribute
            cls.hdus = hduclasses  # overwrite hdus
        if cls.concatenation_constants is not None:
            if len(cls.concatenation_constants):
                cls.factors = cls.factors + cls.concatenation_constants + ['concatenation_constants']
        clses = [i.__name__ for i in inspect.getmro(cls)]
        clses = clses[:clses.index('Graphable')]
        cls.neotypes = clses
        cls.products_and_factors = cls.factors + list(cls.products.keys())
        if cls.idname is not None:
            cls.products_and_factors.append(cls.idname)
        super().__init__(name, bases, dct)


class Graphable(metaclass=GraphableMeta):
    idname = None
    identifier = None
    indexer = None
    type_graph_attrs = {}
    plural_name = None
    singular_name = None
    parents = []
    uses_tables = False
    factors = []
    data = None
    query = None
    is_template = True
    products = {}
    indexes = []
    identifier_builder = None
    version_on = []
    hdus = {}
    produces = []
    concatenation_constants = []
    belongs_to = []
    products_and_factors = []

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, value):
        assert isinstance(value, CypherVariable)
        self._node = value

    @classmethod
    def requirement_names(cls):
        l = []
        for p in cls.parents:
            if isinstance(p, type):
                if issubclass(p, Graphable):
                    l.append(p.singular_name)
            else:
                if isinstance(p, One2One):
                    l.append(p.singular_name)
                if isinstance(p, Multiple):
                    l.append(p.plural_name)
                else:
                    raise RuleBreakingException(f"The parent list of a Hierarchy must contain "
                                                f"only other Hierarchies or Multiple(Hierarchy)")
        return l

    def add_parent_data(self, data):
        self.data = data

    def add_parent_query(self, query):
        self.query = query

    def __getattr__(self, item):
        if self.query is not None:
            warn('Lazily loading a hierarchy attribute can be costly. Consider using a more flexible query.')
            attribute = getattr(self.query, item)()
            setattr(self, item, attribute)
            return attribute
        raise AttributeError(f"Query not added to {self}, cannot search for {self}.{item}")

    @property
    def neoproperties(self):
        identifier_builder = [] if self.identifier_builder is None else self.identifier_builder
        d = {}
        for f in self.factors:
            if f not in identifier_builder and f != self.idname:
                value = getattr(self, f.lower())
                if value is not None:
                    d[f.lower()] = value
        return d

    @property
    def neoidentproperties(self):
        identifier_builder = [] if self.identifier_builder is None else self.identifier_builder
        d = {}
        if self.identifier is None and self.idname is not None:
            raise ValueError(f"{self} must have an identifier")
        if self.idname is None and self.identifier is not None:
            raise ValueError(f"{self} must have an idname to be given an identifier")
        elif self.idname is not None:
            d[self.idname] = self.identifier
            d['id'] = self.identifier
        for f in self.factors:
            if f in identifier_builder:
                value = getattr(self, f.lower())
                if value is not None:
                    d[f.lower()] = value
        return d


    def __init__(self, do_not_create=False, **predecessors):
        self.predecessors = predecessors
        self.data = None
        if do_not_create:
            return
        try:
            query = CypherQuery.get_context()  # type: CypherQuery
            collision_manager = query.collision_manager
        except ContextError:
            return
        merge_strategy = self.__class__.merge_strategy()
        version_parents = []
        if  merge_strategy == 'NODE FIRST':
            self.node = child = merge_node(self.neotypes, self.neoidentproperties, self.neoproperties,
                                           collision_manager=collision_manager)
            for k, parent_list in predecessors.items():
                type = 'is_required_by'
                if isinstance(parent_list, Collection):
                    with unwind(parent_list, enumerated=True) as (parent, i):
                        props = {'order': i}
                        merge_relationship(parent, child, type, {}, props, collision_manager=collision_manager)
                    parent_list = collect(parent)
                    if k in self.version_on:
                        raise RuleBreakingException(f"Cannot version on a collection of nodes")
                else:
                    for parent in parent_list:
                        props = {'order': 0}
                        merge_relationship(parent, child, type, {}, props, collision_manager=collision_manager)
                        if k in self.version_on:
                            version_parents.append(parent)
        elif merge_strategy == 'NODE+RELATIONSHIP':
            parentnames = [p.singular_name if isinstance(p, One2One) else p.plural_name if isinstance(p, Multiple) else p.singular_name for p in self.parents]
            parents = []
            others = []
            for k, parent_list in predecessors.items():
                if isinstance(parent_list, Collection):
                    raise TypeError(f"Cannot merge NODE+RELATIONSHIP for collections")
                if k in parentnames and k in self.identifier_builder:
                    parents += [p for p in parent_list]
                else:
                    others += [(i, p) for i, p in enumerate(parent_list)]
                if k in self.version_on:
                    version_parents += parent_list
            reltype = 'is_required_by'
            relparents = {p: (reltype, {'order': 0}, {}) for p in parents}
            child = self.node = merge_node(self.neotypes, self.neoidentproperties, self.neoproperties,
                                           parents=relparents, collision_manager=collision_manager)
            for i, other in others:
                merge_relationship(other, child, reltype, {'order': i}, {}, collision_manager=collision_manager)
        else:
            ValueError(f"Merge strategy not known: {merge_strategy}")
        if len(version_parents):
            version_factors = {f: self.neoproperties[f] for f in self.version_on if f in self.factors}
            set_version(version_parents, ['is_required_by'] * len(version_parents), self.neotypes[-1], child, version_factors)

    @classmethod
    def has_factor_identity(cls):
        if cls.identifier_builder is None:
            return False
        if len(cls.identifier_builder) == 0:
            return False
        for p in cls.parents:
            if isinstance(p, One2One):
                if p.singular_name in cls.identifier_builder:
                    return False
            elif isinstance(p, Multiple):
                if p.plural_name in cls.identifier_builder:
                    return False
            elif p.singular_name in cls.identifier_builder:
                return False
        return True

    @classmethod
    def has_parent_identity(cls):
        if cls.identifier_builder is None:
            return False
        if len(cls.identifier_builder) == 0:
            return False
        for p in cls.parents:
            if isinstance(p, One2One):
                if p.singular_name in cls.identifier_builder:
                    return True
            elif isinstance(p, Multiple):
                if p.plural_name in cls.identifier_builder:
                    return True
            elif p.singular_name in cls.identifier_builder:
                return True
        return False

    @classmethod
    def make_schema(cls) -> Optional[str]:
        name = cls.__name__
        if cls.idname is not None:
            prop = cls.idname
            return f'CREATE CONSTRAINT {name} ON (n:{name}) ASSERT (n.{prop}) IS NODE KEY'
        elif cls.identifier_builder:
            if cls.has_factor_identity():
                key = ', '.join([f'n.{f}' for f in cls.identifier_builder])
                return f'CREATE CONSTRAINT {name} ON (n:{name}) ASSERT ({key}) IS NODE KEY'
            elif cls.has_parent_identity():
                key = ', '.join([f'n.{f}' for f in cls.identifier_builder if f in cls.factors])
                if not len(key):
                    raise TypeError(f"No factors are present in the identity builder of {name} to make an index from ")
                return f'CREATE INDEX {name} FOR (n:{name}) ON ({key})'
        elif cls.indexes:
            key = ', '.join([f'n.{i}' for i in cls.indexes])
            return f'CREATE INDEX {name} FOR (n:{name}) ON ({key})'
        elif cls.is_template:
            return None
        else:
            raise RuleBreakingException(f"A hierarchy must define an idname, identifier_builder, or index, "
                                        f"unless it is marked as template class for something else (`is_template=True`)")

    @classmethod
    def merge_strategy(cls):
        if cls.idname is not None:
            return 'NODE FIRST'
        elif cls.identifier_builder:
            if cls.has_factor_identity():
                return 'NODE FIRST'
            elif cls.has_parent_identity():
                return 'NODE+RELATIONSHIP'
        return 'NODE FIRST'

    def attach_products(self, file=None, index=None, **hdus):
        """attaches products to a hierarchy with relations like: <-[:PRODUCT {index: rowindex, name: 'flux'}]-"""
        collision_manager = CypherQuery.get_context().collision_manager
        for productname, name in self.products.items():
            props = {}
            if isinstance(name, Indexed):
                if name.column_name is not None:
                    props['column_name'] = name.column_name
                name = name.name
                if index is None:
                    raise IndexError(f"{self} requires an index for {file} product {name}")
                props['index'] = index
            props['name'] = productname
            hdu = hdus[name]
            merge_relationship(hdu, self, 'product', props, {}, collision_manager=collision_manager)
        if file is not None:
            merge_relationship(file, self, 'is_required_by', {'name': 'file'}, {}, collision_manager=collision_manager)

    @classmethod
    def without_creation(cls, **kwargs):
        return cls(do_not_create=True, **kwargs)

    @classmethod
    def find(cls, anonymous_children=None, anonymous_parents=None, **kwargs):
        parent_names = [i.name if isinstance(i, Multiple) else i.singular_name for i in cls.parents]
        parents = [] if anonymous_parents is None else anonymous_parents
        anonymous_children = [] if anonymous_children is None else anonymous_children
        factors = {}
        for k, v in kwargs.items():
            if k in cls.factors:
                factors[k] = v
            elif k in parent_names:
                if not isinstance(v, list):
                    v = [v]
                for vi in v:
                    parents.append(vi)
            elif k == cls.idname:
                factors[k] = v
            else:
                raise ValueError(f"Unknown name {k} for {cls}")
        node = match_pattern_node(labels=cls.neotypes, properties=factors,
                                  parents=parents, children=anonymous_children)
        obj = cls.without_creation(**kwargs)
        obj.node = node
        return obj

    def __repr__(self):
        i = ''
        if self.idname is not None:
            i = f'{self.identifier}'
        return f"<{self.__class__.__name__}({self.idname}={i})>"


class Hierarchy(Graphable):
    parents = []
    factors = []
    is_template = True

    def make_specification(self) -> Tuple[Dict[str, Type[Graphable]], Dict[str, str]]:
        """
        Make a dictionary of {name: HierarchyClass} and a similar dictionary of factors
        """
        parents = {p.singular_name if isinstance(p, (type, One2One)) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        return specification, factors

    def __init__(self, do_not_create=False, tables=None, **kwargs):
        self.uses_tables = False
        if tables is None:
            for value in kwargs.values():
                if isinstance(value, Unwind):
                    self.uses_tables = True
                elif isinstance(value, Hierarchy):
                    self.uses_tables = value.uses_tables
        else:
            self.uses_tables = True
        self.identifier = kwargs.pop(self.idname, None)
        self.specification, factors = self.make_specification()
        # add any data held in a neo4j unwind table
        for k, v in self.specification.items():
            if k not in kwargs:
                if tables is not None:
                    kwargs[k] = tables.get(k, alias=False)
        self._kwargs = kwargs.copy()
        # Make predecessors a dict of {name: [instances of required Factor/Hierarchy]}
        predecessors = {}
        for name, nodetype in self.specification.items():
            if do_not_create:
                value = kwargs.pop(name, None)
            else:
                value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple) and not isinstance(nodetype, One2One):
                if not isinstance(value, (tuple, list)):
                    if isinstance(value, Graphable):
                        if not getattr(value, 'uses_tables', False):
                            raise TypeError(f"{name} expects multiple elements")
            else:
                value = [value]
            if name not in factors:
                predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")
        self.predecessors = predecessors
        if self.identifier_builder is not None:
            if self.identifier is not None:
                raise RuleBreakingException(f"{self} must not take an identifier if it has an identifier_builder")
        if self.idname is not None:
            if not do_not_create and self.identifier is None:
                raise ValueError(f"Cannot assign an id of None to {self}")
            setattr(self, self.idname, self.identifier)
        super(Hierarchy, self).__init__(do_not_create, **predecessors)
