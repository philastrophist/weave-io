import inspect
import logging
from copy import deepcopy
from functools import wraps, partial
from typing import Tuple, Dict, Type, Union, List, Optional as _Optional
from warnings import warn

from . import writequery
from .writequery import CypherQuery, Unwind, Collection, CypherVariable
from .context import ContextError
from .utilities import Varname, make_plural, int_or_none


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


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


class Multiple:
    def __init__(self, node, minnumber=1, maxnumber=None, constrain=None, idname=None):
        self.node = node
        self.minnumber = int_or_none(minnumber) or 0
        self.maxnumber = int_or_none(maxnumber)
        self.constrain = [] if constrain is None else constrain
        self.relation_idname = idname
        self._isself = self.node == 'self'
        if inspect.isclass(self.node):
            if issubclass(self.node, Hierarchy):
                self.instantate_node()

    @property
    def name(self):
        return self.plural_name

    def instantate_node(self, include_hierarchies=None):
        if not inspect.isclass(self.node):
            if isinstance(self.node, str):
                hierarchies = {i.__name__: i for i in all_subclasses(Hierarchy)}
                if include_hierarchies is not None:
                    for h in include_hierarchies:
                        hierarchies[h.__name__] = h  # this overrides the default
                self.node = hierarchies[self.node]
        self.singular_name = self.node.singular_name
        self.plural_name = self.node.plural_name
        self.idname = self.node.idname
        try:
            self.factors =  self.node.factors
        except AttributeError:
            self.factors = []
        try:
            self.parents = self.node.parents
        except AttributeError:
            self.parents = []

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}] id={self.relation_idname})>"

    def __hash__(self):
        if isinstance(self.node, str):
            hsh = hash(self.node)
        else:
            hsh = hash(self.node.__name__)
        return hash(self.__class__) ^ hash(self.minnumber) ^ hash(self.maxnumber) ^\
        hash(''.join(self.constrain)) ^ hash(self.relation_idname) ^ hsh

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def from_names(cls, hierarchy: Type['Hierarchy'], *singles: str, **multiples: Union[int, Tuple[_Optional[int], _Optional[int]]]) -> List['Multiple']:
        single_list = [One2One(hierarchy, idname=name) for name in singles]
        multiple_list = [Multiple(hierarchy, i, i) if isinstance(i, int) else Multiple(cls, *i) for k, i in multiples.items()]
        return single_list + multiple_list


class Single(Multiple):
    def __init__(self, node, constrain=None, idname=None):
        super().__init__(node, 1, 1, constrain, idname)

    def __repr__(self):
        return f"<Single({self.node})>"

    @property
    def name(self):
        return self.singular_name


class One2One(Single):
    def __init__(self, node, constrain=None, idname=None):
        super(One2One, self).__init__(node, constrain, idname)

    def __repr__(self):
        return f"<One2One({self.node})>"


class Optional(Multiple):
    def __init__(self, node, constrain=None, idname=None):
        super(Optional, self).__init__(node, 0, 1, constrain, idname)

    def __repr__(self):
        return f"<Optional({self.node})>"

    @property
    def name(self):
        return self.singular_name


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
            dct['plural_name'] = make_plural(name.lower())
        dct['singular_name'] = name.lower()
        if dct['plural_name'] != dct['plural_name'].lower():
            raise RuleBreakingException(f"plural_name must be lowercase")
        if dct['singular_name'] != dct['singular_name'].lower():
            raise RuleBreakingException(f"singular_name must be lowercase")
        if dct['plural_name'] == dct['singular_name']:
            raise RuleBreakingException(f"plural_name must not be the same as singular_name")
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
        cls.children = deepcopy(cls.children)  # sever link so that changes here dont affect base classes
        cls.parents = deepcopy(cls.parents)
        for i, c in enumerate(cls.children):
            if isinstance(c, Multiple):
                if c._isself:
                    c.node = cls
                c.instantate_node()
        for i, p in enumerate(cls.parents):
            if isinstance(p, Multiple):
                if p._isself:
                    p.node = cls
                p.instantate_node()
                if isinstance(p, One2One):
                    parentnames[p.singular_name] = (1, 1)
                else:
                    parentnames[p.plural_name] = (p.minnumber, p.maxnumber)
            else:
                parentnames[p.singular_name] = (1, 1)
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
        for p in cls.version_on:
            parents = [p.node if isinstance(p, One2One) else p for p in cls.parents]
            if p in [pp.singular_name if isinstance(pp, type) else pp.name for pp in parents]:
                version_parents.append(p)
            elif p in cls.factors:
                version_factors.append(p)
            else:
                raise RuleBreakingException(f"Unknown {p} to version on for {name}. Must refer to a parent or factor.")
        if len(version_factors) > 1 and len(version_parents) == 0:
            raise RuleBreakingException(f"Cannot build a version relative to nothing. You must version on at least one parent.")
        if not cls.is_template:
            if not (len(cls.indexes) or cls.idname or
                    (cls.identifier_builder is not None and len(cls.identifier_builder) > 0)):
                raise RuleBreakingException(f"{name} must define an indexes, idname, or identifier_builder")
        for p in cls.indexes:
            if p not in cls.parents and p not in cls.factors:
                raise RuleBreakingException(f"index {p} of {name} must be a factor or parent of {name}")
        if len(cls.hdus):
            hduclasses = {}
            for p, (hduname, hdu) in enumerate(cls.hdus.items()):
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
        cls.relative_names = {}  # reset, no inheritability
        for relative in cls.children+cls.parents:
            if isinstance(relative, Multiple):
                if relative.relation_idname is not None:
                    cls.relative_names[relative.relation_idname] = relative
        super().__init__(name, bases, dct)


class Graphable(metaclass=GraphableMeta):
    idname = None
    identifier = None
    indexer = None
    type_graph_attrs = {}
    plural_name = None
    singular_name = None
    parents = []
    children = []
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
    relative_names = {}

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


    def __init__(self, predecessors, successors=None, do_not_create=False):
        if successors is None:
            successors = {}
        self.predecessors = predecessors
        self.successors = successors
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
                        props = {'order': i, 'relation_id': k}
                        merge_relationship(parent, child, type, {}, props, collision_manager=collision_manager)
                    parent_list = collect(parent)
                    if k in self.version_on:
                        raise RuleBreakingException(f"Cannot version on a collection of nodes")
                else:
                    for parent in parent_list:
                        props = {'order': 0, 'relation_id': k}
                        merge_relationship(parent, child, type, {}, props, collision_manager=collision_manager)
                        if k in self.version_on:
                            version_parents.append(parent)
        elif merge_strategy == 'NODE+RELATIONSHIP':
            parentnames = [p.singular_name if isinstance(p, (One2One, Optional)) else p.plural_name if isinstance(p, Multiple) else p.singular_name for p in self.parents]
            parents = []
            others = []
            for k, parent_list in predecessors.items():
                if isinstance(parent_list, Collection):
                    raise TypeError(f"Cannot merge NODE+RELATIONSHIP for collections")
                if k in parentnames and k in self.identifier_builder:
                    parents += [p for p in parent_list]
                else:
                    others += [(i, k, p) for i, p in enumerate(parent_list)]
                if k in self.version_on:
                    version_parents += parent_list
            reltype = 'is_required_by'
            relparents = {p: (reltype, {'order': 0}, {}) for p in parents}
            child = self.node = merge_node(self.neotypes, self.neoidentproperties, self.neoproperties,
                                           parents=relparents, collision_manager=collision_manager)
            for i, k, other in others:
                if other is not None:
                    merge_relationship(other, child, reltype, {'order': i, 'relation_id': k}, {}, collision_manager=collision_manager)
        else:
            ValueError(f"Merge strategy not known: {merge_strategy}")
        if len(version_parents):
            version_factors = {f: self.neoproperties[f] for f in self.version_on if f in self.factors}
            set_version(version_parents, ['is_required_by'] * len(version_parents), self.neotypes[-1], child, version_factors)
        # now the children
        for k, child_list in successors.items():
            type = 'is_required_by'
            if isinstance(child_list, Collection):
                with unwind(child_list, enumerated=True) as (child, i):
                    merge_relationship(child, self.node, type,
                                       {}, {'relation_id': k, 'order': i},
                                       collision_manager=collision_manager)
                collect(child)
            else:
                for child in child_list:
                    props =  {'relation_id': k, 'order': 0}
                    merge_relationship(child, self.node, type, {},
                                       props, collision_manager=collision_manager)


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
    def make_schema(cls) -> _Optional[str]:
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

    def make_specification(self) -> Tuple[Dict[str, Type[Graphable]], Dict[str, str], Dict[str, Type[Graphable]]]:
        """
        Make a dictionary of {name: HierarchyClass} and a similar dictionary of factors
        """
        # parents = {p.singular_name if isinstance(p, (type, One2One)) else p.name: p for p in self.parents}
        parents = {getattr(p, 'relation_idname', None) or getattr(p, 'name', None) or p.singular_name: p for p in self.parents}
        children = {getattr(c, 'relation_idname', None) or getattr(c, 'name', None) or c.singular_name: c for c in self.children}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        specification.update(children)
        return specification, factors, children

    @classmethod
    def instantate_nodes(cls, hierarchies=None):
        for i in cls.parents + cls.factors + cls.children:
            if isinstance(i, Multiple):
                if isinstance(i.node, str):
                    i.instantate_node(hierarchies)

    def __init__(self, do_not_create=False, tables=None, **kwargs):
        self.instantate_nodes()
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
        self.specification, factors, children = self.make_specification()
        # add any data held in a neo4j unwind table
        for k, v in self.specification.items():
            if k not in kwargs:
                if tables is not None:
                    kwargs[k] = tables.get(k, alias=False)
        self._kwargs = kwargs.copy()
        # Make predecessors a dict of {name: [instances of required Factor/Hierarchy]}
        predecessors = {}
        successors = {}
        for name, nodetype in self.specification.items():
            if isinstance(nodetype, Multiple):
                if nodetype.minnumber == 0 and name not in kwargs:
                    continue
            if do_not_create:
                value = kwargs.pop(name, None)
            else:
                value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple) and not isinstance(nodetype, (One2One, Single, Optional)):
                if nodetype.maxnumber != 1:
                    if not isinstance(value, (tuple, list)):
                        if isinstance(value, Graphable):
                            if not getattr(value, 'uses_tables', False):
                                raise TypeError(f"{name} expects multiple elements")
            else:
                value = [value]
            if name in children:
                successors[name] = value
            elif name not in factors:
                predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")
        self.predecessors = predecessors
        self.successors = successors
        if self.identifier_builder is not None:
            if self.identifier is not None:
                raise RuleBreakingException(f"{self} must not take an identifier if it has an identifier_builder")
        if self.idname is not None:
            if not do_not_create and self.identifier is None:
                raise ValueError(f"Cannot assign an id of None to {self}")
            setattr(self, self.idname, self.identifier)
        super(Hierarchy, self).__init__(predecessors, successors, do_not_create)
