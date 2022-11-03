"""
1. Create subclass: `class MyHierarchy: ...` in terms of other Hierarchies
2. Instantiate: `MyHierarchy(arg1)` with query arguments
3. Return query object: `hier = MyHierarchy(arg1)` returns a query
"""
import inspect
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from typing import Tuple, Type, Union, List, Optional as _Optional, TYPE_CHECKING
from warnings import warn

from .utilities import int_or_none, camelcase2snakecase, make_plural

if TYPE_CHECKING:
    from .readquery.parser import QueryGraph

FORBIDDEN_LABELS = []
FORBIDDEN_PROPERTY_NAMES = []
FORBIDDEN_LABEL_PREFIXES = ['_']
FORBIDDEN_PROPERTY_PREFIXES = ['_']
FORBIDDEN_IDNAMES = ['idname']

class RuleBreakingException(Exception):
    pass


class Multiple:
    def __init__(self, node, minnumber=1, maxnumber=None, constrain=None, idname=None, one2one=False, notreal=False):
        self.node = node
        self.minnumber = int_or_none(minnumber) or 0
        if maxnumber is None:
            warn(f"maxnumber is None for {node}", RuntimeWarning)
        self.maxnumber = int_or_none(maxnumber)
        self.constrain = [] if constrain is None else (constrain, ) if not isinstance(constrain, (list, tuple)) else tuple(constrain)
        self.relation_idname = idname
        self.one2one = one2one
        self._isself = self.node == 'self'
        self.notreal = notreal
        if inspect.isclass(self.node):
            if issubclass(self.node, Hierarchy):
                self.instantate_node()

    @property
    def is_optional(self):
        return self.minnumber == 0

    @property
    def name(self):
        if self.relation_idname is not None:
            return self.relation_idname
        if self.maxnumber == 1:
            return self.singular_name
        return self.plural_name

    def instantate_node(self, include_hierarchies=None):
        if not inspect.isclass(self.node):
            if isinstance(self.node, str):
                hierarchies = {i.__name__: i for i in all_subclasses(Hierarchy)}
                if include_hierarchies is not None:
                    for h in include_hierarchies:
                        hierarchies[h.__name__] = h  # this overrides the default
                try:
                    self.node = hierarchies[self.node]
                except KeyError:
                    # require hierarchy doesnt exist yet
                    Hierarchy._waiting.append(self)
                    return
        self.singular_name = self.node.singular_name
        self.plural_name = self.node.plural_name

        try:
            self.factors =  self.node.factors
        except AttributeError:
            self.factors = []
        try:
            self.parents = self.node.parents
        except AttributeError:
            self.parents = []
        while Hierarchy._waiting:
            h = Hierarchy._waiting.pop()
            h.instantate_node(include_hierarchies)

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}] id={self.relation_idname})>"

    def __hash__(self):
        if isinstance(self.node, str):
            hsh = hash(self.node)
        else:
            hsh = hash(self.node.__name__)
        return hash(self.__class__) ^ hash(self.minnumber) ^ hash(self.maxnumber) ^\
        reduce(lambda x, y: x ^ y, map(hash, self.constrain), 0) ^ hash(self.relation_idname) ^ hsh

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def from_names(cls, hierarchy: Type['Hierarchy'], *singles: str, **multiples: Union[int, Tuple[_Optional[int], _Optional[int]]]) -> List['Multiple']:
        single_list = [OneOf(hierarchy, idname=name) for name in singles]
        multiple_list = [Multiple(hierarchy, i, i) if isinstance(i, int) else Multiple(cls, *i) for k, i in multiples.items()]
        return single_list + multiple_list

    @classmethod
    def from_argument(cls, arg) -> 'Multiple':
        if isinstance(arg, Multiple):
            return arg
        return OneOf(arg)


class OneOf(Multiple):
    def __init__(self, node, constrain=None, idname=None, one2one=False):
        super().__init__(node, 1, 1, constrain, idname, one2one)

    def __repr__(self):
        return f"<OneOf({self.node})>"

    @property
    def name(self):
        return self.singular_name


class Optional(Multiple):
    def __init__(self, node, constrain=None, idname=None, one2one=False):
        super(Optional, self).__init__(node, 0, 1, constrain, idname, one2one)

    def __repr__(self):
        return f"<Optional({self.node})>"

    @property
    def name(self):
        return self.singular_name


class HierarchyMeta(type):
    def __new__(meta, name: str, bases, _dct):
        dct = {'is_template': False}
        dct.update(_dct)
        dct['aliases'] = dct.get('aliases', [])
        dct['aliases'] += [a for base in bases for a in base.aliases]
        dct['singular_name'] = dct.get('singular_name', None) or camelcase2snakecase(name)
        dct['plural_name'] = dct.get('plural_name', None) or make_plural(dct['singular_name'])
        if dct['plural_name'] != dct['plural_name'].lower():
            raise RuleBreakingException(f"plural_name must be lowercase")
        if dct['singular_name'] != dct['singular_name'].lower():
            raise RuleBreakingException(f"singular_name must be lowercase")
        if dct['plural_name'] == dct['singular_name']:
            raise RuleBreakingException(f"plural_name must not be the same as singular_name")
        dct['name'] = dct['singular_name']
        idname = dct.get('idname', None)
        if idname in FORBIDDEN_IDNAMES:
            raise RuleBreakingException(f"You may not name an id as one of {FORBIDDEN_IDNAMES}")
        if not (isinstance(idname, str) or idname is None):
            raise RuleBreakingException(f"{name}.idname ({idname}) must be a string or None")
        if name[0] != name.capitalize()[0] or '_' in name:
            raise RuleBreakingException(f"{name} must have `CamelCaseName` style name")
        for factor in dct.get('factors', []) + ['idname'] + [dct['singular_name'], dct['plural_name']]:
            # if factor != factor.lower():
            #     raise RuleBreakingException(f"{name}.{factor} must have `lower_snake_case` style name")
            if factor in FORBIDDEN_PROPERTY_NAMES:
                raise RuleBreakingException(f"The name {factor} is not allowed for class {name}")
            if any(factor.startswith(p) for p in FORBIDDEN_PROPERTY_PREFIXES):
                raise RuleBreakingException(f"The name {factor} may not start with any of {FORBIDDEN_PROPERTY_PREFIXES} for {name}")
        # remove duplicates from the list dct['parents'] whilst maintaining its order
        if 'parents' in dct:
            dct['parents'] = list(OrderedDict.fromkeys(dct['parents']))
        if 'children' in dct:
            dct['children'] = list(OrderedDict.fromkeys(dct['children']))
        if 'factors' in dct:
            dct['factors'] = list(OrderedDict.fromkeys(dct['factors']))
        if 'produces' in dct:
            dct['produces'] = list(OrderedDict.fromkeys(dct['produces']))
        r = super(HierarchyMeta, meta).__new__(meta, name, bases, dct)
        return r

    def __init__(cls, name, bases, dct):
        if cls.idname is not None and cls.identifier_builder is not None:
            raise RuleBreakingException(f"You cannot define a separate idname and an identifier_builder at the same time for {name}")
        if cls.indexes and (cls.idname is not None or cls.identifier_builder is not None):
            raise RuleBreakingException(f"You cannot define an index and an id at the same time for {name}")
        parentnames = {}
        cls.children = deepcopy(cls.children)  # sever link so that changes here dont affect base classes
        cls.parents = deepcopy(cls.parents)
        for i, c in enumerate(cls.children):
            if isinstance(c, Multiple):
                if c._isself:
                    c.node = cls
                c.instantate_node()
                for n in c.constrain:
                    if n not in cls.children:
                        cls.children.append(n)
                if c.maxnumber == 1:
                    parentnames[c.singular_name] = (c.minnumber, c.maxnumber)
                else:
                    parentnames[c.plural_name] = (c.minnumber, c.maxnumber)
            else:
                parentnames[c.singular_name] = (1, 1)
        for i, p in enumerate(cls.parents):
            if isinstance(p, Multiple):
                if p._isself:
                    p.node = cls
                p.instantate_node()
                for n in p.constrain:
                    if n not in cls.parents:
                        cls.parents.append(n)
                if p.maxnumber == 1:
                    parentnames[p.singular_name] = (p.minnumber, p.maxnumber)
                else:
                    parentnames[p.plural_name] = (p.minnumber, p.maxnumber)
            else:
                parentnames[p.singular_name] = (1, 1)
        if cls.identifier_builder is not None:
            for p in cls.identifier_builder:
                if isinstance(p, type):
                    if issubclass(p, Hierarchy):
                        p = p.singular_name
                if p in parentnames:
                    mn, mx = parentnames[p]
                elif p in cls.factors:
                    pass
                else:
                    raise RuleBreakingException(f"Unknown identifier source {p} for {name}. "
                                                f"Available are: {list(parentnames.keys())+cls.factors}")
        version_parents = []
        version_factors = []
        if len(version_factors) > 1 and len(version_parents) == 0:
            raise RuleBreakingException(f"Cannot build a version relative to nothing. You must version on at least one parent.")
        if not cls.is_template:
            if not (len(cls.indexes) or cls.idname or
                    (cls.identifier_builder is not None and len(cls.identifier_builder) > 0)):
                raise RuleBreakingException(f"{name} must define an indexes, idname, or identifier_builder")
        for p in cls.indexes:
            if p is not None:
                if p not in cls.parents and p not in cls.factors:
                    raise RuleBreakingException(f"index {p} of {name} must be a factor or parent of {name}")
        clses = [i.__name__ for i in inspect.getmro(cls)]
        clses = clses[:clses.index('object')]
        cls.neotypes = clses
        cls.products_and_factors = cls.factors + cls.products
        if cls.idname is not None:
            cls.products_and_factors.append(cls.idname)
        cls.relative_names = {}  # reset, no inheritability
        for relative in cls.children+cls.parents:
            if isinstance(relative, Multiple):
                if relative.relation_idname is not None:
                    cls.relative_names[relative.relation_idname] = relative

        super().__init__(name, bases, dct)


def make_specification(hierarchy: 'Hierarchy'):
    spec = []
    for p in hierarchy.parents:
        if isinstance(p, Hierarchy):
            spec.append(p.singular_name)
            continue
        elif isinstance(p, Multiple):
            if p.maxnumber == 1:
                spec.append(p.singular_name)
                continue
        else:
            raise TypeError(f"Inputs can only be instances of Multiple or Hierarchy")



class Hierarchy(metaclass=HierarchyMeta):
    # notreal
    # multiple or list
    # optionals
    # idname
    plural_name = None
    singular_name = None
    parents = []
    children = []
    produces = []
    factors = []
    indexes = []
    products = []
    identifier_builder = None
    products_and_factors = []
    idname = None
    identifier = None
    is_template = True
    _waiting = []

    @classmethod
    def as_factors(cls, *names, prefix=''):
        if len(names) == 1 and isinstance(names[0], list):
            names = prefix+names[0]
        if cls.parents+cls.children:
            raise TypeError(f"Cannot use {cls} as factors {names} since it has defined parents and children")
        return [f"{prefix}{name}_{factor}" if factor != 'value' else f"{prefix}{name}" for name in names for factor in cls.factors]

    @property
    def inputs(self):
        from weaveio.readquery.objects import ObjectQuery
        inputs = []
        _inputs = (*self.parents.values(), *self.children.values(), *self.products_and_factors)
        if not isinstance(self.identifier, list) and self.identifier is not None:
            _inputs = (*_inputs, self.identifier)
        for i in _inputs:
            if isinstance(i, (list, tuple)):
                inputs += [ii for ii in i if isinstance(i, ObjectQuery)]
            elif isinstance(i, ObjectQuery):
                inputs.append(i)
        return inputs

    @classmethod
    def has_factor_identity(cls):
        if cls.identifier_builder is None:
            return False
        if len(cls.identifier_builder) == 0:
            return False
        return not any(n.name in cls.identifier_builder for n in cls.parents+cls.children)

    @classmethod
    def rel_identity(cls):
        if cls.identifier_builder is None:
            return False
        if len(cls.identifier_builder) == 0:
            return False
        return sum(n.name in cls.identifier_builder for n in cls.parents + cls.children)

    @classmethod
    def has_simple_rel_identity(cls):
        return 0 < cls.rel_identity() < 3

    @classmethod
    def has_advanced_rel_identity(cls):
        return cls.rel_identity() > 2

    @classmethod
    def make_schema(cls) -> List[str]:
        name = cls.__name__
        indexes = []
        nonunique = False
        if cls.is_template:
            return []
        elif cls.idname is not None:
            prop = cls.idname
            indexes.append(f'CREATE CONSTRAINT {name}_id ON (n:{name}) ASSERT (n.{prop}) IS NODE KEY')
        elif cls.identifier_builder:
            if cls.has_factor_identity():  # only of factors
                key = ', '.join([f'n.{f}' for f in cls.identifier_builder])
                indexes.append(f'CREATE CONSTRAINT {name}_id ON (n:{name}) ASSERT ({key}) IS NODE KEY')
            elif cls.has_rel_identity():  # based on rels from parents/children
                # create 1 index on id factors and 1 index per factor as well
                key = ', '.join([f'n.{f}' for f in cls.identifier_builder if f in cls.factors])
                if key:  # joint index
                    indexes.append(f'CREATE INDEX {name}_rel FOR (n:{name}) ON ({key})')
                # separate indexes
                indexes += [f'CREATE INDEX {name}_{f} FOR (n:{name}) ON (n.{f})' for f in cls.identifier_builder if f in cls.factors]
        else:
            nonunique = True
        if cls.indexes:
            id = cls.identifier_builder or []
            indexes += [f'CREATE INDEX {name}_{i} FOR (n:{name}) ON (n.{i})' for i in cls.indexes if i not in id]
        if not indexes and nonunique:
            raise RuleBreakingException(f"{name} must define an idname, identifier_builder, or indexes, "
                                        f"unless it is marked as template class for something else (`is_template=True`)")
        return indexes

    @classmethod
    def merge_strategy(cls):
        if cls.idname is not None:
            return 'NODE FIRST'
        elif cls.identifier_builder:
            if cls.has_factor_identity():
                return 'NODE FIRST'
            elif cls.has_simple_rel_identity():
                return 'NODE+RELATIONSHIP_SIMPLE'
            elif cls.has_advanced_rel_identity():
                return 'NODE+RELATIONSHIP_ADVANCED'
        return 'NODE FIRST'

    def __init__(self, **kwargs):
        """
        validate kwargs and set parents, children, factors, identifier, etc
        each input could be a CollectionQuery/list[ObjectQuery]/ObjectQuery
        """
        from .readquery.objects import ObjectCollectionQuery, ObjectQuery
        parents = {}
        for p in self.parents:
            p = Multiple.from_argument(p)
            if p.notreal:
                continue
            v = kwargs.get(p.name)
            if not p.is_optional and v is None:
                raise KeyError(f"{self} requires input {p.name}. Argument not found.")
            if isinstance(v, (list, tuple)):
                if not all(isinstance(i, p.node) for i in v):
                    raise TypeError(f"Argument `{p.name}` of {self} is not a list/collection of instances of {p.node}.")
            if isinstance(v, ObjectCollectionQuery):
                if v._obj != p.node.__name__:
                    raise TypeError(f"Argument `{p.name}` of {self} is not a list/collection of instances of {p.node}.")
            elif isinstance(v, ObjectQuery):
                if v._obj != p.node.__name__:
                    raise TypeError(f"Argument `{p.name}` of {self} is not an instance of {p.node}.")
            parents[p] = v
            setattr(self, p.name, v)
        children = {}
        for c in self.children:
            c = Multiple.from_argument(c)
            if c.notreal:
                continue
            v = kwargs.get(c.name)
            if not c.is_optional and v is None:
                raise KeyError(f"{self} requires input {c.name}. Argument not found.")
            if isinstance(v, ()):
                if not all(isinstance(i, c.node) for i in v):
                    raise TypeError(f"Argument `{c.name}` of {self} is not a list/collection of instances of {c.node}.")
            if isinstance(v, ObjectCollectionQuery):
                if v._obj != c.node.__name__:
                    raise TypeError(f"Argument `{c.name}` of {self} is not a list/collection of instances of {c.node}.")
            elif isinstance(v, ObjectQuery):
                if v._obj != c.node.__name__:
                    raise TypeError(f"Argument `{c.name}` of {self} is not an instance of {c.node}.")
            children[c] = v
            setattr(self, c.name, v)
        factors = {}
        for f in self.factors:
            try:
                v = kwargs.get(f)
            except KeyError:
                raise KeyError(f"{self} requires input {f}. Argument not found.")
            factors[f] = v
            setattr(self, f, v)
        products = {}
        for p in self.products:
            try:
                v = kwargs.get(p.name)
            except KeyError:
                raise KeyError(f"{self} requires input {p.name}. Argument not found.")
            products[p] = v
            setattr(self, p.name, v)
        if self.idname is not None:
            identifier = kwargs[self.idname]
        elif self.identifier_builder is not None:
            identifier = [kwargs[i] for i in self.identifier_builder]
        else:
            identifier = []

        self.factors = factors
        self.products = products
        self.parents = parents
        self.children = children
        self.identifier = identifier
        self.products_and_factors = {**factors, **products}


    def __new__(cls, **kwargs):
        from .readquery.objects import ObjectQuery
        from .data import Writer
        writer = Writer.get_context()  # type: Writer
        data = writer.data
        hierarchy = object.__new__(cls)
        cls.__init__(hierarchy, **kwargs)  # do super to actually make this class
        inputs = hierarchy.inputs
        # infer shared hierarchy from parents/children
        G = data.query._G  # type: QueryGraph
        if not inputs:
            previous = data.query
        elif len(inputs) == 1:
            previous = inputs[0]
        else:
            previous = G.latest_object_node(*inputs)
        n = G.add_write(hierarchy, previous, inputs)
        # make a ObjectQuery with shared hierarchy
        return ObjectQuery._spawn(previous, n, cls.__name__, single=True, hierarchy=hierarchy)  # insert Hierarchy into ObjectQuery




