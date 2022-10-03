"""
1. Create subclass: `class MyHierarchy: ...` in terms of other Hierarchies
2. Instantiate: `MyHierarchy(arg1)` with query arguments
3. Return query object: `hier = MyHierarchy(arg1)` returns a query



"""
import inspect
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from typing import Tuple, Type, Union, List, Optional as _Optional
from warnings import warn

from .readquery.parser import QueryGraph
from .utilities import int_or_none, camelcase2snakecase, make_plural


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
        for p in cls.version_on:
            if p in [pp.singular_name if isinstance(pp, type) else pp.name for pp in cls.parents+cls.children]:
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
            if p is not None:
                if p not in cls.parents and p not in cls.factors:
                    raise RuleBreakingException(f"index {p} of {name} must be a factor or parent of {name}")
        if cls.concatenation_constants is not None:
            if len(cls.concatenation_constants):
                cls.factors = cls.factors + cls.concatenation_constants + ['concatenation_constants']
        clses = [i.__name__ for i in inspect.getmro(cls)]
        clses = clses[:clses.index('Graphable')]
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
    factors = []
    indexes = []
    products = []
    identifier_builder = None
    products_and_factors = []
    idname = None
    identifier = None
    is_template = True


    def __init__(self, **kwargs):
        # for each parent/child/factor, get from kwargs, raise if missing (if not optional)
        parents = []
        for p in map(Multiple.from_argument, self.parents):
            if p.maxnumber == 1:
                if p.minnumber == 1:
                    parentskwargs[p.node.singular_name]


    def __init__(self, **kwargs):


        kwargs = {k: v for k, v in kwargs.items() if v is not None}
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
                if isinstance(v, Multiple) and (v.minnumber == 0 or v.notreal):  # i.e. optional
                    continue
                if tables is not None:
                    kwargs[k] = tables.get(tables_replace.get(k, k), alias=False)
        self._kwargs = kwargs.copy()
        # Make predecessors a dict of {name: [instances of required Factor/Hierarchy]}
        predecessors = {}
        successors = {}
        for name, nodetype in self.specification.items():
            if isinstance(nodetype, Multiple):
                if (nodetype.minnumber == 0 or nodetype.notreal) and name not in kwargs:
                    continue
            if do_not_create:
                value = kwargs.pop(name, None)
            else:
                value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple) and not isinstance(nodetype, (OneOf, Optional)):
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



    def __new__(cls, name, bases, dct):
        from .readquery.objects import ObjectQuery
        from .data import Data
        data = Data.get_context()  # type: Data
        hierarchy = super().__new__(cls, name, bases, dct).__init__()  # do super to actually make this class
        inputs = [v for v in hierarchy.__dict__.values() if isinstance(v, ObjectQuery)]
        # infer shared hierarchy from parents/children
        G = data.query._G  # type: QueryGraph
        if not inputs:
            previous = data.query
        else:
            previous = G.latest_object_node(*inputs)
        # make a ObjectQuery with shared hierarchy
        n = G.add_write(hierarchy)
        return ObjectQuery._spawn(previous, n, cls.__name__, single=True, hierarchy=hierarchy)  # insert Hierarchy into ObjectQuery




