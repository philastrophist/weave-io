# Copied the Context structure from Pymc3

import threading
from collections import defaultdict
from contextlib import contextmanager
from sys import modules
from typing import Optional, Type, TypeVar, List, Union, Dict

import py2neo
from py2neo import Graph as NeoGraph


T = TypeVar("T", bound="ContextMeta")


class ContextError(Exception):
    pass


class ContextMeta(type):
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """

    def __new__(cls, name, bases, dct, **kargs):  # pylint: disable=unused-argument
        "Add __enter__ and __exit__ methods to the class."

        def __enter__(self):
            self.__class__.context_class.get_contexts().append(self)
            return self

        def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
            self.__class__.context_class.get_contexts().pop()

        dct[__enter__.__name__] = __enter__
        dct[__exit__.__name__] = __exit__

        # We strip off keyword args, per the warning from
        # StackExchange:
        # DO NOT send "**kargs" to "type.__new__".  It won't catch them and
        # you'll get a "TypeError: type() takes 1 or 3 arguments" exception.
        return super().__new__(cls, name, bases, dct)

    # FIXME: is there a more elegant way to automatically add methods to the class that
    # are instance methods instead of class methods?
    def __init__(
        cls, name, bases, nmspc, context_class: Optional[Type] = None, **kwargs
    ):  # pylint: disable=unused-argument
        """Add ``__enter__`` and ``__exit__`` methods to the new class automatically."""
        if context_class is not None:
            cls._context_class = context_class
        super().__init__(name, bases, nmspc)

    def get_context(cls, error_if_none=True) -> Optional['Graph']:
        """Return the most recently pushed context object of type ``cls``
        on the stack, or ``None``. If ``error_if_none`` is True (default),
        raise a ``ContextError`` instead of returning ``None``."""
        idx = -1
        while True:
            try:
                candidate = cls.get_contexts()[idx]  # type: Optional[T]
            except IndexError as e:
                # Calling code expects to get a TypeError if the entity
                # is unfound, and there's too much to fix.
                if error_if_none:
                    raise ContextError("No %s on context stack" % str(cls))
                return None
            return candidate


    def get_contexts(cls) -> List[T]:
        """Return a stack of context instances for the ``context_class``
        of ``cls``."""
        # This lazily creates the context class's contexts
        # thread-local object, as needed. This seems inelegant to me,
        # but since the context class is not guaranteed to exist when
        # the metaclass is being instantiated, I couldn't figure out a
        # better way. [2019/10/11:rpg]

        # no race-condition here, contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        context_class = cls.context_class
        assert isinstance(context_class, type), (
            "Name of context class, %s was not resolvable to a class" % context_class
        )
        if not hasattr(context_class, "contexts"):
            context_class.contexts = threading.local()

        contexts = context_class.contexts

        if not hasattr(contexts, "stack"):
            contexts.stack = []
        return contexts.stack

    # the following complex property accessor is necessary because the
    # context_class may not have been created at the point it is
    # specified, so the context_class may be a class *name* rather
    # than a class.
    @property
    def context_class(cls) -> Type:
        def resolve_type(c: Union[Type, str]) -> Type:
            if isinstance(c, str):
                c = getattr(modules[cls.__module__], c)
            if isinstance(c, type):
                return c
            raise ValueError("Cannot resolve context class %s" % c)

        assert cls is not None
        if isinstance(cls._context_class, str):
            cls._context_class = resolve_type(cls._context_class)
        if not isinstance(cls._context_class, (str, type)):
            raise ValueError(
                "Context class for %s, %s, is not of the right type"
                % (cls.__name__, cls._context_class)
            )
        return cls._context_class

    # Inherit context class from parent
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.context_class = super().context_class

    # Initialize object in its own context...
    # Merged from InitContextMeta in the original.
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        with instance:  # appends context
            instance.__init__(*args, **kwargs)
        return instance


class Graph(metaclass=ContextMeta):
    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get("graph") is not None:
            instance._parent = kwargs.get("graph")
        else:
            instance._parent = cls.get_context(error_if_none=False)
        return instance

    def __init__(self, profile=None, name=None, write=False, **settings):
        self.neograph = NeoGraph(profile, name, **settings)
        if write:
            self.query = CypherQuery(self)

    def upload(self):
        q = self.query.render_query()
        self.neograph.run(q, parameters={d.name: d for d in self.query.data})

    def create_unique_constraint(self, label, key):
        try:
            self.neograph.schema.create_uniqueness_constraint(label, key)
        except py2neo.database.work.ClientError:
            pass

    def drop_unique_constraint(self, label, key):
        try:
            self.neograph.schema.drop_uniqueness_constraint(label, key)
        except py2neo.database.work.DatabaseError:
            pass


Graph._context_class = Graph


class CypherQuery:
    def __init__(self, graph):
        self.graph = graph
        self.statements = []
        self.open_contexts = [set()]
        self.data = []
        self.timestamp = TimeStamp()
        self.add_statement(self.timestamp)

    @property
    def current_context(self):
        return self.open_contexts[-1]

    @property
    def accessible_variables(self):
        return [v for context in self.current_context[::-1] for v in context]

    def add_statement(self, statement):
        for v in statement.input_variables:
            if v not in self.accessible_variables:
                raise ValueError(f"{v} is not accessible is this context. Have you left a WITH context?")
        self.statements.append(statement)
        self.current_context.update(statement.output_variables)

    def make_variable_names(self):
        d = defaultdict(int)
        for statement in self.statements:
            for v in statement.input_variables + statement.output_variables:
                if v.name is None:
                    i = d[v.namehint]
                    d[v.namehint] += 1
                    v.name = f'{v.namehint}{i}'
        for data in self.data:
            i = d[data.namehint]
            d[data.namehint] += 1
            data.name = f'{data.namehint}{i}'

    def render_query(self):
        self.make_variable_names()
        return '\n'.join([s.to_cypher() for s in self.statements])

    def open_context(self):
        self.open_contexts.append(set())

    def close_context(self):
        del self.open_contexts[-1]

    def add_data(self, data):
        self.data.append(data)


class Statement:
    """A cypher statement that takes inputs and returns outputs"""
    def __init__(self, input_variables, output_variables):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.timestamp = Graph.get_context().query.timestamp

    def to_cypher(self):
        raise NotImplementedError


class TimeStamp(Statement):
    def __init__(self):
        super(TimeStamp, self).__init__([], CypherVariable('time'))

    def to_cypher(self):
        return f'WITH *, timestamp() as {self.output_variables[0]}'


class CypherVariable:
    def __init__(self, namehint=None):
        self.namehint = namehint
        self._name = None

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self.name

    def __getattr__(self, item):
        return CypherVariableAttribute(self, item)

    def __getitem__(self, item):
        return CypherVariableAttribute(self, item)


class DerivedCypherVariable(CypherVariable):
    def __init__(self, parent, args):
        super(DerivedCypherVariable, self).__init__()
        self.parent = parent
        self.args = args

    def string_formatter(self, parent, args):
        raise NotImplementedError

    @property
    def name(self):
        return self.string_formatter(self.parent, self.args)


class CypherVariableItem(DerivedCypherVariable):
    def string_formatter(self, parent, attr):
        return f"{parent}[{attr}]"


class CypherVariableAttribute(DerivedCypherVariable):
    def string_formatter(self, parent, attr):
        return f"{parent}.{attr}"


class Collection(CypherVariable):
    pass


class CypherData(CypherVariable):
    def __init__(self, data, name='data'):
        super(CypherData, self).__init__(name)
        self.data = data
        query = Graph.get_context().query  # type: CypherQuery
        query.add_data(self)

    def __repr__(self):
        return '$' + super(CypherData, self).__repr__()


class Unwind(Statement):
    def __init__(self, *args: CypherVariable):
        output_variables = [CypherVariable('unwound_'+a.namehint) for a in args]
        if len(self.input_variables) > 1:
            self.indexer_variable = CypherVariable('i')
            output_variables.append(self.indexer_variable)
        super(Unwind, self).__init__(args, output_variables)

    def to_cypher(self):
        if len(self.input_variables) == 1:
            return f"UNWIND {self.input_variables[0]} as {self.output_variables[0]}"
        else:
            outaliasstr = 'WITH *'
            for n, (i, o) in enumerate(zip(self.input_variables, self.output_variables[:-1])):
                outaliasstr += f', {i}[{self.indexer_variable}] as {o}'
            inliststr = f'[{",".join(self.input_variables)}]'
            return f"WITH *, apoc.coll.max([x in {inliststr} | SIZE(x)]) as m; UNWIND range(0, m) as {self.indexer_variable}; {outaliasstr}"


class Collect(Statement):
    def __init__(self, *args):
        super(Collect, self).__init__(args, [Collection(a.namehint+'s') for a in args])

    def to_cypher(self):
        return f"WITH " + ','.join([f'collect({a}) as {b}' for a, b in zip(self.input_variables, self.output_variables)])

    def __getitem__(self, inarg):
        i = self.input_variables.index(inarg)
        return self.output_variables[i]


class GroupBy(Statement):
    def __init__(self, nodelist, propertyname):
        super(GroupBy, self).__init__([nodelist], [CypherVariable(propertyname+'_dict')])
        self.propertyname = propertyname

    def to_cypher(self):
        return f"WITH *, apoc.map.groupBy({self.input_variables[0]}, {self.propertyname}) as {self.output_variables[0]}"


class Varname:
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return self.key


class MergeMany2One(Statement):
    def __init__(self, parents: Dict[Union[CypherVariable, Collection], str], labels: List[str], properties: dict, versioned_labels: List[str]):
        self.parents = parents
        self.labels = labels
        self.properties = {Varname(k): v for k, v in properties.items()}
        self.versioned_labels = versioned_labels
        self.child = CypherVariable(labels[-1])
        self.speclist = [CypherVariable(p.namehint) for p in self.parents]
        self.specs = CypherVariable('specs')
        super().__init__(parents, [self.child] + self.speclist + [self.specs])
        if len(versioned_labels):
            self.version = CypherVariable('version')
            self.output_variables.append(self.version)

    def make_specs(self):
        specs = []
        for (parent, reltype), spec in zip(self.parents.items(), self.speclist):
            if isinstance(parent, Collection):
                specs.append(f"WITH *, [i in range(0, size({parent}) - 1) | [{parent}[i], {reltype}, {{order: i}}]] as {spec}")
            else:
                specs.append(f"WITH *, [{parent}, {reltype}, {{order: 0}}] as {spec}")
        return '\n'.join(specs) + f'WITH *, {"+".join(self.specs)} as {self.specs}'

    def to_cypher(self):
        merge = self.make_specs()
        merge += f"\nCALL custom.multimerge({self.specs}, {self.labels}, {self.properties}, {{dbcreated: {self.timestamp}}}, {{}}) YIELD {self.child}"
        if len(self.versioned_labels):
            merge += f"\nCALL custom.version({self.specs}, {self.child}, {self.versioned_labels}, 'version') YIELD {self.version}"
        return merge


class MatchMany2One(MergeMany2One):
    def to_cypher(self):
        match = self.make_specs()
        match += f"\nCALL custom.multimatch({self.specs}, {self.labels}, {self.properties}) YIELD {self.child}"
        return match


class MergeNode(Statement):
    keyword = 'MERGE'

    def __init__(self, labels: List[str], properties: dict):
        self.labels = [Varname(l) for l in labels]
        self.properties = {Varname(k): v for k, v in properties.items()}
        self.out = CypherVariable()
        super(MergeNode, self).__init__([], [self.out])

    def to_cypher(self):
        labels = ':'.join(map(str, self.labels))
        return f"{self.keyword} ({self.out}:{labels} {{{self.properties}}})"


class MatchNode(MergeNode):
    keyword = 'MATCH'


@contextmanager
def unwind(*args):
    query = Graph.get_context().query  # type: CypherQuery
    unwinder = Unwind(*args)
    query.open_context()  # allow this context to accumulate variables
    query.add_statement(unwinder)  # append the actual statement
    yield unwinder.output_variables  # give back the unwound variables
    context_variables = query.current_context
    query.close_context()  # remove the variables from being referenced from now on
    query.statements.append(Collect(*context_variables))  # allow the collections to be accessible - force


def collect(*variables: CypherVariable):
    query = Graph.get_context().query  # type: CypherQuery
    collector = query.statements[-1]
    if not isinstance(collector, Collect):
        raise NameError(f"You must use collect straight after a with context")
    return [collector[variable] for variable in variables]


def groupby(variable_list, propertyname):
    if not isinstance(variable_list, Collection):
        raise TypeError(f"{variable_list} is not a collection")
    query = Graph.get_context().query  # type: CypherQuery
    g = GroupBy(variable_list, propertyname)
    query.add_statement(g)
    return g.output_variables[0]
