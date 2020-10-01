# Copied the Context structure from Pymc3

import threading
from collections import defaultdict
from sys import modules
from typing import Optional, Type, TypeVar, List, Union, Any, Dict

from py2neo import Graph as NeoGraph, Node, Relationship, Transaction

from weaveio.utilities import quote, Varname
import pandas as pd
import numpy as np

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

    def get_context(cls, error_if_none=True) -> Optional[T]:
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


def graphcontext(graph: Optional["Graph"]) -> "Graph":
    """
    Return the given graph or, if none was supplied, try to find one in
    the context stack.
    """
    if graph is None:
        graph = Graph.get_context(error_if_none=False)
        if graph is None:
            raise TypeError("No graph on context stack.")
    return graph


class TransactionWrapper:
    def __init__(self, tx: Transaction):
        self.tx = tx

    def __enter__(self):
        return self.tx

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tx.commit()


class Unwind:
    def __init__(self, data, name, columns=None, _renames=None):
        self.data = data
        self.name = name
        self.columns = self.data.columns if columns is None else columns
        self._renames = {} if _renames is None else _renames

    def rename(self, **names):
        renames = self._renames.copy()
        renames.update(names)
        return Unwind(self.data, self.name, self.columns, renames)

    def to_dict(self):
        columns = {c.lower(): c for c in self.columns}
        for a, b in self._renames.items():
            columns[b.lower()] = columns[a.lower()]
            del columns[a.lower()]
        return {varname: Varname(f'{self.name}.{colname}') for varname, colname in columns.items()}

    def __getitem__(self, item):
        if not isinstance(item, (list, tuple, np.ndarray)):
            item = [item]
        return Unwind(self.data[item], self.name, item)


class Graph(metaclass=ContextMeta):
    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get("graph") is not None:
            instance._parent = kwargs.get("graph")
        else:
            instance._parent = cls.get_context(error_if_none=False)
        return instance

    def __init__(self, profile=None, name=None, **settings):
        self.neograph = NeoGraph(profile, name, **settings)

    def begin(self, **kwargs):
        self.tx = self.neograph.begin(**kwargs)
        self.simples = []
        self.unwinds = []
        self.data = {}
        self.counter = defaultdict(int)
        return self.tx

    def string_properties(self, properties):
        return ', '.join(f'{k}: {quote(v)}' for k, v in properties.items())

    def string_labels(self, labels):
        return ':'.join(labels)

    def add_node(self, *labels, **properties):
        table_properties_list = properties.pop('tables', [])
        if not isinstance(table_properties_list, (tuple, list)):
            table_properties_list = [table_properties_list]
        table_properties = {k: v for p in table_properties_list for k, v in p.to_dict().items()}
        properties.update(table_properties)
        n = self.counter[labels[-1]]
        key = f"{labels[-1]}{n}".lower()
        data = self.string_properties(properties)
        labels = self.string_labels(labels)
        self.simples.append(f'MERGE ({key}:{labels} {{{data}}})')
        return key

    def add_relationship(self, a, b, *labels, **properties):
        data = self.string_properties(properties)
        labels = self.string_labels([l.upper() for l in labels])
        self.simples.append(f'MERGE ({a})-[:{labels} {{{data}}}]->({b})')

    def add_table_as_name(self, table: pd.DataFrame, name: str):
        assert name not in self.counter, "name for the table must not have been used"
        self.unwinds.append(f"UNWIND ${name}s as {name}")
        self.data[name+'s'] = list(table.T.to_dict().values())
        return Unwind(table, name)

    def make_statement(self):
        unwinds = '\n'.join(self.unwinds)
        simples = '\n'.join(self.simples)
        return f'{unwinds}\n{simples}'

    def evaluate_statement(self):
        statement = self.make_statement()
        self.tx.evaluate(statement, **self.data)

    def commit(self):
        if len(self.simples) or len(self.data):
            self.evaluate_statement()
        return self.tx.commit()

Graph._context_class = Graph