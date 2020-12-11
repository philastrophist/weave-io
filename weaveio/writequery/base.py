from collections import defaultdict
from typing import List

import re

from ..context import ContextMeta


def camelcase(x):
    return x[0].upper() + x[1:]


class CypherQuery(metaclass=ContextMeta):
    def __init__(self):
        self.data = []
        self.statements = [TimeStamp()]  # type: List[Statement]
        self.timestamp = self.statements[0].output_variables[0]
        self.open_contexts = [[self.timestamp]]
        self.closed_context = None

    @property
    def current_context(self):
        return self.open_contexts[-1]

    @property
    def accessible_variables(self):
        return [v for context in self.open_contexts[::-1] for v in context] + [i for i in self.data]

    def is_accessible(self, variable):
        if variable in self.accessible_variables:
            return True
        if hasattr(variable, 'parent'):
            return self.is_accessible(variable.parent)
        return False

    def add_statement(self, statement):
        for v in statement.input_variables:
            if not self.is_accessible(v):
                raise ValueError(f"{v} is not accessible is this context. Have you left a WITH context?")
        self.statements.append(statement)
        self.current_context.extend(statement.output_variables)

    def make_variable_names(self):
        d = defaultdict(int)
        for data in self.data:
            namehint = data.namehint.lower()
            i = d[namehint]
            d[namehint] += 1
            data._name = f'{namehint}{i}'
        for statement in self.statements:
            for v in statement.input_variables + statement.output_variables:
                if v.name is None:
                    namehint = v.namehint.lower()
                    i = d[namehint]
                    d[namehint] += 1
                    v._name = f'{namehint}{i}'

    def render_query(self, procedure_tag=''):
        if not isinstance(self.statements[-1], Returns):
            self.returns(self.timestamp)
        self.make_variable_names()
        q = '\n'.join([s.to_cypher() for s in self.statements])
        datadict = {d.name: d.data for d in self.data}
        return re.sub(r'(custom\.[\w\d]+)\(', fr'\1-----{procedure_tag}(', q).replace('-----', ''), datadict

    def open_context(self):
        self.open_contexts.append([])

    def close_context(self):
        self.closed_context = self.open_contexts[-1]
        del self.open_contexts[-1]

    def add_data(self, data):
        self.data.append(data)

    def returns(self, *args, **kwargs):
        if not isinstance(self.statements[-1], Returns):
            self.statements.append(Returns(*args, **kwargs))
        else:
            raise ValueError(f"Cannot have more than one return")


CypherQuery._context_class = CypherQuery


class Varname:
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return self.key


class Statement:
    """A cypher statement that takes inputs and returns outputs"""
    def __init__(self, input_variables, output_variables):
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)
        self.timestamp = CypherQuery.get_context().timestamp

    def to_cypher(self):
        raise NotImplementedError


class Returns(Statement):
    def __init__(self, *unnamed_variables, **named_variables):
        self.unnamed_variables = unnamed_variables
        self.named_variables = named_variables
        self.input_variables = list(unnamed_variables) + list(named_variables.values())
        self.output_variables = []

    def to_cypher(self):
        l = list(map(str, self.unnamed_variables))
        l += [f'{v} as {k}' for k, v in self.named_variables.items()]
        return 'RETURN ' + ', '.join(l)


class TimeStamp(Statement):
    def __init__(self):
        self.input_variables = []
        self.output_variables = [CypherVariable('time')]

    def to_cypher(self):
        return f'WITH *, timestamp() as {self.output_variables[0]}'


class CypherVariable:
    def __init__(self, namehint=None):
        self.namehint = 'variable' if namehint is None else namehint
        self._name = None

    @property
    def name(self):
        return self._name

    def __repr__(self):
        if self.name is None:
            return super(CypherVariable, self).__repr__()
        return self.name

    def __getitem__(self, item):
        return CypherVariableItem(self, item)


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
        if isinstance(attr, str):
            return f"{parent}['{attr}']"
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
        query = CypherQuery.get_context()  # type: CypherQuery
        query.add_data(self)

    def __repr__(self):
        return '$' + super(CypherData, self).__repr__()