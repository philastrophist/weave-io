from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Union, List

import re

from weaveio.context import ContextMeta

def camelcase(x):
    return x[0].upper() + x[1:]


class CypherQuery(metaclass=ContextMeta):
    def __init__(self):
        self.data = []
        self.statements = [TimeStamp()]
        self.timestamp = self.statements[0].output_variables[0]
        self.open_contexts = [[self.timestamp]]

    @property
    def current_context(self):
        return self.open_contexts[-1]

    @property
    def accessible_variables(self):
        return [v for context in self.open_contexts[::-1] for v in context] + [i for i in self.data]

    def add_statement(self, statement):
        for v in statement.input_variables:
            if v not in self.accessible_variables:
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

    def returns(self, *args):
        if not isinstance(self.statements[-1], Returns):
            self.statements.append(Returns(*args))
        else:
            raise ValueError(f"Cannot have more than one return")


CypherQuery._context_class = CypherQuery


class Statement:
    """A cypher statement that takes inputs and returns outputs"""
    def __init__(self, input_variables, output_variables):
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)
        self.timestamp = CypherQuery.get_context().timestamp

    def to_cypher(self):
        raise NotImplementedError


class Returns(Statement):
    def __init__(self, *input_variables):
        self.input_variables = list(input_variables)
        self.output_variables = []

    def to_cypher(self):
        return 'RETURN ' + ', '.join(map(str, self.input_variables))


class TimeStamp(Statement):
    def __init__(self):
        self.input_variables = []
        self.output_variables = [CypherVariable('time')]

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
        if self.name is None:
            return super(CypherVariable, self).__repr__()
        return self.name

    def __getattr__(self, item):
        return CypherVariableAttribute(self, item)

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


class Unwind(Statement):
    def __init__(self, *args: CypherVariable, enumerated=False):
        output_variables = [CypherVariable('unwound_'+a.namehint.replace('$', "")) for a in args]
        super(Unwind, self).__init__(args, output_variables)
        self.passed_outputs = self.output_variables.copy()
        if enumerated or len(self.input_variables) > 1:
            self.indexer_variable = CypherVariable('i')
            self.output_variables.append(self.indexer_variable)
            if enumerated:
                self.passed_outputs.append(self.indexer_variable)
        self.enumerated = enumerated

    def to_cypher(self):
        if len(self.input_variables) == 1 and not self.enumerated:
            return f"WITH * UNWIND {self.input_variables[0]} as {self.output_variables[0]}"
        else:
            outaliasstr = 'WITH *'
            for n, (i, o) in enumerate(zip(self.input_variables, self.output_variables[:-1])):
                outaliasstr += f', {i}[{self.indexer_variable}] as {o}'
            inliststr = f'[{",".join(map(str, self.input_variables))}]'
            return f"WITH *, apoc.coll.max([x in {inliststr} | SIZE(x)])-1 as m\n" \
                   f"UNWIND range(0, m) as {self.indexer_variable} {outaliasstr}"


class Collect(Statement):
    def __init__(self, previous, *args):
        self.previous = previous
        super(Collect, self).__init__(args, [Collection(a.namehint+'s') for a in args])

    def to_cypher(self):
        collections =  ','.join([f'collect({a}) as {b}' for a, b in zip(self.input_variables, self.output_variables)])
        r = f"WITH " + ', '.join(map(str, self.previous))
        if len(collections):
            r += ', ' + collections
        return r

    def __getitem__(self, inarg):
        i = self.input_variables.index(inarg)
        return self.output_variables[i]

    def __delitem__(self, key):
        i = self.input_variables.index(key)
        del self.input_variables[i]


class GroupBy(Statement):
    def __init__(self, nodelist, propertyname):
        super(GroupBy, self).__init__([nodelist], [CypherVariable(propertyname+'_dict')])
        self.propertyname = propertyname

    def to_cypher(self):
        return f"WITH *, apoc.map.groupBy({self.input_variables[0]}, '{self.propertyname}') as {self.output_variables[0]}"


class Varname:
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return self.key


class MergeMany2One(Statement):
    def __init__(self, parents: Dict[Union[CypherVariable, Collection], str], labels: List[str], properties: dict, versioned_labels: List[str]):
        self.parents = parents
        self.labels = [camelcase(l) for l in labels]
        self.properties = {Varname(k): v for k, v in properties.items()}
        self.versioned_labels = versioned_labels
        self.child = CypherVariable(labels[-1])
        self.speclist = [CypherVariable(p.namehint+'spec') for p in self.parents]
        self.specs = CypherVariable('specs')
        factors = [v for v in properties.values() if isinstance(v, CypherVariable)]
        super().__init__(factors + list(parents.keys()), [self.child] + self.speclist + [self.specs])
        if len(versioned_labels):
            self.version = CypherVariable('version')
            self.output_variables.append(self.version)

    def make_specs(self):
        lines = []
        specs = []
        for (parent, reltype), spec in zip(self.parents.items(), self.speclist):
            if isinstance(parent, Collection):
                lines.append(f"WITH *, [i in range(0, size({parent}) - 1) | [{parent}[i], '{reltype}', {{order: i}}]] as {spec}")
            else:
                lines.append(f"WITH *, [[{parent}, '{reltype}', {{order: 0}}]] as {spec}")
            specs.append(spec)
        return '\n'.join(lines) + f'\nWITH *, {"+".join(map(str, specs))} as {self.specs}'

    def to_cypher(self):
        merge = self.make_specs()
        merge += f"\nCALL custom.multimerge({self.specs}, {self.labels}, {self.properties}, {{dbcreated: {self.timestamp}}}, {{dbcreated: {self.timestamp}}}) YIELD child as {self.child}"
        if len(self.versioned_labels):
            merge += f"\nCALL custom.version({self.specs}, {self.child}, {self.versioned_labels}, 'version') YIELD version as {self.version}"
        return merge


class MatchMany2One(MergeMany2One):
    def to_cypher(self):
        match = self.make_specs()
        match += f"\nCALL custom.multimatch({self.specs}, {self.labels}, {self.properties}) YIELD child as {self.child}"
        return match


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


class MergeNode(MatchNode):
    keyword = 'MERGE'

    def to_cypher(self):
        return super().to_cypher() + f" ON CREATE SET {self.out}.dbcreated = {self.timestamp}"


def _mergematch_node(labels, properties, parents=None, versioned_labels=None, merge=False):
    query = CypherQuery.get_context()
    if parents is None:
        parents = {}
    if versioned_labels is None:
        versioned_labels = []
    if not parents:
        if merge:
            statement = MergeNode(labels, properties)
        else:
            statement = MatchNode(labels, properties)
    else:
        if merge:
            statement = MergeMany2One(parents, labels, properties, versioned_labels)
        else:
            statement = MatchMany2One(parents, labels, properties, versioned_labels)
    query.add_statement(statement)
    return statement.output_variables[0]


def match_node(labels, properties, parents=None, versioned_labels=None):
    return _mergematch_node(labels, properties, parents, versioned_labels, merge=False)


def merge_node(labels, properties, parents=None, versioned_labels=None):
    return _mergematch_node(labels, properties, parents, versioned_labels, merge=True)


@contextmanager
def unwind(*args, enumerate=False):
    query = CypherQuery.get_context()  # type: CypherQuery
    unwinder = Unwind(*args, enumerated=enumerate)
    query.open_context()  # allow this context to accumulate variables
    query.add_statement(unwinder)  # append the actual statement
    if len(unwinder.passed_outputs) == 1:
        yield unwinder.passed_outputs[0]  # give back the unwound variables
    else:
        yield tuple(unwinder.passed_outputs)
    query.close_context()  # remove the variables from being referenced from now on
    previous = [param for context in query.open_contexts for param in context]
    query.statements.append(Collect(previous))  # allow the collections to be accessible - force


def collect(*variables: CypherVariable):
    query = CypherQuery.get_context()  # type: CypherQuery
    collector = query.statements[-1]
    if not isinstance(collector, Collect):
        raise NameError(f"You must use collect straight after a with context")
    for variable in variables:
        if variable not in query.closed_context:
            raise ValueError(f"Cannot collect a non unwound variable")
    collector = Collect(collector.previous, *variables)
    query.statements[-1] = collector
    r = [collector[variable] for variable in variables]
    query.open_contexts[-1] = r
    if len(r) == 1:
        return r[0]
    return r


def groupby(variable_list, propertyname):
    if not isinstance(variable_list, Collection):
        raise TypeError(f"{variable_list} is not a collection")
    query = CypherQuery.get_context()  # type: CypherQuery
    g = GroupBy(variable_list, propertyname)
    query.add_statement(g)
    return g.output_variables[0]
