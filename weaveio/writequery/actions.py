from contextlib import contextmanager

from .base import CypherQuery, CypherVariable, Collection
from .statements import Unwind, Collect, GroupBy


@contextmanager
def unwind(*args, enumerated=False):
    query = CypherQuery.get_context()  # type: CypherQuery
    unwinder = Unwind(*args, enumerated=enumerated)
    query.open_context()  # allow this context to accumulate variables
    query.add_statement(unwinder)  # append the actual statement
    if len(unwinder.passed_outputs) == 1:
        yield unwinder.passed_outputs[0]  # give back the unwound variables
    else:
        yield tuple(unwinder.passed_outputs)
    query.close_context()  # remove the variables from being referenced from now on
    query.open_contexts = [[param for param in c if param not in args] for c in query.open_contexts]
    # previous = [param for context in query.open_contexts for param in context if param not in args]
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
    query.open_contexts[-1] += r
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