from weaveio.basequery.common import FrozenQuery
from weaveio.basequery.dissociated import Dissociated
from weaveio.data import Data


def _template_aggregator(string_function, name, item: Dissociated, wrt: FrozenQuery = None):
    if isinstance(wrt, Data):
        branch = item.branch.handler.entry
    elif wrt is None:
        branch = item.branch.handler.entry
    else:
        branch = wrt.branch
    new = branch.aggregate(string_function, item.variable, item.branch, True, name)
    return Dissociated(item.handler, new, new.action.target)

def sum(item, wrt):
    return _template_aggregator('sum({x})', 'sum', item, wrt)


def max(item, wrt):
    return _template_aggregator('max({x})', 'max', item, wrt)


def min(item, wrt):
    return _template_aggregator('min({x})', 'min', item, wrt)


def all(item, wrt):
    raise NotImplementedError


def any(item, wrt):
    raise NotImplementedError


def count(item, wrt):
    raise NotImplementedError
