import itertools
import logging
import time
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import copy
from typing import Tuple, List, Dict, Any, Union, TYPE_CHECKING
from warnings import warn

import networkx as nx
from networkx import NetworkXNoPath, NodeNotFound
from py2neo.cypher import Cursor

from weaveio.hierarchy import Hierarchy

from .exceptions import AmbiguousPathError, CardinalityError, DisjointPathError, AttributeNameError
from .parser import QueryGraph
from .results import Table, AstropyTable
from ..path_finding import collapse_classes_to_superclasses

if TYPE_CHECKING:
    from .objects import ObjectQuery, Query, AttributeQuery
    from ..data import Data


@contextmanager
def logtime(name):
    start = time.perf_counter()
    yield
    logging.info(f"{name} took {time.perf_counter() - start: .3f} seconds")


def get_groupids(graph, params):
    groupids = []
    names = []
    for name, query in graph.groupbys.items():
        if name in params:
            groupids.append(query._iterate(limit=None, skip=0, distinct=True, no_groups=True))
            names.append(name)
    return groupids, names

def _execute_groups(groupids, names, params):
    for values in itertools.product(*groupids):
        for name, value in zip(names, values):
            params[name] = value
        yield values, params


SplitQuery = namedtuple('SplitQuery', ['index', 'query'])
SplitResult = namedtuple('SplitResult', ['index', 'result'])


class QueryFunctionBase:
    def _precompile(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'




class BaseQuery(QueryFunctionBase):
    one_row = False
    one_column = False

    def _debug_output(self, skip=0, limit=None, distinct=False, no_cache=False, graph_export_fname=None,
                      exclusive=True):
        n = self._precompile()
        cached_params = n._get_cached_parameters()
        placeholder_params = n._G.dependency_parameters(self._node)
        params = {**placeholder_params, **cached_params}
        c = n._to_cypher(skip, limit, distinct, no_cache)
        params = n._prepare_parameters(c, params)
        if graph_export_fname is not None:
            if exclusive:
                n._G.export(graph_export_fname, n._node)
            else:
                n._G.export(graph_export_fname)
        return '\n'.join(c), self._G.G.nodes[n._node]['ordering'], params

    def raise_error_with_suggestions(self, obj, exception: Exception):
        self._data.autosuggest(obj, self._obj, exception)
        raise exception

    def __repr__(self):
        try:
            return f'<{self.__class__.__name__}({self._previous._obj}-{self._obj})>'
        except AttributeError:
            return f'<{self.__class__.__name__}({self._obj})>'

    def _precompile(self) -> 'BaseQuery':
        return self

    def _prepare_parameters(self, lines, parameters):
        params = {}
        for k, v in parameters.items():
            if any(k in l for l in lines):
                if isinstance(v, AstropyTable):
                    cols = [c for c in v.colnames if any(c in l for l in lines)]
                    assert len(cols), "No columns are used, but table is used. This should not happen"
                    params[k] = v[cols]
                else:
                    params[k] = v
        return params

    def _to_cypher(self, skip, limit, distinct, no_cache=False) -> List[str]:
        """
        returns the cypher lines, cypher parameters, names of columns, expect_one_row, expect_one_column
        """
        lines = self._G.cypher_lines(self._node, no_cache)
        if distinct:
            lines[-1] = 'RETURN DISTINCT'.join(lines[-1].rsplit('RETURN', 1))
        if skip > 0:
            lines.append(f"\nSKIP {skip}")
        if limit is not None:
            lines.append(f"\nLIMIT {limit}")
        return lines

    def _compile(self, skip, limit, distinct) -> Tuple['BaseQuery', List[str]]:
        with logtime('compiling'):
            r = self._precompile()
            return r, r._to_cypher(skip, limit, distinct)

    def _execute_single_query(self, query, cypher, params):
        return query._data.graph.execute(cypher, **{k.replace('$', ''): v for k, v in params.items()})

    def _get_cached_parameters(self):
        params = {}
        for node in self._G.restricted(self._node).nodes:
            params.update(self._G.G.nodes[node].get('cached_params', {}))
        return params

    def _execute(self, skip=0, limit=None, distinct=False, no_groups=False):
        """
        returns a new query (using _precompile) and an iterator over cursors.
        For queries that have not been `split`, this is a single cursor.
        For queries that have been `split`, there will be multiple cursors.
        :param skip: number of rows to skip
        :param limit: number of rows to limit to
        :param distinct: whether to return distinct rows
        :param no_groups: whether to return a single cursor or a generator of cursors
        """
        with logtime('executing'):
            cached_params = self._get_cached_parameters()
            placeholder_params = self._G.dependency_parameters(self._node)
            params = {**placeholder_params, **cached_params}
            if not no_groups and any(v == '<placeholder>' for v in params.values()):
                groupids, names = get_groupids(self._G, params)
                if groupids:
                    return _execute_groups(groupids, names, params), self
            new, lines = self._compile(skip, limit, distinct)
            cypher = '\n'.join(lines)
            params = self._prepare_parameters(lines, params)
            return self._execute_single_query(new, cypher, params), new

    def _iterate_groups(self, generator, query):
        for split_indexes, params in generator:
            copied = copy(query)  # copy the query, so that the parameters are not shared between the cursors
            query._G.G.nodes[copied._node]['cached_params'] = params  # store the groups for when the query is executed
            yield SplitQuery(split_indexes, copied)

    def _iterate(self, skip=0, limit=None, distinct=False, no_groups=False):
        cursor_or_gen, new = self._execute(skip, limit, distinct, no_groups)  # produces either a cursor or a generator of cursors
        if not isinstance(cursor_or_gen, Cursor):
            yield from self._iterate_groups(cursor_or_gen, new)
        else:
            rows = new._data.rowparser.iterate_cursor(cursor_or_gen, new._names, new._is_products, True)
            i = None
            with logtime('total streaming (by iteration)'):
                for i, row in enumerate(rows):
                    yield new._post_process_row(row)
            if limit == i:
                warnings.simplefilter("always")
                warn(f"This query has been capped to {limit} rows but the result is larger than that. "
                     f"Consider using the `limit` parameter to better limit the result or set `limit` to None to get "
                     f"the full result (although this is not recommended).")

    def __iter__(self):
        yield from self._iterate()

    def _to_table(self, skip=0, limit=None, distinct=False) -> Tuple[Table, 'BaseQuery']:
        with logtime('total streaming'):
            cursor_or_gen, new = self._execute(skip, limit, distinct)  # produces either a cursor or a generator of cursors
            if not isinstance(cursor_or_gen, Cursor):
                return [SplitResult(split_indexes, query._to_table(skip, limit, distinct)) for split_indexes, query in self._iterate_groups(cursor_or_gen, new)], new
            return new._data.rowparser.parse_to_table(cursor_or_gen, new._names, new._is_products), new

    def _post_process_table(self, result):
        if isinstance(result, SplitResult):
            return result.index, self._post_process_table(result.result)
        if self.one_column:
            result = result[result.colnames[0]].data
        if self.one_row:
            result = result[0]
        return result

    def _post_process_row(self, row):
        if self.one_column:
            return row[row.colnames[0]]
        return row

    def __call__(self, skip=0, limit=1000, distinct=False, **kwargs):
        tbl, new = self._to_table(skip, limit, distinct)
        if isinstance(tbl, list):
            return tbl
        tbl = new._post_process_table(tbl)
        try:
            if not len(tbl):
                return tbl
        except TypeError:
            return tbl  # not a table, just a value, so return it
        if limit == 1:
            return tbl[0]
        try:
            if limit == len(tbl):
                warnings.simplefilter("always")
                warn(f"This query has been capped to {limit} rows but the result is larger than that. "
                     f"Consider using the `limit` parameter to better limit the result or set `limit` to None to get "
                     f"the full result (although this is not recommended).")
        except TypeError:
            pass
        return tbl


    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: Union['Query', 'AttributeQuery', 'ObjectQuery'] = None,
                 obj: str = None, start: 'Query' = None, index_node = None,
                 single=False, names=None, is_products=None, attrs=None, dtype=None, *args, **kwargs) -> None:
        from .objects import ObjectQuery
        self._single = single
        self._data = data
        if G is None:
            self._G = QueryGraph()
        else:
            self._G = G
        if node is None:
            self._node = self._G.start
        else:
            self._node = node
        self._previous = previous
        self.__cypher = None
        self._index_node = index_node
        self._obj = obj
        if previous is not None:
            if self._index_node is None:
                if isinstance(previous, ObjectQuery):
                    self._index_node = previous._node
                elif self._index_node != 'start':
                    self._index_node = previous._index_node
            if obj is None:
                self._obj = previous._obj
        if start is None:
            self._start = self
        else:
            self._start = start
        if self._obj is not None:
            if not self._obj.startswith('_'):  # system name for other purposes
                self._obj = self._normalise_object(self._obj)[0]
        self._names = [] if names is None else names
        self._is_products = [False]*len(self._names) if is_products is None else is_products
        self.attrs = attrs
        self.dtype = dtype

    def _get_object_of(self, maybe_attribute: str) -> Tuple[str, bool, bool]:
        """
        Given a str that might refer to an attribute, return the object that contains that attribute
        and also whether the attribute and the object are singular.
        plurality table:
            inferred obj | attr
                s        |  s      (i.e. run.obid -> run.ob.obid)
                p        |  p      (i.e. ...-> run.l1specs.snrs) - cannot be inferred just from a name
                s        |  p      (i.e. run.obids -> run.ob.obids)
                p        |  s      (i.e. run.snr -> run.l1spec.snr) - is not allowed
        attribute storage scenarios:
            * attribute is contained in this object - return self
            * attribute is contained in exactly one object - return object
            * attribute is contained in more than one object and ...
                * any of those objects don't have the same inherited class - raise AmbiguousError
                * only subclass objects and all subclass objects from a superclass contain the attribute - return the superclass
                * only subclass objects and not all subclass objects from a superclass contain the attribute - return the superclass
        If a singular path exists, use that
        :return: obj, obj_is_singular, attr_is_singular
        """
        single_name = self._data.singular_name(maybe_attribute)
        if self._obj is not None:
            if self._data.class_hierarchies[self._obj] in self._data.factor_hierarchies[single_name]:
                return self._obj, True, self._data.is_singular_name(maybe_attribute)
        want_single = single_name == maybe_attribute
        # try to get singular paths
        hs = {h for h in self._data.factor_hierarchies[single_name] if self._has_path_to_object(h, True)}
        if len(hs) == 0: # if no singular paths then get all paths
            hs = {h for h in self._data.factor_hierarchies[single_name] if self._has_path_to_object(h, want_single)}
        hs = set(collapse_classes_to_superclasses(self._data.hierarchy_graph, hs))
        if len(hs) == 1:
            obj = hs.pop().__name__
        elif not hs:
            raise AmbiguousPathError(f"There is no path to any object that contains `{single_name}`. {self._data.factor_hierarchies[single_name]} are not reachable")
        else:
            raise AmbiguousPathError(f"`{single_name}` is contained within more than one unrelated object {[h.__name__ for h in hs]}. Be specific.")
        if self._obj is None:
            return obj, True, False
        if not self._data.is_factor_name(maybe_attribute):
            raise ValueError(f"{maybe_attribute} is not a valid attribute name")
        _, obj_is_singulars = self._get_path_to_object(obj, False)
        obj_is_singular = len(obj_is_singulars) == 1 and obj_is_singulars[0]
        attr_is_singular = self._data.is_singular_name(maybe_attribute)
        if obj_is_singular and attr_is_singular:
            pass  # run.obid -> run.ob.obid
        elif not obj_is_singular and not attr_is_singular:
            pass
        elif obj_is_singular and not attr_is_singular:
            pass  # run.obids -> run.ob.obids
        else:
            # ob.runid -> Error
            plural_name = self._data.plural_name(obj)
            original = self._data.singular_name(self._obj)
            raise CardinalityError(f"Requested one `{single_name}` from `{original}` "
                                   f"when `{original}` has several `{plural_name}`")
        return obj, obj_is_singular, attr_is_singular

    def _normalise_object(self, obj: str):
        """
        returns the __name__ of the respective class of the obj and whether it is plural/singular.
        If not a class, raise KeyError
        """
        if obj in self._data.class_hierarchies:
            return obj, True
        singular = self._data.singular_name(obj)
        if singular not in self._data.singular_hierarchies:
            raise KeyError(f"{obj} is not a valid object name")
        try:
            h = self._data.plural_hierarchies[obj.lower()]
            singular = False
        except KeyError:
            h = self._data.singular_hierarchies[obj.lower()]
            singular = True
        return h.__name__, singular

    @property
    def _cypher(self):
        if self.__cypher is None:
            self.__cypher = self._G.cypher_lines(self._node)
        return self.__cypher

    @property
    def _result_graph(self):
        return self._G.restricted(self._node)

    def _plot_graph(self, fname):
        return self._G.export(fname, self._node)

    @classmethod
    def _spawn(cls, parent: 'BaseQuery', node, obj=None, index_node=None, single=False, *args, **kwargs):
        return cls(parent._data, parent._G, node, parent, obj, parent._start, index_node, single, *args, **kwargs)

    def _get_path_to_object(self, obj, want_single) -> Tuple[List[str], List[bool]]:
        return self._data.paths_to_hierarchy(self._obj, obj, want_single)

    def _has_path_to_object(self, obj, want_single) -> bool:
        if self._obj is None:
            return True
        try:
            return len(self._get_path_to_object(obj, want_single)[0]) > 0
        except NetworkXNoPath:
            return False

    def _slice(self, slc):
        """
        obj[slice]
        filter, shrinks, destructive
        filter by using HEAD/TAIL/SKIP/LIMIT
        e.g. obs.runs[:10] will return the first 10 runs for each ob (in whatever order they were in)
        you must use query(skip, limit) to request a specific number of rows in total since this is unrelated to the actual query
        """
        raise NotImplementedError

    def _filter_by_mask(self, mask, single=None, split_node=False):
        """
        obj[boolean_filter]
        filter, shrinks, destructive
        filters based on a list of True/False values constructed beforehand, parentage of the booleans must be derived from the obj
        e.g. `ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red']` gives only the red stacks
             `ob.l1stackedspectra[ob.l1singlespectra == 'red']` is invalid since the lists will not be the same size or have the same parentage
        """
        if mask.dtype != 'boolean':
            raise TypeError(f"The predicate in a filter `query[predicate]`, must be of type boolean. {mask} is of type {mask.dtype}. ")
        single = single if single is not None else self._single
        try:
            n = self._G.add_filter(self._node, mask._node, direct=False, force_single=single, split_node=split_node)
        except SyntaxError:
            raise DisjointPathError(f"{self} cannot be filtered by {mask} since `{self._obj}` != `{mask._obj}`."
                                    f"This may be because a nonsensical `wrt` has specified in an aggregation.")
        return self.__class__._spawn(self, n, single=single)

    def _aggregate(self, wrt, string_op, op_name, predicate=False, expected_dtype=None, returns_dtype=None, remove_infs=None):
        if wrt is None:
            wrt = self._start
        try:
            if predicate:
                n = self._G.add_predicate_aggregation(self._node, wrt._node, string_op)
            else:
                n = self._G.add_aggregation(self._node, wrt._node, string_op, remove_infs, expected_dtype, self.dtype)
        except SyntaxError:
            raise DisjointPathError(f"Cannot aggregate {self} into {wrt} since they don't share a parent query")
        from .objects import AttributeQuery
        return AttributeQuery._spawn(self, n, wrt._obj, wrt._node, dtype=returns_dtype, single=True, factor_name=op_name)
