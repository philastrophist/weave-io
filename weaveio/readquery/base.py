import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any, Union, TYPE_CHECKING

import networkx as nx
from networkx import NetworkXNoPath, NodeNotFound
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


class BaseQuery:
    one_row = False
    one_column = False

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

    def _prepare_parameters(self, lines):
        params = {}
        aliases = self._G.get_unwind_variables(self._node)
        for k, v in self._G.parameters.items():
            if any(k in l for l in lines):
                if isinstance(v, AstropyTable):
                    cols = [c for c in v.colnames if any(f"{aliases[k]}.`{c}`" in l for l in lines)]
                    assert len(cols), "No columns are used, but table is used. This should not happen"
                    params[k] = v[cols]
                else:
                    params[k] = v
        return params

    def _to_cypher(self) -> Tuple[List[str], Dict[str, Any]]:
        """
        returns the cypher lines, cypher parameters, names of columns, expect_one_row, expect_one_column
        """
        lines = self._G.cypher_lines(self._node)
        return lines, self._prepare_parameters(lines)

    def _compile(self) -> Tuple['BaseQuery', Tuple[List[str], Dict[str, Any]]]:
        with logtime('compiling'):
            r = self._precompile()
            return r, r._to_cypher()

    def _execute(self, skip=0, limit=None):
        with logtime('executing'):
            new, (lines, params) = self._compile()
            if skip > 0:
                lines.append(f"SKIP {skip}")
            if limit is not None:
                lines.append(f"LIMIT {limit}")
            cursor = new._data.graph.execute('\n'.join(lines), **{k.replace('$', ''): v for k,v in params.items()})
            return cursor, new

    def _iterate(self, skip=0, limit=None):
        cursor, new = self._execute(skip, limit)
        rows = new._data.rowparser.iterate_cursor(cursor, new._names, new._is_products, True)
        with logtime('total streaming (by iteration)'):
            for row in rows:
                yield new._post_process_row(row)

    def __iter__(self):
        yield from self._iterate()

    def _to_table(self, skip=0, limit=None) -> Tuple[Table, 'BaseQuery']:
        with logtime('total streaming'):
            cursor, new = self._execute(skip, limit)
            return new._data.rowparser.parse_to_table(cursor, new._names, new._is_products), new

    def _post_process_table(self, result):
        if self.one_column:
            result = result[result.colnames[0]].data
        if self.one_row:
            result = result[0]
        return result

    def _post_process_row(self, row):
        if self.one_column:
            return row[row.colnames[0]]
        return row

    def __call__(self, skip=0, limit=None, **kwargs):
        tbl, new = self._to_table(skip, limit)
        tbl = new._post_process_table(tbl)
        if limit == 1:
            return tbl[0]
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

    def _filter_by_mask(self, mask):
        """
        obj[boolean_filter]
        filter, shrinks, destructive
        filters based on a list of True/False values constructed beforehand, parentage of the booleans must be derived from the obj
        e.g. `ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red']` gives only the red stacks
             `ob.l1stackedspectra[ob.l1singlespectra == 'red']` is invalid since the lists will not be the same size or have the same parentage
        """
        try:
            n = self._G.add_filter(self._node, mask._node, direct=False)
        except SyntaxError:
            raise DisjointPathError(f"{self} cannot be filtered by {mask} since `{self._obj}` != `{mask._obj}`."
                                    f"This may be because a nonsensical `wrt` has specified in an aggregation.")
        return self.__class__._spawn(self, n, single=self._single)

    def _aggregate(self, wrt, string_op, predicate=False, expected_dtype=None, returns_dtype=None, remove_infs=None):
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
        return AttributeQuery._spawn(self, n, wrt._obj, wrt._node, dtype=returns_dtype, single=True)
