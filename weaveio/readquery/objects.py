"""
Queries are a little different
Each time you iterate along, you dont add to the graph straight away.
This gives the Query time to modify the meaning of statements:
without waiting,
    ob.runs[1234] would be [traverse(ob), traverse(run), CopyAndFilter(id=1234)]
with waiting,
    ob.runs[1234] would be [traverse(ob), FilteredMatch(run, id=1234, optional=True)]
    much better!
it treats two chained expressions as one action
"""
from itertools import chain
from typing import List, TYPE_CHECKING, Tuple, Dict, Any, Union

from .base import BaseQuery, CardinalityError
from .parser import QueryGraph, ParserError
from .utilities import is_regex

if TYPE_CHECKING:
    from weaveio.data import Data


class ObjectQuery(BaseQuery):
    def _precompile(self) -> 'TableQuery':
        """
        Automatically returns all single links to this object
        """
        single_objs = self._data.all_single_links_to_hierarchy(self._obj)
        factors = {f"{o.singular_name}" if o.__name__ == self._obj else f"{o.singular_name}.{f}" for o in single_objs for f in o.factors}
        try:
            factors.remove('id')
            factors = ['id'] + sorted(factors)
        except KeyError:
            factors = sorted(factors)
        return self._make_table(*factors)

    def _select_all_attrs(self):
        h = self._data.class_hierarchies[self._data.class_name(self._obj)]
        return self.__getitem__(h.products_and_factors)

    def _traverse_to_generic_object(self):
        """
        obj.generic_obj['specific_type_name']
        filter, shrinks, destructive
        filters based on the type of object to make it more specific
        only shared factors can be accessed
        e.g. `obj.spectra` could be l1singlespectra, l1stackedspectra, l2modelspectra etc
             but `obj.spectra['l1']` ensures that it is only `l1` and only `l1` factors can be accessed
                 `obj.spectra['single']`
        """
        raise NotImplementedError

    def _traverse_to_specific_object(self, obj, want_single):
        """
        obj.obj
        traversal, expands
        e.g. `ob.runs`  reads "for each ob, get its runs"
        """
        path, single = self._get_path_to_object(obj, want_single)
        n = self._G.add_traversal(self._node, path, obj, single)
        return ObjectQuery._spawn(self, n, obj, single=want_single)

    def _traverse_by_object_index(self, obj, index):
        """
        obj['obj_id']
        filter, id, shrinks, destructive
        this is a filter which always returns one object
        if the object with this id is not in the hierarchy, then null is returned
        e.g. `obs[1234]` filters to the single ob with obid=1234
        """
        name = self._G.add_parameter(index)
        path, single = self._get_path_to_object(obj, False)
        travel = self._G.add_traversal(self._node, path, obj, single)
        i = self._G.add_getitem(travel, 'id')
        eq, _ = self._G.add_scalar_operation(i, f'{{0}} = {name}', f'id={index}')
        n = self._G.add_filter(travel, eq, direct=True)
        return ObjectQuery._spawn(self, n, obj, single=True)

    def _traverse_by_object_indexes(self, obj, indexes: List):
        raise NotImplementedError

    def _select_attribute(self, attr, want_single):
        """
        # obj['factor'], obj.factor
        fetches an attribute from the object and cuts off any other functions after that
        if the factor is not found in the object then the nearest corresponding object is used.
        e.g. `exposure.mjd` returns the mjd held by exposure
             `run.mjd` returns the mjd by a run's exposure (still only one mjd per run though)
             `run.cnames` returns the cname of each target in a run (this is a list per run)
        """
        attr = self._data.singular_name(attr)
        n = self._G.add_getitem(self._node, attr)
        return AttributeQuery._spawn(self, n, single=want_single, factor_name=attr)

    def _select_or_traverse_to_attribute(self, attr):
        obj, obj_is_single, attr_is_single = self._get_object_of(attr)  # if factor
        if obj == self._obj:
            return self._select_attribute(attr, attr_is_single)
        r = self._traverse_to_specific_object(obj, obj_is_single)._select_attribute(attr, attr_is_single)
        r._index_node = self._node
        return r

    def _make_table(self, *items):
        """
        obj['factor_string', AttributeQuery]
        obj['factora', 'factorb']
        obj['factora', obj.obj.factorb]
        """
        attrs = [item if isinstance(item, AttributeQuery) else self.__getitem__(item) for item in items]
        names = [item._factor_name if isinstance(item, AttributeQuery) else item for item in items]
        n = self._G.add_results_table(self._node, [a._node for a in attrs],
                                      [a._single for a in attrs], dropna=[self._node])
        return TableQuery._spawn(self, n, names=names)

    def _traverse_to_relative_object(self):
        """
        obj.relative_path
        traversal, expands
        traverses by a specifically labelled relation instead of a name of an object
        e.g. l1stackedspectra.adjunct_spectrum will get the spectrum in the other arm
        """
        raise NotImplementedError

    def _filter_by_relative_index(self):
        """
        obj1.obj2s['relative_id']
        filter, relative id, shrinks, destructive
        filters based on the relations between many `obj2`s and the `obj1`
        e.g. `ob.plate_exposures.fibre_exposures.single_spectra['red']` returns the red spectra
        the relative_id are explicitly stated in the database structure
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """
        item can be an id, a factor name, a list of those, a slice, or a boolean_mask
        """
        if isinstance(item, (tuple, list)):
            if not all(isinstance(i, (str, float, int, AttributeQuery)) for i in item):
                raise TypeError(f"Cannot index by non str/float/int/AttributeQuery values")
            if any(self._data.is_valid_name(i) for i in item):
                return self._make_table(*item)
            else:
                # go back and be better
                return self._previous._traverse_by_object_indexes(self._obj, item)
        elif isinstance(item, slice):
            return self._slice(item)
        elif isinstance(item, AttributeQuery):
            return self._filter_by_mask(item)  # a boolean_mask
        elif not isinstance(item, str):
            return self._previous._traverse_by_object_index(self._obj, item)
        else:
            try:
                return self._select_or_traverse_to_attribute(item)
            except (KeyError, ValueError):
                if '.' in item:
                    try:
                        obj, attr = item.split('.')
                    except ValueError:
                        raise ValueError(f"{item} cannot be parsed as an `obj.attribute`.")
                    return self.__getitem__(obj).__getitem__(attr)
                try:
                    obj, single = self._normalise_object(item)
                    if obj == self._obj:
                        return self
                    return self._traverse_to_specific_object(obj, single)
                except ValueError:
                    return self._previous._traverse_by_object_index(self._obj, item)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __eq__(self, other):
        return self._select_attribute('id', True).__eq__(other)

class Query(BaseQuery):
    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: 'BaseQuery' = None, obj: str = None, start=None) -> None:
        super().__init__(data, G, node, previous, obj, start, 'start')

    def _compile(self):
        raise NotImplementedError(f"{self.__class__} is not compilable")

    def _traverse_to_specific_object(self, obj):
        obj, single = self._normalise_object(obj)
        if single:
            raise CardinalityError(f"Cannot start query with a single object `{obj}`")
        n = self._G.add_start_node(obj)
        return ObjectQuery._spawn(self, n, obj, single=False)

    def _traverse_by_object_index(self, obj, index):
        obj, single = self._normalise_object(obj)
        name = self._G.add_parameter(index)
        travel = self._G.add_start_node(obj)
        i = self._G.add_getitem(travel, 'id')
        eq, _ = self._G.add_scalar_operation(i, f'{{0}} = {name}', f'id={index}')
        n = self._G.add_filter(travel, eq, direct=True)
        return ObjectQuery._spawn(self, n, obj, single=True)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        try:
            if self._data.is_singular_name(item):
                raise CardinalityError(f"Cannot start a query with a single object `{item}`")
            obj = self._get_object_of(item)[0]
            obj = self._data.plural_name(obj)
            return self._traverse_to_specific_object(obj)._select_attribute(item, True)
        except (KeyError, ValueError):
            return self._traverse_to_specific_object(item)


class AttributeQuery(BaseQuery):
    expect_one_column = True

    def __repr__(self):
        return f'<{self.__class__.__name__}({self._obj}.{self._factor_name})>'

    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: Union['Query', 'AttributeQuery', 'ObjectQuery'] = None,
                 obj: str = None, start: Query = None, index_node=None,
                 single=False, factor_name: str = None, *args, **kwargs) -> None:
        super().__init__(data, G, node, previous, obj, start, index_node, single, [factor_name], *args, **kwargs)
        self._factor_name = factor_name

    def _perform_arithmetic(self, op_string, op_name, other=None, expected_dtype=None):
        """
        arithmetics
        [+, -, /, *, >, <, ==, !=, <=, >=]
        performs some maths on factors that have a shared parentage
        e.g. ob1.runs.snrs + ob2.runs.snrs` is not allowed, even if the number of runs/spectra is the same
        e.g. ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red'].snr + ob.l1stackedspectra[ob.l1stackedspectra.camera == 'blue'].snr is not allowed
        e.g. ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red'].snr + ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red'].adjunct.snr is not allowed
        e.g. sum(ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red'].snr, wrt=None) > ob.l1stackedspectra.snr is allowed since one is scalar,
             we take ob.l1stackedspectra as the hierarchy level in order to continue
        e.g. sum(ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red'].snr, wrt=ob) > ob.l1stackedspectra.snr is allowed since there is a shared parent,
             we take ob.l1stackedspectra as the hierarchy level in order to continue
        """
        if isinstance(other, ObjectQuery):
            raise TypeError(f"Cannot do arithmetic directly on objects")
        if expected_dtype is not None:
            op_string = op_string.replace('{0}', f'to{expected_dtype}({{0}})')
            op_string = op_string.replace('{1}', f'to{expected_dtype}({{1}})')  # both arguments
        if isinstance(other, BaseQuery):
            try:
                n, wrt = self._G.add_combining_operation(op_string, op_name, self._node, other._node)
            except ParserError:
                raise SyntaxError(f"You may not perform an operation on {self} and {other} since one is not an ancestor of the other")
        else:
            n, wrt = self._G.add_scalar_operation(self._node, op_string, op_name)
        return AttributeQuery._spawn(self, n, index_node=wrt, single=True)

    def _basic_scalar_function(self, name):
        return self._perform_arithmetic(f'{name}({{0}})', name)

    def _basic_math_operator(self, operator, other, switch=False):
        if not isinstance(other, AttributeQuery):
            other = self._G.add_parameter(other, 'add')
            string_op = f'{other} {operator} {{0}}' if switch else f'{{0}} {operator} {other}'
        else:
            string_op = f'{{1}} {operator} {{0}}' if switch else f'{{0}} {operator} {{1}}'
        return self._perform_arithmetic(string_op, operator, other)

    def __and__(self, other):
        return self._basic_math_operator('and', other)

    def __rand__(self, other):
        return self._basic_math_operator('and', other, switch=True)

    def __or__(self, other):
        return self._basic_math_operator('or', other)

    def __ror__(self, other):
        return self._basic_math_operator('or', other, switch=True)

    def __xor__(self, other):
        return self._basic_math_operator('xor', other)

    def __rxor__(self, other):
        return self._basic_math_operator('xor', other, switch=True)

    def __invert__(self):
        return self._basic_scalar_function('not')

    def __add__(self, other):
        return self._basic_math_operator('+', other)

    def __radd__(self, other):
        return self._basic_math_operator('+', other, switch=True)

    def __mul__(self, other):
        return self._basic_math_operator('*', other)

    def __rmul__(self, other):
        return self._basic_math_operator('*', other, switch=True)

    def __sub__(self, other):
        return self._basic_math_operator('-', other)

    def __rsub__(self, other):
        return self._basic_math_operator('-', other, switch=True)

    def __truediv__(self, other):
        return self._basic_math_operator('/', other)

    def __rtruediv__(self, other):
        return self._basic_math_operator('/', other, switch=True)

    def __eq__(self, other):
        op = '='
        if isinstance(other, str):
            if is_regex(other):
                op = '=~'
                other = other.strip('/')
        return self._basic_math_operator(op, other)

    def __ne__(self, other):
        return self._basic_math_operator('<>', other)

    def __lt__(self, other):
        return self._basic_math_operator('<', other)

    def __le__(self, other):
        return self._basic_math_operator('<=', other)

    def __gt__(self, other):
        return self._basic_math_operator('>', other)

    def __ge__(self, other):
        return self._basic_math_operator('>=', other)

    def __ceil__(self):
        return self._basic_scalar_function('ceil')

    def __floor__(self):
        return self._basic_scalar_function('floor')

    def __round__(self, ndigits: int):
        raise self._perform_arithmetic(f'round({{0}}, {ndigits}', f'round{ndigits}')

    def __neg__(self):
        return self._perform_arithmetic('-{{0}}', 'neg')

    def __abs__(self):
        return self._basic_scalar_function('abs')

    def _precompile(self) -> 'TableQuery':
        if self._index_node == 'start':
            index = self._G.start
        else:
            index = self._index_node
        r = self._G.add_results_table(index, [self._node], [self._single], dropna=[self._node])
        return AttributeQuery._spawn(self, r, self._obj, index, self._single, factor_name=self._factor_name)


class TableQuery(BaseQuery):
    def __init__(self, data: 'Data', G: QueryGraph = None, node=None,
                 previous: Union['Query', 'AttributeQuery', 'ObjectQuery'] = None, obj: str = None,
                 start: Query = None, index_node=None,
                 single=False, attr_queries=None, names=None, *args, **kwargs) -> None:
        super().__init__(data, G, node, previous, obj, start, index_node, single, names, *args, **kwargs)
        self._attr_queries = attr_queries



class ListAttributeQuery(AttributeQuery):
    pass
