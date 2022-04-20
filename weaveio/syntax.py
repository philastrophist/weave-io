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
from typing import List, overload
from typing_extensions import SupportsIndex

from .parser import QueryGraph


class BaseQuery:
    def __init__(self, G: QueryGraph = None, node=None, previous: 'BaseQuery' = None) -> None:
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
        self._index = None
        if previous is not None:
            if isinstance(previous, ObjectQuery):
                self._index = previous
            else:
                self._index = previous._index

    @property
    def _cypher(self):
        if self.__cypher is None:
            self.__cypher = self._G.cypher_lines(self._node)
        return self.__cypher

    @classmethod
    def _spawn(cls, parent, node):
        return cls(parent._G, node, parent)

    def _get_path_to_object(self, obj):
        return '-->', False

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
        n = self._G.add_filter(self._node, mask._node, direct=False)
        return self.__class__._spawn(self, n)


class Query(BaseQuery):
    def _traverse_to_specific_object(self, obj):
        n = self._G.add_start_node(obj)
        return ObjectQuery._spawn(self, n, obj)

    def _traverse_by_object_index(self, obj, index):
        name = self._G.add_parameter(index)
        travel = self._G.add_start_node(obj)
        i = self._G.add_getitem(travel, 'id')
        eq = self._G.add_scalar_operation(i, f'{{0}} = {name}', f'id={index}')
        n = self._G.add_filter(travel, eq, direct=True)
        return ObjectQuery._spawn(self, n, obj)

class ObjectQuery(BaseQuery):
    def __init__(self, G: QueryGraph = None, node=None, previous: 'BaseQuery' = None, obj = None) -> None:
        super().__init__(G, node, previous)
        self._obj = obj

    @classmethod
    def _spawn(cls, parent, node, obj):
        return cls(parent._G, node, parent, obj)

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

    def _traverse_to_specific_object(self, obj):
        """
        obj.obj
        traversal, expands
        e.g. `ob.runs`  reads "for each ob, get its runs"
        """
        path, single = self._get_path_to_object(obj)
        n = self._G.add_traversal(self._node, path, obj, single)
        return ObjectQuery._spawn(self, n, obj)

    def _traverse_by_object_index(self, obj, index):
        """
        obj['obj_id']
        filter, id, shrinks, destructive
        this is a filter which always returns one object
        if the object with this id is not in the hierarchy, then null is returned
        e.g. `obs[1234]` filters to the single ob with obid=1234
        """
        name = self._G.add_parameter(index)
        path, single = self._get_path_to_object(obj)
        travel = self._G.add_traversal(self._node, path, obj, single)
        i = self._G.add_getitem(travel, 'id')
        eq = self._G.add_scalar_operation(i, f'{{0}} = {name}', f'id={index}')
        n = self._G.add_filter(travel, eq, direct=True)
        return ObjectQuery._spawn(self, n, obj)

    def _traverse_by_object_indexes(self, obj, indexes: List):
        raise NotImplementedError
        indexes = list(indexes)
        name = self._G.add_parameter(indexes)
        path, single = self._get_path_to_object(obj)
        travel = self._G.add_traversal(self._node, path, obj, True, name)
        i = self._G.add_getitem(travel, 'id')
        n = self._G.add_filter(travel, eq, direct=True)
        return ObjectQuery._spawn(self, n, obj)

    def _select_attribute(self, attr):
        """
        # obj['factor'], obj.factor
        fetches an attribute from the object and cuts off any other functions after that
        if the factor is not found in the object then the nearest corresponding object is used.
        e.g. `exposure.mjd` returns the mjd held by exposure
             `run.mjd` returns the mjd by a run's exposure (still only one mjd per run though)
             `run.cnames` returns the cname of each target in a run (this is a list per run)
        """
        n = self._G.add_getitem(self._node, attr)
        return AttributeQuery._spawn(self, n)

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
            if not all(isinstance(i, (str, float, int)) for i in item):
                raise TypeError(f"Cannot index by non str/float/int values")
            if any(self._get_object_of(i, raise_error=False) for i in item):
                return self._select_attributes(item)
            else:
                # go back and be better
                return self._previous._traverse_by_object_indexes(self._obj, item)
        elif isinstance(item, slice):
            return self._slice(item)
        elif isinstance(item, AttributeQuery):
            return self._filter_by_mask(item)  # a boolean_mask
        else:
            try:
                obj = self._get_object_of(item)  # if factor
                return self._traverse_to_specific_object(obj)._select_attribute(item)
            except ValueError:
                # index
                return self._previous._traverse_by_object_index(self._obj, item)


class AttributeQuery(BaseQuery):
    def __init__(self, G: QueryGraph = None, node=None, previous=None) -> None:
        super().__init__(G, node, previous)

    def _perform_arithmetic(self, op_string, op_name, other=None):
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
        if isinstance(other, Query):
            n = self._G.add_combining_operation(op_string, op_name, self._node, other._node)
        elif isinstance(other, ObjectQuery):
            raise TypeError(f"Cannot do arithmetic directly on objects")
        else:
            n = self._G.add_scalar_operation(self._node, op_string, op_name)
        return AttributeQuery._spawn(self, n)

    def _basic_scalar_function(self, name):
        return self._perform_arithmetic(f'{name}({{0}})', name)

    def _basic_math_operator(self, operator, other, switch=False):
        if not isinstance(other, AttributeQuery):
            other = self._G.add_parameter(other, 'add')
            string_op = f'{other} {operator} {{0}}' if switch else f'{{0}} {operator} {other}'
        else:
            string_op = f'{{1}} {operator} {{0}}' if switch else f'{{0}} {operator} {{1}}'
        return self._perform_arithmetic(string_op, operator, other)

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
        return self._basic_math_operator('=', other)

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

    def __str__(self):
        return self._basic_scalar_function('toString')

    def __int__(self):
        return self._basic_scalar_function('toInteger')

    def __float__(self):
        return self._basic_scalar_function('toFloat')

    def __abs__(self):
        return self._basic_scalar_function('abs')

    def _aggregate(self):
        """
        aggregation(factor, wrt=obj)
        aggregation, destructive
        shrinks to one level in the parentage
        [sum, mean, max, any, all, etc...]
        e.g. sum(obs.snrs, wrt=None)  will return the sum of all snrs ever
        e.g. sum(obs.snrs, wrt=obs) will return the sum of all snrs 'per' ob (so rows are maintained for ob)
        """
        raise NotImplementedError

    def _compile(self):
        r = self._G.add_results(self._index._node, self._node)
        return self._G.cypher_lines(r), self._G.parameters



class ListAttributeQuery(BaseQuery):
    pass
