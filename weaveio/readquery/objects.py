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
from functools import reduce
from typing import List, TYPE_CHECKING, Union, Tuple, Dict, Any

from networkx import NetworkXNoPath

from .base import BaseQuery, CardinalityError
from .parser import QueryGraph, ParserError
from .utilities import is_regex
from ..basequery.common import NotYetImplementedError
from ..opr3.hierarchy import Exposure

if TYPE_CHECKING:
    from weaveio.data import Data

class PathError(SyntaxError):
    pass


class BaseObjectQuery(BaseQuery):
    def _traverse_to_specific_object(self, obj, want_single):
        raise NotImplementedError

    def _traverse_to_template_object(self, obj, want_single):
        """
        Template objects are those that are not instantiated in the database, but have subclasses
        which are instantiated in the database.
        """
        obj, obj_is_singular = self._normalise_object(obj)
        options = self._data.expand_template_object(obj, self._obj)  # list of paths of specifics

    def _traverse_dependent_object(self, obj, want_single):
        """
        Dependent objects are instantiated in the database, but they have a variety of parent objects
        e.g. L1SingleSpectrum.NOSS, L1StackSpectrum.NOSS... NOSS is a dependent object since it does
        not define its parents itself
        """
        obj, obj_is_singular = self._normalise_object(obj)
        options = self._data.expand_dependent_object(obj, self._obj)  # list of paths of specifics

    def _traverse_to_generic_object(self, obj, want_single):
        """
        each generic query can have multiple queries
        when a generic query is spawned, its possible options are spawned as well and stored in the generic one
        if the generic is turned into a specific e.g. data.l1spectra['single'], the relevant options are passed on (in this case, one option)
        if the generic is traversed from e.g. data.l1spectra.run, then a traversal of {UNION paths} is added to the query and we continue

        when traversing from a generic,
            it is singular only if all options are singular: l1spectra.noss, l1spectra.adjunct
        when traversing to a generic [this function]!
            if is singular only if there is only one option: run.l1spectra, weavetarget.l1spectra, l1single_spectrum.noss
        """
        obj, obj_is_singular = self._normalise_object(obj)
        options = self._data.expand_generic_object(obj, self._obj)  # list of paths of specifics
        if len(options) > 1 and want_single and not self._subqueries:
            raise SyntaxError(f"{obj} is a generic object and cannot be used as a single object")
        specifics = []
        for path in options:
            if len(path) == 1:
                start = self._traverse_to_specific_object(path[0], want_single, not_generic=True)
                specifics.append(start)
            else:
                try:
                    start = self._subqueries[path[0]]
                except KeyError:
                    start = self._traverse_to_specific_object(path[0], want_single, not_generic=True)
                specifics.append(reduce(lambda a, b: a._traverse_to_specific_object(b, want_single, not_generic=True), path[1:], start))
        if self._previous is None:
            n = self._G.add_start_node(obj)
        else:
            paths = {s._obj: list(self._G.G.in_edges(s._node, data=True))[0][-1]['statement'].path for s in specifics}
            n = self._G.add_union_traversal(self._node, paths, obj, single=False)
        return TemplateObjectQuery._spawn(self, n, obj=obj, subqueries={s._obj: s for s in specifics}, single=want_single)


class SpecificObjectQuery(BaseObjectQuery):
    def _traverse_to_specific_object(self, obj, want_single, not_generic=False):
        """
        obj.obj
        traversal, expands
        e.g. `ob.runs`  reads "for each ob, get its runs"
        """
        if self._data.is_template_object(obj) and not not_generic:
            return self._traverse_to_template_object(obj, want_single)
        elif self._data.is_dependent_object(obj) and not not_generic:
            return self._traverse_to_dependent_object(obj, want_single)
        try:
            path, single = self._get_path_to_object(obj, want_single)
        except NetworkXNoPath:
            if want_single:
                plural = self._data.plural_name(obj)
                msg = f"There is no singular {obj} relative to {self._obj} try using its plural {plural}"
            else:
                msg = f"There is no {obj} relative to {self._obj}"
            raise CardinalityError(msg)
        n = self._G.add_traversal(self._node, path, obj, single)
        return SpecificObjectQuery._spawn(self, n, obj, single=want_single)

    def _traverse_to_template_object(self, obj, want_single):
        options = self._data.expand_template_object(obj)  # specific options
        # check whether there is exactly one option which has a singular path
        if want_single:
            singles = [specific_option for specific_option in options if self._get_path_to_object(specific_option, False)[1]]
            if len(singles) == 1:
                return self._traverse_to_specific_object(singles[0], want_single)
            else:
                raise CardinalityError(f"{self._obj} has multiple {obj}: {options}. Check the cardinality of your query")
        specifics = []
        for specific_obj in options:
            start = self._traverse_to_specific_object(specific_obj, want_single, not_generic=True)
            specifics.append(start)
        if self._previous is None:
            n = self._G.add_start_node(obj)
        else:
            paths = {s._obj: list(self._G.G.in_edges(s._node, data=True))[0][-1]['statement'].path for s in specifics}
            n = self._G.add_union_traversal(self._node, paths, obj, single=False)
        return TemplateObjectQuery._spawn(self, n, obj=obj, subqueries={s._obj: s for s in specifics}, single=want_single)

    def _traverse_to_dependent_object(self, obj, want_single):
        options = self._data.expand_dependent_object(obj)  # specific intermediate options
        # if no options match this query's object, then use all options
        # if exactly one option matches this query's object, then use that option
        if self._obj in options:
            return self._traverse_to_specific_object(obj, want_single, not_generic=True)
        elif want_single:
            raise CardinalityError(f"{self._obj} more than one {obj}: {options}. Check the cardinality of your query")
        else:
            specifics = {}
            for specific_obj in options:
                start = self._traverse_to_specific_object(specific_obj, want_single, not_generic=True)
                end = start._traverse_to_specific_object(obj, False, not_generic=True)
                specifics[specific_obj] = end
            if self._previous is None:
                n = self._G.add_start_node(obj)
            else:
                final_path = list(self._G.G.in_edges(end._node, data=True))[0][-1]['statement'].path
                paths = {specific_obj: self._get_path_to_object(specific_obj, False)[0] for specific_obj in options}
                intermediate = self._G.add_union_traversal(self._node, paths, obj, single=False)
                n = self._G.add_traversal(intermediate, final_path, obj, single=False)
            return DependentObjectQuery._spawn(self, n, obj=obj, subqueries=specifics, single=want_single)

    def _precompile(self) -> 'TableQuery':
        """
        If the object contains only one factor/product and defines no parents/children, return that
        Otherwise, try to return just the id or error
        """
        Obj = self._data.class_hierarchies[self._obj]
        if Obj.idname is not None:
            if len(Obj.products_and_factors) == 2:
                attr = Obj.products_and_factors[0]  # take the only data in it
            else:
                attr = Obj.idname
        elif len(Obj.products_and_factors) == 1:
            attr = Obj.products_and_factors[0]  # take the only data in it
        else:
            raise SyntaxError(f"{self._obj} cannot be returned/identified since it doesn't define any unique idname. "
                              f"If you want to return all singular data for {self._obj} use ...['*']")
        return self.__getitem__(attr)._precompile()

    def _get_all_factors_table(self):
        """
        For use with ['*']
        """
        single_objs = self._data.all_single_links_to_hierarchy(self._obj)
        factors = {f if o.__name__ == self._obj else f"{o.singular_name}.{f}" for o in single_objs for f in o.products_and_factors}
        try:
            factors.remove('id')
            factors = ['id'] + sorted(factors)
        except KeyError:
            factors = sorted(factors)
        return factors

    def _select_all_attrs(self):
        h = self._data.class_hierarchies[self._data.class_name(self._obj)]
        return self.__getitem__(h.products_and_factors)

    def _traverse_by_object_index(self, obj, index):
        """
        obj['obj_id']
        filter, id, shrinks, destructive
        this is a filter which always returns one object
        if the object with this id is not in the hierarchy, then null is returned
        e.g. `obs[1234]` filters to the single ob with obid=1234
        """
        param = self._G.add_parameter(index)
        path, single = self._get_path_to_object(obj, False)
        travel = self._G.add_traversal(self._node, path, obj, single)
        i = self._G.add_getitem(travel, 'id')
        eq, _ = self._G.add_scalar_operation(i, f'{{0}} = {param}', f'id={index}')
        n = self._G.add_filter(travel, eq, direct=True)
        return SpecificObjectQuery._spawn(self, n, obj, single=True)

    def _traverse_by_object_indexes(self, obj, indexes: List):
        param = self._G.add_parameter(indexes)
        path, single = self._get_path_to_object(obj, False)
        one_id = self._G.add_unwind_parameter(self._node, param)
        travel = self._G.add_traversal(self._node, path, obj, single, one_id)
        i = self._G.add_getitem(travel, 'id')
        eq, _ = self._G.add_combining_operation('{0} = {1}', 'ids', i, one_id)
        n = self._G.add_filter(travel, eq, direct=True)
        return SpecificObjectQuery._spawn(self, n, obj, single=True)

    def _select_product(self, attr, want_single):
        attr = self._data.singular_name(attr)
        n = self._G.add_getproduct(self._node, attr)
        return ProductAttributeQuery._spawn(self, n, single=want_single, factor_name=attr)

    def _select_attribute(self, attr, want_single):
        """
        # obj['factor'], obj.factor
        fetches an attribute from the object and cuts off any other functions after that
        if the factor is not found in the object then the nearest corresponding object is used.
        e.g. `exposure.mjd` returns the mjd held by exposure
             `run.mjd` returns the mjd by a run's exposure (still only one mjd per run though)
             `run.cnames` returns the cname of each target in a run (this is a list per run)
        """
        if self._data.is_product(attr, self._obj):
            return self._select_product(attr, want_single)
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
        obj['factora', 'obj.factorb']
        """
        attrs = []
        names = []
        for item in items:
            if isinstance(item, SpecificObjectQuery):
                item = item._precompile()
            if isinstance(item, AttributeQuery):
                attrs.append(item)
                names.append(f"{item._names[0]}")
            else:
                new = self.__getitem__(item)
                if isinstance(new, SpecificObjectQuery):
                    new = new._precompile()
                if isinstance(new, list):
                    for a in new:
                        if a._factor_name not in names or a._factor_name is None:
                            names.append(f"{a._names[0]}")
                            attrs.append(a)
                elif new._factor_name not in names or new._factor_name is None:
                    names.append(f"{new._names[0]}")
                    attrs.append(new)
        force_plurals = [not a._single for a in attrs]
        is_products = [a._is_products[0] for a in attrs]
        n = self._G.add_results_table(self._node, [a._node for a in attrs], force_plurals, dropna=[self._node])
        return TableQuery._spawn(self, n, names=names, is_products=is_products, attrs=attrs)

    def _traverse_to_relative_object(self, obj, index):
        """
        obj.relative_path
        traversal, expands
        traverses by a specifically labelled relation instead of a name of an object
        e.g. l1stackedspectra.adjunct_spectrum will get the spectrum in the other arm
        CYPHER: (from:From)-[r]->(to:To) WHERE r.id = ...
        the last relationship will have a variable name:
            (Exposure)-->()-[r]->(Run)
            (Run)<-[r]-()<--(Exposure)
        """
        obj, obj_singular = self._normalise_object(obj)
        singular_name = self._data.singular_name(index)
        path, single = self._get_path_to_object(obj, False)
        if not single and singular_name == index:
            raise SyntaxError(f"Relative index `{index}` is plural relative to `{self._obj}`.")
        n = self._G.add_traversal(self._node, path, obj, False)
        relation_id = self._G.add_getitem(n, 'relation_id', 1)
        name = self._G.add_parameter(singular_name)
        eq, _ = self._G.add_scalar_operation(relation_id, f'{{0}} = {name}', 'rel_id')
        f = self._G.add_filter(n, eq, direct=True)
        return SpecificObjectQuery._spawn(self, f, obj, single=single)

    def _filter_by_relative_index(self):
        """
        obj1.obj2s['relative_id']
        filter, relative id, shrinks, destructive
        filters based on the relations between many `obj2`s and the `obj1`
        e.g. `ob.plate_exposures.fibre_exposures.single_spectra['red']` returns the red spectra
        the relative_id are explicitly stated in the database structure
        """
        raise NotImplementedError

    def _getitems(self, items, by_getitem):
        if not all(isinstance(i, (str, float, int, AttributeQuery)) for i in items):
            raise TypeError(f"Cannot index by non str/float/int/AttributeQuery values")
        if all(self._data.is_valid_name(i) or isinstance(i, AttributeQuery) for i in items):
            return self._make_table(*items)
        if any(self._data.is_valid_name(i) for i in items):
            raise SyntaxError(f"You may not mix filtering by id and building a table with attributes")
        # go back and be better
        return self._previous._traverse_by_object_indexes(self._obj, items)

    def _getitem(self, item, by_getitem):
        """
        item can be an id, a factor name, a list of those, a slice, or a boolean_mask
        """
        if isinstance(item, (tuple, list)):
            return self._getitems(item, by_getitem)
        elif isinstance(item, slice):
            return self._slice(item)
        elif isinstance(item, AttributeQuery):
            return self._filter_by_mask(item)  # a boolean_mask
        elif not isinstance(item, str):
            return self._previous._traverse_by_object_index(self._obj, item)
        elif item == '*':
            all_factors = self._get_all_factors_table()
            return self._getitems(all_factors, by_getitem)
        elif item == '**':
            all_factors = self._data.class_hierarchies[self._obj].products_and_factors
            return self._getitems(all_factors, by_getitem)
        else:
            try:
                return self._select_or_traverse_to_attribute(item)
            except (KeyError, ValueError):
                if '.' in item:  # split the parts and parse
                    try:
                        obj, attr = item.split('.')
                        return self.__getitem__(obj).__getitem__(attr)
                    except ValueError:
                        raise ValueError(f"{item} cannot be parsed as an `obj.attribute`.")
                try: # try assuming its an object
                    obj, single = self._normalise_object(item)
                    if obj == self._obj:
                        return self
                    return self._traverse_to_specific_object(obj, single)
                except (KeyError, ValueError):  # assume its an index of some kind
                    singular = self._data.singular_name(item)
                    if singular in self._data.relative_names:  # if it's a relative id
                        if by_getitem:  # filter by relative relation: ob.runs['red'] gets red runs
                            if singular != item:
                                raise SyntaxError(f"Filtering by relative index `{item}` must use its singular name `{singular}`")
                            print(self._data.relative_names[singular])
                            if self._previous._obj not in self._data.relative_names[singular]:
                                raise PathError(f"There are no `{singular}` `{self._obj}` of `{self._previous._obj}`")
                            return self._previous._traverse_to_relative_object(self._obj, item)
                        else:  # l1singlespectrum.adjunct gets the adjunct spectrum of a spectrum
                            try:
                                relation = self._data.relative_names[singular][self._obj]
                            except KeyError:
                                raise PathError(f"`{self._obj}` has no relative relation called `{singular}`")
                            return self._traverse_to_relative_object(relation.node.__name__, item)
                    elif by_getitem:  # otherwise treat as an index
                        return self._previous._traverse_by_object_index(self._obj, item)
                    else:
                        raise KeyError(f"Unknown attribute `{item}`")

    def _getitem_handled(self, item, by_getitem):
        try:
            return self._getitem(item, by_getitem)
        except (KeyError, ValueError, NetworkXNoPath, CardinalityError) as e:
            # self._data.autosuggest(item, self._obj)
            raise e

    def __getattr__(self, item):
        return self._getitem_handled(item, False)

    def __getitem__(self, item):
        return self._getitem_handled(item, True)

    def __eq__(self, other):
        return self._select_attribute('id', True).__eq__(other)


class DependentObjectQuery(SpecificObjectQuery):
    def _traverse_to_specific_object(self, obj, want_single, not_generic=False):
        specifics = []
        for specific_obj, subquery in self._subqueries.items():
            specifics.append(subquery._traverse_to_specific_object(obj, want_single, not_generic))
        paths = {s._obj: list(self._G.G.in_edges(s._node, data=True))[0][-1]['statement'].path for s in specifics}
        n = self._G.add_union_traversal(self._node, paths, obj, single=False)
        return TemplateObjectQuery._spawn(self, n, obj=obj, subqueries={s._obj: s for s in specifics}, single=want_single)

    def _traverse_to_template_object(self, obj, want_single):
        options = self._data.expand_template_object(obj)
        specifics = {}
        for specific_obj, subquery in self._subqueries.items():
            paths = []
            for option in options:
                obj_path = subquery._get_path_to_object(option, False, True)[-1]
                if specific_obj in obj_path:
                    paths.append(obj_path)
            if len(paths) == 0:  # does rely on the dependency, so accept all
                specifics[specific_obj] = subquery._traverse_to_template_object(obj, want_single)
            elif want_single:
                if len(paths) != 1:
                    raise CardinalityError(f"traversing from {self} to {obj} is ambiguous")
                # match the only path
                # e.g. l1singlespectrum.noss.l1single
                specifics[specific_obj] = subquery._traverse_to_specific_object([o for o in options if o in paths[0]][0], want_single)
            else:
                specifics[specific_obj] = subquery._traverse_to_template_object(obj, want_single)
        # specifics is a dictionary of {specific_obj: query}
        specifics = [s for q in specifics.values() for s in q._subqueries.values()]
        paths = {s._obj: list(self._G.G.in_edges(s._node, data=True))[0][-1]['statement'].path for s in specifics}
        n = self._G.add_union_traversal(self._node, paths, obj, single=False)
        return TemplateObjectQuery._spawn(self, n, obj=obj, subqueries={s._obj: s for s in specifics}, single=want_single)


    def _traverse_to_dependent_object(self, obj, want_single):
        return super()._traverse_to_dependent_object(obj, want_single)


class TemplateObjectQuery(SpecificObjectQuery):
    def _filter_by_type(self, htype: str):
        """
        Choose from the generic options by htype
        """
        htype = self._normalise_object(htype)
        types = {self._data.class_hierarchies[self._normalise_object(k)]: q for k, q in self._subqueries.items()}
        matched = {t: q for t, q in types.items() if issubclass(t, htype) or issubclass(htype, t)}
        if len(matched) == 0:
            raise SyntaxError(f"No objects of type `{htype}` found in {self._obj}")
        elif len(matched) == 1:
            return matched[list(matched.keys())[0]]  # return the only matched query
        else:
            specifics = list(matched.values())
            paths = {s._obj: self._G.G.edges[(self._node, s._node)]['statement'].path for s in specifics}
            n = self._G.add_union_traversal(self._node, paths, self._obj, single=False)
            return BaseObjectQuery._spawn(self, n, obj=self._obj, subqueries={s.__name__: s for s in matched}, single=False)



class Query(BaseObjectQuery):
    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: 'BaseQuery' = None, obj: str = None, start=None) -> None:
        super().__init__(data, G, node, previous, obj, start, 'start')

    def _compile(self):
        raise NotImplementedError(f"{self.__class__} is not compilable")

    def _traverse_to_specific_object(self, obj, want_single=False, not_generic=False):
        if want_single:
            raise CardinalityError(f"Cannot start query with a single object `{obj}`")
        obj, single = self._normalise_object(obj)
        if self._data.is_generic_object(obj) and not not_generic:
            return self._traverse_to_generic_object(obj, False)
        n = self._G.add_start_node(obj)
        return SpecificObjectQuery._spawn(self, n, obj, single=False)

    def _traverse_by_object_index(self, obj, index):
        obj, single = self._normalise_object(obj)
        name = self._G.add_parameter(index)
        travel = self._G.add_start_node(obj)
        i = self._G.add_getitem(travel, 'id')
        eq, _ = self._G.add_scalar_operation(i, f'{{0}} = {name}', f'id={index}')
        n = self._G.add_filter(travel, eq, direct=True)
        return SpecificObjectQuery._spawn(self, n, obj, single=True)

    def _traverse_by_object_indexes(self, obj, indexes: List):
        param = self._G.add_parameter(indexes)
        one_id = self._G.add_unwind_parameter(self._node, param)
        travel = self._G.add_start_node(obj, one_id)
        i = self._G.add_getitem(travel, 'id')
        eq, _ = self._G.add_combining_operation('{0} = {1}', 'ids', i, one_id)
        n = self._G.add_filter(travel, eq, direct=True)
        return SpecificObjectQuery._spawn(self, n, obj, single=True)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __getattr__(self, item):
        try:
            if self._data.is_singular_name(item):
                raise CardinalityError(f"Cannot start a query with a single object `{item}`")
            obj = self._get_object_of(item)[0]
            obj = self._data.plural_name(obj)
            return self._traverse_to_specific_object(obj, False)._select_attribute(item, True)
        except (KeyError, ValueError):
            return self._traverse_to_specific_object(item, False)


class AttributeQuery(BaseQuery):
    expect_one_column = True

    def __repr__(self):
        return f'<{self.__class__.__name__}({self._obj}.{self._factor_name})>'

    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: Union['Query', 'AttributeQuery', 'SpecificObjectQuery'] = None,
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
        if isinstance(other, SpecificObjectQuery):
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

    def __iadd__(self, other):
        raise TypeError

    def _precompile(self) -> 'TableQuery':
        if self._index_node == 'start':
            index = self._G.start
        else:
            index = self._index_node
        r = self._G.add_results_table(index, [self._node], [not self._single], dropna=[self._node])
        return AttributeQuery._spawn(self, r, self._obj, index, self._single, factor_name=self._factor_name)


class ProductAttributeQuery(AttributeQuery):

    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: Union['Query', 'AttributeQuery', 'ObjectQuery'] = None,
                 obj: str = None, start: Query = None, index_node=None, single=False, factor_name: str = None, *args, **kwargs) -> None:
        super().__init__(data, G, node, previous, obj, start, index_node, single, factor_name, is_product=[True], *args, **kwargs)

    def _perform_arithmetic(self, op_string, op_name, other=None, expected_dtype=None):
        raise TypeError(f"Binary data products cannot be operated upon. "
                        f"This is because they are not stored in the database")


class TableQuery(BaseQuery):
    def __init__(self, data: 'Data', G: QueryGraph = None, node=None,
                 previous: Union['Query', 'AttributeQuery', 'SpecificObjectQuery'] = None, obj: str = None,
                 start: Query = None, index_node=None,
                 single=False, attr_queries=None, names=None, *args, **kwargs) -> None:
        super().__init__(data, G, node, previous, obj, start, index_node, single, names, *args, **kwargs)
        self._attr_queries = attr_queries



class ListAttributeQuery(AttributeQuery):
    pass
