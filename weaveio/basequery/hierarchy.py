from collections import defaultdict
from typing import List, Union, Any, Type, Tuple

import py2neo

from .common import FrozenQuery, AmbiguousPathError
from .factor import SingleFactorFrozenQuery, ColumnFactorFrozenQuery, RowFactorFrozenQuery, TableFactorFrozenQuery
from .tree import Branch, TraversalPath
from ..hierarchy import Hierarchy, Multiple
from ..utilities import quote
from ..writequery import CypherVariable


class HierarchyFrozenQuery(FrozenQuery):
    def __getitem__(self, item):
        raise NotImplementedError

    def __getattr__(self, item):
        raise NotImplementedError


class HeterogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
    """The start point for building queries"""
    executable = False

    def __repr__(self):
        return f'query("{self.data.rootdir}/")'

    def __getattr__(self, item):
        if item in self.data.plural_factors:
            return self._get_plural_factor(item)
        elif item in self.data.singular_factors:
            raise AmbiguousPathError(f"Cannot return a single factor from a heterogeneous dataset")
        elif item in self.data.singular_hierarchies:
            raise AmbiguousPathError(f"Cannot return a singular hierarchy without filtering first")
        else:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)

    def _get_plural_hierarchy(self, hierarchy_name) -> 'HomogeneousHierarchyFrozenQuery':
        hier = self.data.singular_hierarchies[hierarchy_name]
        multiplicity, number, path,  hier = self.handler.data.node_implies_plurality_of(self.branch.find_hierarchies()[-1], hier.singular_name)
        new = self.branch.traverse(path)
        return HomogeneousHierarchyFrozenQuery(self.handler, new, hier, self.current_hierarchy, self)

    def _get_plural_factor(self, factor_name):
        hierarchy_name, factor_name, singular_name = self.handler.hierarchy_of_factor(factor_name)
        return self._get_plural_hierarchy(hierarchy_name)._get_plural_factor(factor_name)


class DefiniteHierarchyFrozenQuery(HierarchyFrozenQuery):
    """The template class for hierarchy classes that are not heterogeneous i.e. they have a defined hierarchy type"""
    SingleFactorReturnType = None

    def __init__(self, handler, branch: Branch, hierarchy_type: Type[Hierarchy], hierarchy_variable: CypherVariable, parent: 'FrozenQuery'):
        super().__init__(handler, branch, parent)
        self.hierarchy_type = hierarchy_type
        self.hierarchy_variable = hierarchy_variable

    def _prepare_query(self):
        """Add a hierarchy node return statement"""
        query = super(DefiniteHierarchyFrozenQuery, self)._prepare_query()
        with query:
            query.returns(self.branch.find_hierarchies()[-1])
        return query

    def _process_result_row(self, row, nodetype):
        node, indexer = row
        inputs = {}
        for f in nodetype.factors:
            inputs[f] = node[f]
        inputs[nodetype.idname] = node[nodetype.idname]
        base_query = getattr(self.handler.begin_with_heterogeneous(), nodetype.plural_name)[node['id']]
        for p in nodetype.parents:
            if p.singular_name == nodetype.indexer:
                inputs[p.singular_name] = self._process_result_row([indexer, {}], p)
            elif isinstance(p, Multiple):
                inputs[p.plural_name] = getattr(base_query, p.plural_name)
            else:
                inputs[p.singular_name] = getattr(base_query, p.singular_name)
        h = nodetype(**inputs)
        h.add_parent_query(base_query)
        return h

    def _post_process(self, result: py2neo.Cursor):
        result = result.to_table()
        if len(result) == 1 and result[0] is None:
            return []
        results = []
        for row in result:
            h = self._process_result_row(row, self.hierarchy_type)
            results.append(h)
        return results

    def node_implies_plurality_of(self, end: str) -> Tuple[bool, TraversalPath, int, Type[Hierarchy]]:
        start = self.hierarchy_type.singular_name.lower()
        multiplicity, number, path, hier = self.data.node_implies_plurality_of(start, end)
        return multiplicity, path, number, hier

    def _get_plural_hierarchy(self, name):
        multiplicity, path, number, hier = self.node_implies_plurality_of(name)
        # dont check for multiplicity here, since plural is requested anyway
        new = self.branch.traverse(path)
        return HomogeneousHierarchyFrozenQuery(self.handler, new, hier, new.current_hierarchy, self)

    def _get_factor_query(self, *names) -> Tuple[Branch, List[CypherVariable], List[bool], List[int]]:
        multiplicities = []
        numbers = []
        factor_variables = []
        branch = self.branch
        for name in names:
            hierarchy_name, factor_name, singular_name = self.handler.hierarchy_of_factor(name, self.hierarchy_type)
            friendly_name = '_'.join(name.split('.'))
            if hierarchy_name != self.hierarchy_type.__name__:  # if it refers to another hierarchy
                multiplicity, path, number, _ = self.node_implies_plurality_of(hierarchy_name)
                branch = branch.traverse(path)
            else:
                multiplicity = False
                number = 1
                if not (singular_name in self.hierarchy_type.factors or singular_name == self.hierarchy_type.idname):
                    raise KeyError(f"{self} does not have factor {singular_name}")
            operated = branch.operate('{h}.' + f'{singular_name} as {friendly_name}', h=self.hierarchy_variable)
            if multiplicity:
                new = branch.collect([], [operated])  # collapse back to the original level
                factor_variable = new.action.transformed_variables[operated.current_variables[0]]
            else:
                new = operated
                factor_variable = new.current_variables[0]
            multiplicities.append(multiplicity)
            numbers.append(number)
            factor_variables.append(factor_variable)
        return branch, factor_variables, multiplicities, numbers

    def _get_plural_factor(self, name):
        singular_name = self.data.singular_name(name)
        branch, factor_variables, multiplicities, numbers = self._get_factor_query(singular_name)
        if self.data.is_singular_name(name) and multiplicities[0]:
            plural = self.data.plural_name(singular_name)
            raise AmbiguousPathError(f"{self} has multiple {plural}, you need to explicitly pluralise them.")
        return ColumnFactorFrozenQuery(self.handler, branch, [name], factor_variables, numbers, self)

    def _get_factor_table_query(self, item):
        """
        __getitem__ is for returning factors and ids
        There are three types of getitem input values:
        List: [[a, b]], where labelled table-like rows are output
        Tuple: [a, b], where a list of unlabelled dictionaries are output
        str: [a], where a single value is returned

        In all three cases, you still need to specify plural or singular forms.
        This allows you to have a row of n dimensional heterogeneous data.
        returns query and the labels (if any) for the table
        """
        if isinstance(item, tuple):  # return without headers
            return_keys = None
            keys = list(item)
        elif isinstance(item, list):
            keys = item
            return_keys = item
        elif item is None:
            raise TypeError("item must be of type list, tuple, or str")
        else:
            raise KeyError(f"Unknown item {item} for `{self}`")
        branch, factor_variables, multiplicities, numbers = self._get_factor_query(*keys)
        expected_multi = [k for m, k in zip(multiplicities, keys) if self.data.is_singular_name(k) and m]
        if expected_multi:
            plurals = [self.data.plural_name(i) for i in expected_multi]
            raise AmbiguousPathError(f"Each {self.hierarchy_type} in {self} has multiple `{', '.join(plurals)}`, you need to explicitly pluralise them.")
        return branch, return_keys, factor_variables, numbers

    def _get_single_factor_query(self, item):
        branch, factor_variables, multiplicities, numbers = self._get_factor_query(item)
        if multiplicities[0] and self.data.is_singular_name(item):
            plural = self.data.plural_name(item)
            raise AmbiguousPathError(f"Each `{self.hierarchy_type.singular_name}` in `{self}` has multiple `{plural}`, you need to explicitly use `{plural}`.")
        return self.SingleFactorReturnType(self.handler, branch, [item], factor_variables, numbers, self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._get_single_factor_query(item)
        return self._get_factor_table_query(item)

    def __getattr__(self, item):
        if self.data.is_plural_name(item) and self.data.is_factor_name(item):
            return self._get_plural_factor(item)
        elif item in self.data.singular_hierarchies:
            return self._get_singular_hierarchy(item)
        elif item in self.data.plural_hierarchies:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)
        else:
            raise AttributeError(f"{self} has no attribute {item}")


class SingleHierarchyFrozenQuery(DefiniteHierarchyFrozenQuery):
    """Contains only a single hierarchy type with an identifier"""
    SingleFactorReturnType = SingleFactorFrozenQuery

    def __init__(self, handler, branch: Branch, hierarchy_type: Type[Hierarchy], hierarchy_variable: CypherVariable,  identifier: Any, parent: 'FrozenQuery'):
        super().__init__(handler, branch, hierarchy_type, hierarchy_variable, parent)
        self._identifier = identifier

    def __getattr__(self, item):
        if self.data.is_singular_name(item) and self.data.is_factor_name(item):
            return self._get_singular_factor(item)
        return super().__getattr__(item)

    def _get_factor_table_query(self, keys):
        branch, return_keys, factor_variables, numbers = super()._get_factor_table_query(keys)
        return RowFactorFrozenQuery(self.handler, branch, keys, factor_variables, numbers, return_keys, self)

    def __repr__(self):
        if self._identifier is None:
            return f'{self.parent}.{self.hierarchy_type.singular_name}'
        return f'{self.parent}[{quote(self._identifier)}]'

    def _get_singular_hierarchy(self, name):
        multiplicity, path, number, hier = self.node_implies_plurality_of(name)
        if multiplicity:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"You have requested a single {name} but {self} has multiple {plural}. Use .{plural}")
        branch = self.branch.traverse(path)
        return SingleHierarchyFrozenQuery(self.handler, branch, hier, branch.current_hierarchy, None, self)

    def _get_singular_factor(self, name):
        branch, factor_variables, multiplicities, numbers = self._get_factor_query(name)
        if multiplicities[0]:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"{self} has multiple {name}s. Use {plural} instead")
        return SingleFactorFrozenQuery(self.handler, branch, name, factor_variables, numbers, self)

    def _post_process(self, result: py2neo.Cursor):
        rows = super()._post_process(result)
        if len(rows) != 1:
            idents = defaultdict(list)
            for frozen in self._traverse_frozenquery_stages():
                if isinstance(frozen, SingleHierarchyFrozenQuery):
                    idents[frozen.hierarchy_type.idname].append(frozen._identifier)
                elif isinstance(frozen, IdentifiedHomogeneousHierarchyFrozenQuery):
                    idents[frozen.hierarchy_type.idname] += frozen._identifiers
            if idents:
                d = {k: [i for i in v if i is not None] for k, v in idents.items()}
                d = {k: v for k, v in d.items() if len(v)}
                raise KeyError(f"One or more identifiers in {d} are not present in the database")
        return rows[0]


class HomogeneousHierarchyFrozenQuery(DefiniteHierarchyFrozenQuery):
    """A list of hierarchies of the same type that are not identified"""
    SingleFactorReturnType = ColumnFactorFrozenQuery

    def __repr__(self):
        return f'{self.parent}.{self.hierarchy_type.plural_name}'

    def _get_factor_table_query(self, item):
        branch, return_keys, factor_variables, numbers = super()._get_factor_table_query(item)
        return TableFactorFrozenQuery(self.handler, branch, item, factor_variables, numbers, return_keys, self)

    def __getitem__(self, item):
        """
        Returns a table of factor values or (if that fails) a filter by identifiers
        """
        try:
            return super(HomogeneousHierarchyFrozenQuery, self).__getitem__(item)
        except KeyError:
            if isinstance(item, (list, tuple)):
                disallowed_factors = [i for i in item if self.data.is_factor_name(i)]
                if disallowed_factors:
                    ids = list(set(item) - set(disallowed_factors))
                    raise SyntaxError(f"You cannot index factors and hierarchies at the same time. "
                                      f"Separate your queries for {ids} and `{disallowed_factors}`")
                return self._filter_by_identifiers(item)
            return self._filter_by_identifier(item)

    def __getattr__(self, item):
        if item in self.data.singular_hierarchies:
            plural = self.data.plural_name(item)
            raise AmbiguousPathError(f"{self} has multiple {plural}. Use .{plural} instead")
        if self.data.is_singular_name(item) and self.data.is_factor_name(item):
            plural = self.data.plural_name(item)
            raise AmbiguousPathError(f"{self} has multiple {plural}. Use .{plural} instead.")
        return super(HomogeneousHierarchyFrozenQuery, self).__getattr__(item)

    def _filter_by_identifiers(self, identifiers: List[Union[str, int, float]]) -> 'IdentifiedHomogeneousHierarchyFrozenQuery':
        idname = self.hierarchy_type.idname
        new = self.branch.add_data(identifiers)
        identifiers_var = new.current_variables[0]
        branch = self.branch.filter('{h}.' + idname + ' in {identifiers}', h=self.hierarchy_variable, identifiers=identifiers_var)
        return IdentifiedHomogeneousHierarchyFrozenQuery(self.handler, branch, self.hierarchy_type, self.hierarchy_variable, identifiers, self)

    def _filter_by_identifier(self, identifier: Union[str, int, float]):
        idname = self.hierarchy_type.idname
        new = self.branch.add_data(identifier)
        identifier_var = new.current_variables[0]
        branch = self.branch.filter('{h}.' + idname + ' = {identifier}', h=self.hierarchy_variable, identifier=identifier_var)
        if isinstance(self.parent, (HeterogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery)):
            return SingleHierarchyFrozenQuery(self.handler, branch, self.hierarchy_type, self.hierarchy_variable, identifier, self)
        else:
            raise AmbiguousPathError(f"`{self.parent}` is plural, to identify `{self}` by id, you must use "
                                     f"`{self}[[{quote(identifier)}]]` instead of "
                                     f"`{self}[{quote(identifier)}]`.")


class IdentifiedHomogeneousHierarchyFrozenQuery(HomogeneousHierarchyFrozenQuery):
    """
    An ordered duplicated list of hierarchies each identified by an id
    If an id appears more than once, it will be duplicated appropriately
    The list is ordered by id input order
    """
    def __init__(self, handler, branch: Branch, hierarchy_type: Type[Hierarchy], hierarchy_variable: CypherVariable, identifiers: List[Any], parent: 'FrozenQuery'):
        super().__init__(handler, branch, hierarchy_type, hierarchy_variable, parent)
        self._identifiers = identifiers

    def __repr__(self):
        return f'{self.parent}.{self.hierarchy_type.plural_name}[{self._identifiers}]'

    def _post_process(self, result: py2neo.Cursor):
        r = super(IdentifiedHomogeneousHierarchyFrozenQuery, self)._post_process(result)
        ids = set(i.identifier for i in r)
        missing = [i for i in self._identifiers if i not in ids]
        if any(missing):
            raise KeyError(f"{self.hierarchy_type.idname} {missing} not found")
        return r
