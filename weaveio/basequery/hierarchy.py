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
        paths, hiers, startbase, endbase = self.handler.paths2hierarchy(hierarchy_name, plural=True)
        new = self.branch.handler.begin(endbase.__name__)
        return HomogeneousHierarchyFrozenQuery(self.handler, new, endbase, new.current_hierarchy, self)

    def _get_plural_factor(self, factor_name):
        pathdict, base = self.handler.paths2factor(factor_name, plural=True)
        begin = self.branch.handler.begin(base.__name__)
        new = begin.operate(f'{factor_name}', factor_name=begin.current_hierarchy[factor_name])
        return ColumnFactorFrozenQuery(self.handler, new, [factor_name], new.current_variables[0], None, self)


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
        node = row[0]
        inputs = {}
        for f in nodetype.factors:
            inputs[f] = node[f]
        inputs[nodetype.idname] = node[nodetype.idname]
        base_query = getattr(self.handler.begin_with_heterogeneous(), nodetype.plural_name)[node['id']]
        for p in nodetype.parents:
            if isinstance(p, Multiple):
                inputs[p.plural_name] = getattr(base_query, p.plural_name)
            else:
                try:
                    inputs[p.singular_name] = getattr(base_query, p.singular_name)
                except AmbiguousPathError:
                    inputs[p.singular_name] = getattr(base_query, p.plural_name)  # this should not have to be done
        h = nodetype(**inputs, do_not_create=True)
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

    def _get_plural_hierarchy(self, name):
        pathlist, endlist, starthier, endhier = self.handler.paths2hierarchy(name, plural=True, start=self.hierarchy_type)
        new = self.branch.traverse(*pathlist)
        return HomogeneousHierarchyFrozenQuery(self.handler, new, endhier, new.current_hierarchy, self)

    def _get_factor_query(self, names: Union[List[str], str], plurals: Union[List[bool], bool]) -> Tuple[Branch, List[CypherVariable]]:
        """
        Return the query branch, variables of a list of factor/product names
        We do this by grouping into the containing hierarchies and traversing each branch before collapsing
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        if not isinstance(plurals, (list, tuple)):
            plurals = [plurals]
        local = []
        remote = defaultdict(list)
        remote_paths = {}
        for name, plural in zip(names, plurals):
            pathsdict, basehier = self.handler.paths2factor(name, plural, self.hierarchy_type)
            if basehier == self.hierarchy_type:
                local.append((name, plural))
            else:
                remote_paths[basehier] = (pathsdict, plural)
                remote[basehier].append(name)

        variables = {}
        branch = self.branch
        branch = branch.operate(*[f'{{h}}.{name}' for name, plural in local], h=self.hierarchy_variable)
        for v, (k, _) in zip(branch.action.output_variables, local):
            variables[k] = v

        for basehier, factors in remote.items():
            paths = remote_paths[basehier][0].values()
            plural = remote_paths[basehier][1]
            travel = branch.traverse(*paths)
            operate = travel.operate(*[f'{{h}}.{name}' for name in factors], h=travel.current_hierarchy)
            if plural:
                branch = branch.collect([], [operate])
            else:
                branch = branch.collect([operate], [])
            for v, name in zip(operate.current_variables, factors):
                variables[name] = branch.action.transformed_variables[v]

        return branch, [variables[name] for name in names]

    def _get_plural_factor(self, name):
        branch, factor_variables = self._get_factor_query([name], [True])
        return ColumnFactorFrozenQuery(self.handler, branch, [name], factor_variables, None, self)

    def _get_factor_table_query(self, item) -> Tuple[Branch, List[str], List[CypherVariable]]:
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
        plurals = [not self.is_singular_name(i) for i in item]
        branch, factor_variables = self._get_factor_query(keys, plurals)
        return branch, return_keys, factor_variables

    def _get_single_factor_query(self, item):
        branch, factor_variables = self._get_factor_query([item], [False])
        return self.SingleFactorReturnType(self.handler, branch, [item], factor_variables, None, self)

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
        branch, return_keys, factor_variables = super()._get_factor_table_query(keys)
        return RowFactorFrozenQuery(self.handler, branch, keys, factor_variables, None, return_keys, self)

    def __repr__(self):
        if self._identifier is None:
            return f'{self.parent}.{self.hierarchy_type.singular_name}'
        return f'{self.parent}[{quote(self._identifier)}]'

    def _get_singular_hierarchy(self, name):
        pathlist, hierlist, startbase, endbase = self.handler.paths2hierarchy(name, plural=False, start=self.hierarchy_type)
        branch = self.branch.traverse(*pathlist)
        return SingleHierarchyFrozenQuery(self.handler, branch, endbase, branch.current_hierarchy, None, self)

    def _get_singular_factor(self, name):
        branch, factor_variables = self._get_factor_query(name, True)
        return SingleFactorFrozenQuery(self.handler, branch, name, factor_variables, None, self)

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
        branch, return_keys, factor_variables = super()._get_factor_table_query(item)
        return TableFactorFrozenQuery(self.handler, branch, item, factor_variables, None, return_keys, self)

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

    def _filter_by_identifiers(self, identifiers: List[Union[str, int, float]]) -> 'IdentifiedHomogeneousHierarchyFrozenQuery':
        idname = self.hierarchy_type.idname
        new = self.branch.add_data(identifiers)
        identifiers_var = new.current_variables[0]
        branch = new.filter('{h}.' + idname + ' in {identifiers}', h=self.hierarchy_variable, identifiers=identifiers_var)
        return IdentifiedHomogeneousHierarchyFrozenQuery(self.handler, branch, self.hierarchy_type, self.hierarchy_variable, identifiers, self)

    def _filter_by_identifier(self, identifier: Union[str, int, float]):
        idname = self.hierarchy_type.idname
        new = self.branch.add_data(identifier)
        identifier_var = new.current_variables[0]
        branch = new.filter('{h}.' + idname + ' = {identifier}', h=self.hierarchy_variable, identifier=identifier_var)
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
