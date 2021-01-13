from collections import defaultdict
from copy import deepcopy
from typing import List, Union, Any, Type

import py2neo

from .common import FrozenQuery
from .factor import SingleFactorFrozenQuery, ColumnFactorFrozenQuery, RowFactorFrozenQuery, TableFactorFrozenQuery
from .query import AmbiguousPathError, FullQuery
from .query_objects import Collection, Path, Condition
from ..hierarchy import Hierarchy, Multiple
from ..utilities import quote


class HierarchyFrozenQuery(FrozenQuery):
    def __getitem__(self, item):
        raise NotImplementedError

    def __getattr__(self, item):
        raise NotImplementedError


class HeterogeneousHierarchyFrozenQuery(HierarchyFrozenQuery):
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
        label = hier.__name__
        start = self.handler.generator.node(label)
        root = [Path(start)]
        return HomogeneousHierarchyFrozenQuery(self.handler, FullQuery(root), hier, self)

    def _get_plural_factor(self, factor_name):
        hierarchy_name, factor_name, singular_name = self.handler.hierarchy_of_factor(factor_name)
        return self._get_plural_hierarchy(hierarchy_name)._get_plural_factor(factor_name)


class DefiniteHierarchyFrozenQuery(HierarchyFrozenQuery):
    SingleFactorReturnType = None

    def __init__(self, handler, branch: FullQuery, hierarchy: Type[Hierarchy], parent: 'FrozenQuery'):
        super().__init__(handler, branch, parent)
        self._hierarchy = hierarchy

    def _prepare_query(self):
        query = super(DefiniteHierarchyFrozenQuery, self)._prepare_query()
        indexer = self.handler.generator.node()
        query.branches[Path(self.branch.current_node, '<-[:INDEXES]-', indexer)].append(indexer)
        query.returns += [self.branch.current_node, indexer]
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
            h = self._process_result_row(row, self._hierarchy)
            results.append(h)
        return results

    def node_implies_plurality_of(self, end):
        start = self._hierarchy.singular_name.lower()
        multiplicity, direction, path, number = self.data.node_implies_plurality_of(start, end)
        path = [self.data.singular_hierarchies[n].__name__ for n in path]
        if direction == 'above':
            arrow = '<--'
        elif direction == 'below':
            arrow = '-->'
        else:
            raise ValueError(f"direction {direction} not known")
        nodes = self.handler.generator.nodes(*path[1:])
        node_path = [self.branch.current_node]
        for node in nodes:
            node_path += [arrow, node]
        path = Path(*node_path)
        return multiplicity, path, number

    def _get_plural_hierarchy(self, name):
        query = deepcopy(self.branch)
        multiplicity, path, number = self.node_implies_plurality_of(name)
        # dont check for multiplicity here, since plural is requested anyway
        query.matches.append(path)
        h = self.handler.data.singular_hierarchies[name]
        return HomogeneousHierarchyFrozenQuery(self.handler, query, h, self)

    def _get_factor_query(self, *names):
        query = deepcopy(self.branch)
        multiplicities = []
        numbers = []
        for name in names:
            is_singular_name = self.data.is_singular_name(name)
            hierarchy_name, factor_name, singular_name = self.handler.hierarchy_of_factor(name, self._hierarchy)
            friendly_name = '_'.join(name.split('.'))
            if hierarchy_name == self._hierarchy.name:
                if not (singular_name in self._hierarchy.factors or singular_name == self._hierarchy.idname):
                    raise KeyError(f"{self} does not have factor {singular_name}")
                prop = query.current_node.__getattr__(singular_name)
                prop.alias = friendly_name
                multiplicity = False
                number = 1
                if not is_singular_name:
                    prop = Collection(prop, friendly_name)
            else:
                multiplicity, path, number = self.node_implies_plurality_of(hierarchy_name)
                prop = path.nodes[-1].__getattr__(singular_name)
                if not is_singular_name:
                    prop = Collection(prop, friendly_name)
                query.branches[path].append(prop)
            query.returns.append(prop)
            multiplicities.append(multiplicity)
            numbers.append(number)
        return query, multiplicities, numbers

    def _get_plural_factor(self, name):
        singular_name = self.data.singular_name(name)
        query, multiplicities, numbers = self._get_factor_query(singular_name)
        if self.data.is_singular_name(name) and multiplicities[0]:
            plural = self.data.plural_name(singular_name)
            raise AmbiguousPathError(f"{self} has multiple {plural}, you need to explicitly pluralise them.")
        return ColumnFactorFrozenQuery(self.handler, query, [name], numbers, self)

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
        query, multiplicities, numbers = self._get_factor_query(*keys)
        expected_multi = [k for m, k in zip(multiplicities, keys) if self.data.is_singular_name(k) and m]
        if expected_multi:
            plurals = [self.data.plural_name(i) for i in expected_multi]
            raise AmbiguousPathError(f"Each {self._hierarchy} in {self} has multiple `{', '.join(plurals)}`, you need to explicitly pluralise them.")
        return query, return_keys, numbers

    def _get_single_factor_query(self, item):
        query, multiplicities, numbers = self._get_factor_query(item)
        if multiplicities[0] and self.data.is_singular_name(item):
            plural = self.data.plural_name(item)
            raise AmbiguousPathError(f"Each `{self._hierarchy.singular_name}` in `{self}` has multiple `{plural}`, you need to explicitly use `{plural}`.")
        return self.SingleFactorReturnType(self.handler, query, [item], numbers, self)

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
    SingleFactorReturnType = SingleFactorFrozenQuery

    def __init__(self, handler, branch: FullQuery, hierarchy: Type[Hierarchy], identifier: Any, parent: 'FrozenQuery'):
        super().__init__(handler, branch, hierarchy, parent)
        self._identifier = identifier

    def __getattr__(self, item):
        if self.data.is_singular_name(item) and self.data.is_factor_name(item):
            return self._get_singular_factor(item)
        return super().__getattr__(item)

    def _get_factor_table_query(self, item):
        query, headers, numbers =  super()._get_factor_table_query(item)
        return RowFactorFrozenQuery(self.handler, query, item, numbers, headers, self)

    def __repr__(self):
        if self._identifier is None:
            return f'{self.parent}.{self._hierarchy.singular_name}'
        return f'{self.parent}[{quote(self._identifier)}]'

    def _get_singular_hierarchy(self, name):
        query = deepcopy(self.branch)
        multiplicity, path, number = self.node_implies_plurality_of(name)
        if multiplicity:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"You have requested a single {name} but {self} has multiple {plural}. Use .{plural}")
        query.matches.append(path)
        h = self.handler.data.singular_hierarchies[name]
        return SingleHierarchyFrozenQuery(self.handler, query, h, None, self)

    def _get_singular_factor(self, name):
        query, multiplicities, numbers = self._get_factor_query(name)
        if multiplicities[0]:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"{self} has multiple {name}s. Use {plural} instead")
        return SingleFactorFrozenQuery(self.handler, query, name, numbers, self)

    def _post_process(self, result: py2neo.Cursor):
        rows = super()._post_process(result)
        if len(rows) != 1:
            idents = defaultdict(list)
            for frozen in self._traverse_frozenquery_stages():
                if isinstance(frozen, SingleHierarchyFrozenQuery):
                    idents[frozen._hierarchy.idname].append(frozen._identifier)
                elif isinstance(frozen, IdentifiedHomogeneousHierarchyFrozenQuery):
                    idents[frozen._hierarchy.idname] += frozen._identifiers
            if idents:
                d = {k: [i for i in v if i is not None] for k,v in idents.items()}
                d = {k: v for k, v in d.items() if len(v)}
                raise KeyError(f"One or more identifiers in {d} are not present in the database")
        return rows[0]


class HomogeneousHierarchyFrozenQuery(DefiniteHierarchyFrozenQuery):
    SingleFactorReturnType = ColumnFactorFrozenQuery

    def __repr__(self):
        return f'{self.parent}.{self._hierarchy.plural_name}'

    def _get_factor_table_query(self, item):
        query, headers, numbers =  super()._get_factor_table_query(item)
        return TableFactorFrozenQuery(self.handler, query, item, numbers, headers, self)

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

    def _filter_by_identifiers(self, identifiers: List[Union[str,int,float]]) -> 'IdentifiedHomogeneousHierarchyFrozenQuery':
        query = deepcopy(self.branch)
        ids = self.handler.generator.data(identifiers)
        query.matches.insert(-1, ids)  # give the query the data before the last match
        condition = Condition(query.current_node.id, '=', ids)
        if query.conditions is not None:
            query.conditions = query.conditions & condition
        else:
            query.conditions = condition
        return IdentifiedHomogeneousHierarchyFrozenQuery(self.handler, query, self._hierarchy, identifiers, self)

    def _filter_by_identifier(self, identifier: Union[str,int,float]):
        query = deepcopy(self.branch)
        condition = Condition(query.current_node.id, '=', identifier)
        if query.conditions is not None:
            query.conditions = query.conditions & condition
        else:
            query.conditions = condition
        if isinstance(self.parent, (HeterogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery)):
            return SingleHierarchyFrozenQuery(self.handler, query, self._hierarchy, identifier, self)
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
    def __init__(self, handler, branch: FullQuery, hierarchy: Type[Hierarchy], identifiers: List[Any], parent: 'FrozenQuery'):
        super().__init__(handler, branch, hierarchy, parent)
        self._identifiers = identifiers

    def __repr__(self):
        return f'{self.parent}.{self._hierarchy.plural_name}[{self._identifiers}]'

    def _post_process(self, result: py2neo.Cursor):
        r = super(IdentifiedHomogeneousHierarchyFrozenQuery, self)._post_process(result)
        ids = set(i.identifier for i in r)
        missing = [i for i in self._identifiers if i not in ids]
        if any(missing):
            raise KeyError(f"{self._hierarchy.idname} {missing} not found")
        return r
