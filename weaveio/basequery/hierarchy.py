from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Union, Any, Type

import py2neo

from .factor import SingleFactorFrozenQuery, ColumnFactorFrozenQuery
from .query import AmbiguousPathError, FullQuery, Node, Path, Condition
from ..address import Address
from .common import NotYetImplementedError, FrozenQuery
from ..hierarchy import Hierarchy, Multiple
from ..neo4j import parse_apoc_tree
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
            return self._get_set_of_factors()
        elif item in self.data.singular_factors:
            raise IndexError(f"Cannot return a single factor from a heterogeneous dataset")
        elif item in self.data.singular_hierarchies:
            raise IndexError(f"Cannot return a singular hierarchy without filtering first")
        else:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)

    def _get_plural_hierarchy(self, hierarchy_name):
        hier = self.data.singular_hierarchies[hierarchy_name]
        label = hier.__name__
        start = self.handler.generator.node(label)
        root = [Path(start)]
        return HomogeneousHierarchyFrozenQuery(self.handler, FullQuery(root), hier, self)

    def _get_plural_factor(self, factor_name):
        hierarchy_name = self.data.get_factor_hierarchy(factor_name).single_name.lower()



class DefiniteHierarchyFrozenQuery(HierarchyFrozenQuery):
    def __init__(self, handler, query: FullQuery, hierarchy: Type[Hierarchy], parent: 'FrozenQuery'):
        super().__init__(handler, query, parent)
        self._hierarchy = hierarchy

    def _prepare_query(self):
        query = super(DefiniteHierarchyFrozenQuery, self)._prepare_query()
        indexer = self.handler.generator.node()
        query.branches.append(Path(self.query.current_node, '<-[:INDEXES]-', indexer))
        query.returns += [self.query.current_node, indexer]
        return query

    def _process_result_row(self, row, nodetype):
        node, indexer = row
        inputs = {}
        for f in nodetype.factors:
            inputs[f] = node[f]
        inputs[nodetype.idname] = node[nodetype.idname]
        base_query = getattr(self, nodetype.plural_name)[node['id']]
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

    def __getattr__(self, item):
        if item in self.data.plural_factors or item in self.data.plural_idnames:
            name = self.data.singular_name(item)
            return self._get_plural_factor(name)
        elif item in self.data.singular_hierarchies:
            return self._get_singular_hierarchy(item)
        elif item in self.data.plural_hierarchies:
            name = self.data.singular_name(item)
            return self._get_plural_hierarchy(name)
        else:
            raise AttributeError(f"{self} has no attribute {item}")

    def node_implies_plurality_of(self, end):
        start = self._hierarchy.singular_name.lower()
        multiplicity, direction, path = self.data.node_implies_plurality_of(start, end)
        path = [self.data.singular_hierarchies[n].__name__ for n in path]
        if direction == 'above':
            arrow = '<--'
        elif direction == 'below':
            arrow = '-->'
        else:
            raise ValueError(f"direction {direction} not known")
        nodes = self.handler.generator.nodes(*path[1:])
        node_path = [self.query.current_node]
        for node in nodes:
            node_path += [arrow, node]
        path = Path(*node_path)
        return multiplicity, path

    def _get_plural_hierarchy(self, name):
        query = deepcopy(self.query)
        multiplicity, path = self.node_implies_plurality_of(name)
        # dont check for multiplicity here, since plural is requested anyway
        query.matches.append(path)
        h = self.handler.data.singular_hierarchies[name]
        return HomogeneousHierarchyFrozenQuery(self.handler, query, h, self)

    def _get_plural_factor_query(self, name):
        query = deepcopy(self.query)
        if name in self._hierarchy.factors or name == self._hierarchy.idname:
            query.returns.append(query.current_node.__getattr__(name))
            multiplicity = False
        else:
            hierarchy_names = self.data.factor_hierarchies[name]
            if len(hierarchy_names) > 1:
                raise AmbiguousPathError(f"The factor {name} is ambiguous when starting from {self}. "
                                         f"{name} has {len(hierarchy_names)} parents: {hierarchy_names}."
                                         f"Be explicit and choose one of them. "
                                         f"E.g. {self}.{hierarchy_names[0]}.{name}")
            else:
                hierarchy_name = hierarchy_names[0].singular_name.lower()
            multiplicity, path = self.node_implies_plurality_of(hierarchy_name)
            query.branches.append(path)
            query.returns.append(path.nodes[-1].__getattr__(name))
        return query, multiplicity

    def _get_plural_factor(self, name):
        query, multiplicity = self._get_plural_factor_query(name)
        return ColumnFactorFrozenQuery(self.handler, query, self)


class SingleHierarchyFrozenQuery(DefiniteHierarchyFrozenQuery):
    def __init__(self, handler, query: FullQuery, hierarchy: Type[Hierarchy], identifier: Any, parent: 'FrozenQuery'):
        super().__init__(handler, query, hierarchy, parent)
        self._identifier = identifier

    def __getattr__(self, item):
        if item in self.data.singular_factors or item in self.data.singular_idnames:
            return self._get_singular_factor(item)
        return super().__getattr__(item)

    def __repr__(self):
        if self._identifier is None:
            return f'{self.parent}.{self._hierarchy.singular_name}'
        return f'{self.parent}[{quote(self._identifier)}]'

    def _get_singular_hierarchy(self, name):
        query = deepcopy(self.query)
        multiplicity, path = self.node_implies_plurality_of(name)
        if multiplicity:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"You have requested a single {name} but {self} has multiple {plural}. Use .{plural}")
        query.matches.append(path)
        h = self.handler.data.singular_hierarchies[name]
        return SingleHierarchyFrozenQuery(self.handler, query, h, None, self)

    def _get_singular_factor(self, name):
        query, multiplicity = self._get_plural_factor_query(name)
        if multiplicity:
            plural = self.data.plural_name(name)
            raise AmbiguousPathError(f"{self} has multiple {name}s. Use .{plural} instead")
        return SingleFactorFrozenQuery(self.handler, query, self)

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
    def __repr__(self):
        return f'{self.parent}.{self._hierarchy.plural_name}'

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
            return self._filter_by_identifiers(item)
        return self._filter_by_identifier(item)

    def __getattr__(self, item):
        if item in self.data.singular_hierarchies:
            plural = self.data.plural_name(item)
            raise AmbiguousPathError(f"You have requested an ambiguous single {item}. Use .{plural}")
        if item in self.data.singular_factors or item in self.data.singular_idnames:
            plural = self.data.plural_name(item)
            raise AmbiguousPathError(f"{self} has multiple {plural}. Use .{plural} instead.")
        return super(HomogeneousHierarchyFrozenQuery, self).__getattr__(item)

    def _filter_by_identifiers(self, identifiers: List[Union[str,int,float]]) -> 'IdentifiedHomogeneousHierarchyFrozenQuery':
        query = deepcopy(self.query)
        ids = self.handler.generator.data(identifiers)
        query.matches.insert(-1, ids)  # give the query the data before the last match
        condition = Condition(query.current_node.id, '=', ids)
        if query.conditions is not None:
            query.conditions = query.conditions & condition
        else:
            query.conditions = condition
        return IdentifiedHomogeneousHierarchyFrozenQuery(self.handler, query, self._hierarchy, identifiers, self)

    def _filter_by_identifier(self, identifier: Union[str,int,float]) -> SingleHierarchyFrozenQuery:
        query = deepcopy(self.query)
        condition = Condition(query.current_node.id, '=', identifier)
        if query.conditions is not None:
            query.conditions = query.conditions & condition
        else:
            query.conditions = condition
        if isinstance(self.parent, (HeterogeneousHierarchyFrozenQuery, SingleHierarchyFrozenQuery)):
            return SingleHierarchyFrozenQuery(self.handler, query, self._hierarchy, identifier, self)
        else:
            raise AmbiguousPathError(f"`{self.parent}` is plural, to identify `{self}` by id, you must use "
                                     f"`{self}[[{quote(identifier)}]]` instead.")


class IdentifiedHomogeneousHierarchyFrozenQuery(HomogeneousHierarchyFrozenQuery):
    """
    An ordered duplicated list of hierarchies each identified by an id
    If an id appears more than once, it will be duplicated appropriately
    The list is ordered by id input order
    """
    def __init__(self, handler, query: FullQuery, hierarchy: Type[Hierarchy], identifiers: List[Any], parent: 'FrozenQuery'):
        super().__init__(handler, query, hierarchy, parent)
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
