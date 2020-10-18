from collections import defaultdict
from copy import copy as copy
from typing import Union, Any

from .hierarchy import *
from .factor import *
from .dissociated import *
from .branchlogic import *
from .boolean import *
from ..utilities import quote


class Node:
    def __init__(self, name, label, **properties):
        self.label = label
        self.properties = properties
        self.name = name

    def identify(self, idvalue):
        self.properties['id'] = idvalue

    def __repr__(self):
        if self.properties:
            properties = ''
            for k, v in self.properties.items():
                properties += f'{k}: {quote(v)}'
            return f"({self.name}:{self.label} {{{properties}}})"
        return f"({self.name}:{self.label})"


class Path:
    def __init__(self, *path):
        if len(path) == 1:
            self.nodes = path
            self.directions = []
        elif not len(path) % 2:
            raise RuntimeError(f"Path expects input as [Node, <-, Node, <-, Node]")
        else:
            self.nodes, self.directions = path[::2], path[1::2]
        self.path = path

    def __repr__(self):
        s = ''.join(map(str, self.path))
        return s

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.nodes[item]
        else:
            return self.nodes[self.names.index(item)]

    @property
    def names(self):
        return [n.name for n in self.nodes]

    def extend(self, direction, path: 'Path'):
        self.path += [direction] + path.nodes


class Generator:
    def __init__(self):
        self.node_counter = defaultdict(int)
        self.property_name_counter = defaultdict(int)

    def node(self, label, name=None, **properties):
        if name is None:
            self.node_counter[label] += 1
            return Node(''.join([label.lower(), str(self.node_counter[label] - 1)]), label, **properties)
        else:
            return Node(name, label, **properties)

    def property_list(self, property_name):
        self.property_name_counter[property_name] += 1
        return ''.join([property_name, str(self.property_name_counter[property_name] - 1)])


class BaseQuery:
    """A Query which consists of a root path, where conditions"""
    def __init__(self, root: Path = None, conditions: List[str] = None):
        self.root = root
        self.conditions = conditions


class Branch(BaseQuery):
        """
        Branches are paths which are attached to other queries in the WHERE EXISTS {...} clause.
        They have no effect on the return value/node since they dont return anything themselves.
        Args:
            root: The path from which the query extends
            conditions: The conditions for the root path
        """


class Predicate(BaseQuery):
    """
    Predicates are parts of a query which act like sub-queries.
    They are run before the main query and return collected unordered distinct node properties.
    Predicates cannot return nodes
    Args:
        root: The path from which the query extends
        conditions: The conditions for the root path
        exist_branches: Paths which must exist with some logic between them of |&^ can be nested tuples
            e.g. [(path1, 'or', path2), 'and', (path3, 'xor', path4), 'and', path5]
        return_properties: The pairs of (node, property_name) to return
    """
    def __init__(self, root: Path = None,
                 conditions:  List[Union[List[str], str]] = None,
                 exist_branches: List[Union[List[str, Branch], str, Branch]] = None,
                 return_properties: List[Tuple[Node, str]] = None):
        super(Predicate, self).__init__(root, conditions)
        self.return_properties = return_properties
        for node, prop in self.return_properties:
            if node not in self.root.nodes:
                raise ValueError(f"A return_property {node}.{prop} references a node that does not exist in the root")
        if self.root is None and self.return_properties:
            raise ValueError('There must be a root to return_properties from')
        self.exist_branches = [] if exist_branches is None else exist_branches

    @staticmethod
    def simplify_logic(logic: List[Union[Any, str, Tuple]]):
        """
        Turn a list of variable, operation, and brackets (represented by a tuple) into something simpler
        """



    def make_neo4j_statement(self, generator):
        match = f'MATCH {self.root}'



        # exists = []
        # for i, branch in enumerate(self.exist_branches):
        #     if isinstance(branch, Branch):
        #         exists.append(f'EXISTS {{ {branch} }}')
        #     elif branch in ['and', 'or', 'xor']:
        #         exists.append(f' {branch} ')
        #     elif isinstance(branch,)
        #     else:
        #         raise ValueError(f"{branch} is not a valid Branch or logic operation ('and', 'or', 'xor')")
        exists = '\n'.join(exists)
        returns = self.make_neo4j_with(generator)
        return '\n'.join([match, 'WHERE ' + exists, returns])

    def make_neo4j_with(self, generator):
        varnames = []
        for node, prop in self.return_properties:
            varname = generator.property_list(prop)
            varnames.append(f"{node.name}.{prop} as {varname}")
        return "with DISTINCT " + ', '.join(varnames)


class FullQuery(Predicate):
    """
    A FullQuery holds all the relevant information about how to construct any DB query.
    The one restriction is that there may not be nested EXISTS statments.
    This is done by requiring the use of exist_branches and predicates.
    Args:
        root: The path from which the query extends
        conditions: The conditions for the root path
        exist_branches: Paths which must exist with some logic between them of |&^ can be nested tuples
            e.g. [(path1, '|', path2), '&', (path3, '^', path4), '&', path5]
        return_nodes: The nodes to return from the query
        return_properties: The pairs of (node, property_name) to return
    """
    def __init__(self, root: Path = None, conditions: List[str] = None,
                 exist_branches: List[Branch] = None,
                 predicates: List[Union[List[str, Predicate], str, Predicate]]  = None,
                 return_nodes: List[Node] = None,
                 return_properties: List[Tuple[Node, str]]  = None):
        super(FullQuery, self).__init__(root, conditions, exist_branches, return_properties)
        self.return_nodes = return_nodes
        for node in self.return_nodes:
            if node not in self.root.nodes:
                raise ValueError(f"A return_node {node} references a node that does not exist in the root")
        self.predicates = [] if predicates is None else predicates

    @property
    def has_root(self):
        return self.root is not None

    @property
    def only_has_root(self):
        return self.has_root and not self.predicates and not self.exist_branches and not self.return_properties

    def __repr__(self):
        props = defaultdict(list)
        for k, v in self.return_properties:
            props[k.name].append(v)
        props = dict(props)
        return f"<root={self.root} exist={len(self.exist_branches)} predicates={len(self.predicates)} properties={props}>"

    def make_neo4j_query(self, generator):
        predicate_statements = []
        for predicate in self.predicates:
            predicate_statements.append(predicate.make_neo4j_statement(generator))




class Handler:
    def hierarchy_of_factor(self, factor_name: str) -> str:
        raise NotImplementedError

    def path(self, start, end) -> Path:
        raise NotImplementedError

    def _filter_by_address(self, parent, address):
        if not isinstance(parent, (HeterogeneousHierarchyFrozenQuery,
                                   HomogeneousHierarchyFrozenQuery)):
            raise TypeError(f"Addresses can only filter heterogeneous or homogeneous hierarchies."
                            f"e.g. data[address] or data.runs[address]")
        cls = parent.__class__
        query = copy(parent.query)  # type: FullQuery
        for factor, value in address.items():
            h = self.hierarchy_of_factor(factor)
            if h not in query.root:
                long_path = self.path(query.root.nodes[0], h)
                short_path = query.root.merge(long_path)
                query.exist_branches.append(short_path)
            node = query.root[h].name
            query.conditions.append(f'{node}.{factor} = {quote(value)}')
        return cls(self, query, parent)

    def _filter_by_boolean(self, parent, boolean):
        raise NotImplementedError

    def _filter_by_identifier(self, parent, identifier):
        raise NotImplementedError

    def _get_single_factor(self, parent: HierarchyFrozenQuery, factor_name: str) -> SingleFactorFrozenQuery:
        query = copy(parent.query)
        h = self.hierarchy_of_factor(factor_name)
        if h not in query.root:
            long_path = self.path(query.root.nodes[0], h)
            short_path = query.root.merge(long_path)
            query.exist_branches.append(short_path)
        query.return_properties.append((query.root[h], factor_name))  # append pair of node, prop
        return SingleFactorFrozenQuery(self, query, parent)

    def _get_plural_factor(self, parent: , factor_name):
        raise NotImplementedError

    def _get_single_hierarchy(self, parent, hierarchy_name):
        raise NotImplementedError

    def _get_plural_hierarchy(self, parent, hierarchy_name):
        raise NotImplementedError

    def _equality(self, parent, other, negate=False):
        raise NotImplementedError

    def _compare(self, parent, other, operation):
        raise NotImplementedError

    def _combine(self, parent, other, operation):
        raise NotImplementedError