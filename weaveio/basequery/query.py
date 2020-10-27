from collections import defaultdict
from typing import List, Union, Tuple, Any
from copy import deepcopy as copy

from weaveio.utilities import quote


class Node:
    def __init__(self, label, name=None, **properties):
        self.label = label
        self.properties = properties
        self.name = name

    def identify(self, idvalue):
        self.properties['id'] = idvalue

    def stringify(self, mentioned_nodes):
        if self in mentioned_nodes:
            return f'({self.name})'
        return str(self)

    def __repr__(self):
        name = '' if self.name is None else self.name
        if self.properties:
            properties = ''
            for k, v in self.properties.items():
                properties += f'{k}: {quote(v)}'
            return f"({name}:{self.label} {{{properties}}})"
        return f"({name}:{self.label})"

    def __eq__(self, other):
        return (self.label == other.label) and \
               ((self.name == other.name) or (self.name is None and other.name is None)) and \
               list(self.properties.items()) == list(other.properties.items())


class Path:
    def __init__(self, *path: Union[Node, str]):
        if len(path) == 1:
            self.nodes = path
            self.directions = []
        elif not len(path) % 2 and len(path) > 0:
            raise RuntimeError(f"Path expects input as [Node, <-, Node, <-, Node]")
        else:
            self.nodes, self.directions = path[::2], path[1::2]
        self.path = path

    def reversed(self):
        return Path(*['<--' if i == '-->' else '-->' if i == '<--' else i for i in self.path[::-1]])

    def __repr__(self):
        s = ''.join(map(str, self.path))
        return s

    def stringify(self, mentioned_nodes):
        s = ''.join([p if isinstance(p, str) else p.stringify(mentioned_nodes) for p in self.path])
        return s

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.nodes[item]
        else:
            return self.nodes[self.names.index(item)]

    @property
    def names(self):
        return [n.name for n in self.nodes]

    def __len__(self):
        return len(self.nodes)


class Generator:
    def __init__(self):
        self.node_counter = defaultdict(int)
        self.property_name_counter = defaultdict(int)

    def node(self, label, name=None, **properties):
        if name is None:
            self.node_counter[label] += 1
            return Node(label, ''.join([label.lower(), str(self.node_counter[label] - 1)]), **properties)
        else:
            return Node(label, name, **properties)

    def property_list(self, property_name):
        self.property_name_counter[property_name] += 1
        return ''.join([property_name, str(self.property_name_counter[property_name] - 1)])


class BaseQuery:
    """A Query which consists of a root path, where conditions"""
    def __init__(self, root: Path = None, branches: List[Path] = None, conditions: List[str] = None):
        self.root = Path() if root is None else root
        self.conditions = [] if conditions is None else conditions
        self.branches = [] if branches is None else branches

    def append_to_root(self, arrow: str, path: Path):
        query = copy(self)
        if not len(query.root):
            query.root = path
        else:
            query.root = Path(*query.root.path, arrow, *path.path)
        return query

    def merge_into_branches(self, path):
        query = copy(self)
        for n in query.root.nodes:
            if n == path.nodes[0]:
                break
        else:
            raise ValueError(f"There is no shared node between {path} and {self}")
        query.branches.append(path)
        return query


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
                 branches: List[Path] = None,
                 conditions:  List[Union[List[str], str]] = None,
                 exist_branches: List[Union[List[Union[str, Branch]], str, Branch]] = None,
                 return_properties: List[Tuple[Union[Node, str]]] = None):
        super(Predicate, self).__init__(root, branches, conditions)
        self.return_properties = [] if return_properties is None else return_properties
        for node, prop in self.return_properties:
            if node not in self.root.nodes:
                raise ValueError(f"A return_property {node}.{prop} references a node that does not exist in the root")
        if self.root is None and self.return_properties:
            raise ValueError('There must be a root to return_properties from')
        self.exist_branches = [] if exist_branches is None else exist_branches

    def add_exist_branches(self, *branches):
        query = copy(self)
        query.exist_branches += branches
        return query

    def identify_in_root(self, identity):
        query = copy(self)
        query.root.nodes[-1].identify(identity)
        return query

    def return_property(self, node, prop):
        query = copy(self)
        query.return_properties.append((node, prop))
        return query

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
    def __init__(self, root: Path = None, branches: List[Path] = None,
                 conditions: List[str] = None,
                 exist_branches: List[Branch] = None,
                 predicates: List[Union[List[Union[str, Predicate]], str, Predicate]]  = None,
                 return_nodes: List[Node] = None,
                 return_properties: List[Tuple[Union[Node, str]]]  = None):
        super(FullQuery, self).__init__(root, branches, conditions, exist_branches, return_properties)
        self.return_nodes = [] if return_nodes is None else return_nodes
        for node in self.return_nodes:
            if node not in self.root.nodes:
                raise ValueError(f"A return_node {node} references a node that does not exist in the root")
        self.predicates = [] if predicates is None else predicates


    def __repr__(self):
        props = defaultdict(list)
        for k, v in self.return_properties:
            props[k.name].append(v)
        props = dict(props)
        return f"<root={self.root} exist={len(self.exist_branches)} predicates={len(self.predicates)} properties={props}>"

    def to_branches(self):


    def return_node(self, node):
        query = copy(self)
        query.return_nodes.append(node)
        return query

    def to_neo4j(self, generator):
        predicate_statements = []
        for predicate in self.predicates:
            predicate_statements.append(predicate.make_neo4j_statement(generator))
        predicates = '\n\n'.join(predicate_statements)
        offshoots = ', ' + ', '.join([b.stringify(self.root.nodes) for b in self.branches]) if len(self.branches) else ''
        main = f'MATCH {self.root}{offshoots}'
        if len(self.conditions):
            wheres = '\nWHERE' + ', '.join(map(str, self.conditions))
        else:
            wheres = ''
        return_nodes = ', '.join([i.name for i in self.return_nodes])
        return_properties = ', '.join([f'{node.name}.{prop}' for node, prop in self.return_properties])
        comma = ', ' * ((len(return_properties) > 0) or (len(return_nodes) > 0))
        returns = f'{return_nodes}{comma}{return_properties}'
        return f'{predicates}\n{main}{wheres}\nRETURN {returns}'



class AmbiguousPathError(Exception):
    pass