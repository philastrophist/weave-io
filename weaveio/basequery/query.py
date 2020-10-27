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
        mentioned_nodes.append(self)
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

    def __getattr__(self, item):
        return NodeProperty(self, item)


class NodeProperty:
    def __init__(self, node, property_name):
        self.node = node
        self.property_name = property_name
        self.name = f"{node.name}.{property_name}"

    def stringify(self, mentioned_nodes):
        n = self.node.stringify(mentioned_nodes)
        return f"{n}.{self.property_name}"

    def __repr__(self):
        return f"{self.stringify([])}"


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

    def nodes(self, *labels):
        return [self.node(l) for l in labels]

    def property_list(self, property_name):
        self.property_name_counter[property_name] += 1
        return ''.join([property_name, str(self.property_name_counter[property_name] - 1)])


class Condition:
    def __init__(self, a, comparison, b):
        self.a = a
        self.comparison = comparison
        self.b = b

    def stringify(self):
        a = self.a.stringify() if isinstance(self.a, Condition) else getattr(self.a, 'name', quote(str(self.a)))
        b = self.b.stringify() if isinstance(self.b, Condition) else getattr(self.b, 'name', quote(str(self.b)))
        return f"({a} {self.comparison} {b})"

    def __repr__(self):
        return self.stringify()

    def __and__(self, other):
        return Condition(self, 'and', other)

    def __or__(self, other):
        return Condition(self, 'or', other)

    def __eq__(self, other):
        return Condition(self, '==', other)

    def __ne__(self, other):
        return Condition(self, '<>', other)


class Exists:
    def __init__(self, path: Path):
        self.path = path


class BaseQuery:
    """A Query which consists of a root path, where conditions"""
    def __init__(self, matches: List[Path] = None, conditions: Condition = None):
        self.matches = [] if matches is None else matches
        self.conditions = conditions
        for i, path in enumerate(self.matches):
            if i > 0:
                if not any(n in self.matches[i-1].nodes for n in path.nodes):
                    raise ValueError(f"A list of matches must have overlapping nodes")

    @property
    def matches(self):
        return self._matches

    @matches.setter
    def matches(self, value):
        if not all(isinstance(i, Path) for i in value) and value is not None:
            raise TypeError("matches must be a list of paths")
        self._matches = value

    @property
    def conditions(self):
        return self._conditions

    @conditions.setter
    def conditions(self, value):
        if not isinstance(value, Condition) and value is not None:
            raise TypeError(f"conditions must be of type Condition")
        self._conditions = value

    @property
    def current_node(self):
        return self.matches[-1].nodes[-1]



class Branch(BaseQuery):
        """
        Branches are paths which are attached to other queries in the WHERE EXISTS {...} clause.
        They have no effect on the return value/node since they dont return anything themselves.
        """


class Predicate(BaseQuery):
    """
    Predicates are parts of a query which act like sub-queries.
    They are run before the main query and return collected unordered distinct node properties.
    Predicates cannot return nodes
    """
    def __init__(self, matches: Path = None,
                 conditions:  Condition = None,
                 exist_branches: Exists = None,
                 returns: List[Union[Node, NodeProperty]] = None):
        super(Predicate, self).__init__(matches, conditions)
        self.returns = [] if returns is None else returns
        for node in self.returns:
            if isinstance(node, NodeProperty):
                node = node.node
                if not any(node in path.nodes for path in self.matches):
                    raise ValueError(f"A return {node} references a node that does not exist in the root")
        if not self.matches and self.returns:
            raise ValueError('There must be a root to return things from')
        self.exist_branches = exist_branches


class FullQuery(Predicate):
    """
    A FullQuery holds all the relevant information about how to construct any DB query.
    The one restriction is that there may not be nested EXISTS statments.
    This is done by requiring the use of exist_branches and predicates.
    Args:
        matches: The path from which the query extends
        conditions: The conditions for the root path
        exist_branches: Paths which must exist with some logic between them of |&^ can be nested tuples
            e.g. [(path1, '|', path2), '&', (path3, '^', path4), '&', path5]
        return_nodes: The nodes to return from the query
        return_properties: The pairs of (node, property_name) to return
    """
    def __init__(self, matches: List[Path] = None,
                 conditions: Condition = None,
                 exist_branches: Exists = None,
                 predicates: List[Union[List[Union[str, Predicate]], str, Predicate]]  = None,
                 returns: List[Union[Node, NodeProperty]] = None):
        super(FullQuery, self).__init__(matches, conditions, exist_branches, returns)
        self.predicates = [] if predicates is None else predicates

    def to_neo4j(self, generator, mentioned_nodes=None):
        mentioned_nodes = [] if mentioned_nodes is None else mentioned_nodes
        predicate_statements = []
        for predicate in self.predicates:
            predicate_statements.append(predicate.to_neo4j(generator, mentioned_nodes))
        predicates = '\n\n'.join(predicate_statements)
        main = '\n'.join([f'MATCH {p.stringify(mentioned_nodes)}' for p in self.matches])
        if self.conditions:
            wheres = f'\nWHERE {self.conditions}'
        else:
            wheres = ''
        returns = ', '.join([i.name for i in self.returns])
        return f'{predicates}\n\n{main}{wheres}\nRETURN {returns}'.strip().strip(',')



class AmbiguousPathError(Exception):
    pass