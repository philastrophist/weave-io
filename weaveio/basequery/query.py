from collections import defaultdict
from typing import List, Union, Tuple, Dict

from weaveio.basequery.query_objects import Copyable, Node, NodeProperty, Collection, Path, Unwind, Condition


class Exists(Copyable):
    def __init__(self, path: Path):
        self.path = path


class BaseQuery:
    """A Query which consists of a root path, where conditions"""
    def __init__(self, matches: List[Union[Path, Unwind]] = None,
                 branches: Dict[Path, List[Union[Node, NodeProperty]]] = None,
                 conditions: Condition = None):
        self.matches = [] if matches is None else matches
        self.branches = defaultdict(list)
        if branches is not None:
            for path, nodelikes in branches.items():
                self.branches[path] += nodelikes
        self.conditions = conditions
        matches_only = [i for i in self.matches if not isinstance(i, Unwind)]
        for i, path in enumerate(matches_only):
            if i > 0:
                if not any(n in matches_only[i-1].nodes for n in path.nodes):
                    raise ValueError(f"A list of matches must have overlapping nodes")

    @property
    def matches(self):
        return self._matches

    @matches.setter
    def matches(self, value):
        if not all(isinstance(i, (Path, Unwind)) for i in value) and value is not None:
            raise TypeError("matches must be a list of paths or unwind data")
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
    def __init__(self, matches: List[Union[Path, Unwind]] = None,
                 branches: Dict[Path, List[Union[Node, NodeProperty]]] = None,
                 conditions: Condition = None,
                 exist_branches: Exists = None,
                 returns: List[Union[Node, NodeProperty]] = None):
        super(Predicate, self).__init__(matches, branches, conditions)
        self.returns = [] if returns is None else returns
        for node in self.returns:
            node = node.node
            if not any(node == path.nodes[-1] for path in self.matches+list(self.branches.keys())):
                raise KeyError(f"A return {node} references a node that does not exist in the root")
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
    def __init__(self, matches: List[Union[Path, Unwind]] = None,
                 branches: Dict[Path, List[Union[Node, NodeProperty]]] = None,
                 conditions: Condition = None,
                 exist_branches: Exists = None,
                 predicates: List[Union[List[Union[str, Predicate]], str, Predicate]]  = None,
                 returns: List[Union[Node, NodeProperty, Collection]] = None):
        super(FullQuery, self).__init__(matches, branches, conditions, exist_branches, returns)
        self.predicates = [] if predicates is None else predicates

    def to_neo4j(self, mentioned_nodes=None) -> Tuple[str, Dict]:
        mentioned_nodes = [] if mentioned_nodes is None else mentioned_nodes
        predicate_statements = []
        for predicate in self.predicates:
            predicate_statements.append(predicate.to_neo4j(mentioned_nodes))
        predicates = '\n\n'.join(predicate_statements)
        statements = {Path: 'MATCH', Unwind: 'UNWIND'}
        main = '\n'.join([f'{statements[p.__class__]} {p.stringify(mentioned_nodes)}' for p in self.matches])
        if self.conditions:
            wheres = f'\nWHERE {self.conditions}'
        else:
            wheres = ''
        returns_from_branches = [node for nodes in self.branches.values() for node in nodes]
        returns_from_matches = [r for r in self.returns if r not in returns_from_branches]
        carry_nodes = [node[-1].name for node in self.matches if not isinstance(node, Unwind)]  # nodes that need to be used later
        initial_aliases = [nodelike.context_string for nodelike in returns_from_matches if nodelike.context_string not in carry_nodes]
        context_statements = ['WITH ' + ', '.join(carry_nodes+initial_aliases)]

        withs = carry_nodes
        withs += [nodelike.alias_name for nodelike in returns_from_matches if nodelike.alias_name not in withs]
        for path, nodelikes in self.branches.items():
            optional = f"OPTIONAL MATCH {path.stringify(mentioned_nodes)}"
            aggregations = [f'{nodelike.name} as {nodelike.alias_name}' for nodelike in nodelikes if nodelike.alias_name not in withs]
            context_statements.append(optional)
            context_statements.append(f'WITH ' + ', '.join(withs + aggregations))
            withs += [nodelike.alias_name for nodelike in nodelikes if nodelike.alias_name not in withs]
        if context_statements:
            context_statements = '\n' + '\n'.join(context_statements)

        returns = ', '.join([f'{i.alias_name}' for i in self.returns])
        payload = {i.name: i.data for i in self.matches if isinstance(i, Unwind)}
        return f'{predicates}\n\n{main}{wheres}{context_statements}\nRETURN {returns}'.strip().strip(','), payload



class AmbiguousPathError(Exception):
    pass