from collections import defaultdict
from functools import reduce, wraps
from operator import xor, or_
from textwrap import dedent
from typing import Union, List, Tuple, Dict
from warnings import warn

from networkx import OrderedDiGraph
import networkx as nx

from weaveio.writequery.base import BaseStatement, CypherVariable, CypherQuery


def typeerror_is_false(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            return False
    return inner


class CardinalityDelta:
    def __init__(self, low: int = None, high: int = None):
        self.low = low
        self.high = high

    def __add__(self, other):
        if self.low is None and other.low is not None:
            low = other.low
        elif other.low is None and self.low is not None:
            low = self.low

    @typeerror_is_false
    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    @typeerror_is_false
    def __gt__(self, other):
        return self.low > other.high

    @typeerror_is_false
    def __lt__(self, other):
        return self.low < other.high

    @typeerror_is_false
    def __ge__(self, other):
        return self.low >= other.high

    @typeerror_is_false
    def __le__(self, other):
        return self.low <= other.high


class Step:
    def __init__(self, direction: str, label: str = 'is_required_by', properties: Dict = None):
        if isinstance(direction, Step):
            self.direction = direction.direction
            self.label = direction.label
            self.properties = direction.properties
        elif direction in ['->', '<-', '-']:
            self.direction = direction
            self.label = label
            self.properties = properties
        else:
            raise ValueError(f"Direction {direction} is not supported")

        @typeerror_is_false
        def __eq__(self, other):
            return self.direction == other.direction and self.properties == other.properties and self.label == other.label

    def __str__(self):
        if self.properties is None:
            mid = f'-[:{self.label}]-'
        else:
            mid = f'-[:{self.label} {self.properties}]-'
        if self.direction == '->':
            return f"{mid}>"
        elif self.direction == '<-':
            return f"<{mid}"
        else:
            return mid


class TraversalPath:
    def __init__(self, *path: Union[Step, str]):
        self.nodes = []
        self.steps = []
        self.path = []
        self.end = CypherVariable(str(path[-1]))
        for i, entry in enumerate(path[:-1]):
            if not i % 2:  # even number
                step = Step(entry)
                self.steps.append(step)
                self.path.append(step)
            else:
                self.nodes.append(str(entry))
                self.path.append(f'(:{entry})')

    def __str__(self):
        end = f'({self.end}:{self.end.namehint})'
        return ''.join(map(str, self.path)) + end


class Action(BaseStatement):
    def __init__(self, input_variables: List[CypherVariable], output_variables: List[CypherVariable]):
        super(Action, self).__init__(input_variables, output_variables, [])

    def __eq__(self, other):
        return set(self.input_variables) == set(other.input_variables) \
               and set(self.output_variables) == set(other.output_variables) and \
               self.__class__ is other.__class__

    def __hash__(self):
        return reduce(xor, map(hash, [tuple(self.input_variables), tuple(self.output_variables)]))


class StartingPoint(Action):

    def __init__(self, label):
        self.label = label
        self.hierarchy = CypherVariable(self.label)
        super().__init__([], [self.hierarchy])

    def to_cypher(self):
        return f"MATCH ({self.hierarchy}:{self.label})"


class Traversal(Action):
    """
    Traverse from one hierarchy level to another. This extends the branch and
    potentially increases the cardinality.
    Given a `source` branch and one or more `paths` of form (minnumber, maxnumber, direction),
     traverse to the nodes described by the `paths`.
    For example:
        >>> Traversal(<branch Run>, ['->', 'Exposure', '->', 'OB', '->', 'OBSpec'])
        results in `OPTIONAL MATCH (run)-[]->(:Exposure)-[]->(:OB)-[]->(obspec:OBSpec)`
    To traverse multiple paths at once, we use unions in a subquery
    """
    def __init__(self, source: CypherVariable, *paths: TraversalPath, name=None):
        if name is None:
            name = ''.join(p.end.namehint for p in paths)
        if len(paths) > 1:
            self.out = CypherVariable(name)
            outs = [p.end for p in paths] + [self.out]
        else:
            self.out = paths[0].end
            outs = [self.out]
        super(Traversal, self).__init__([source], outs)
        self.source = source
        self.paths = paths

    def to_cypher(self):
        lines = [f'OPTIONAL MATCH ({self.source}){p}' for p in self.paths]
        if len(self.paths) == 1:
            return lines[0]
        lines = '\n\nUNION\n\n'.join([f'\tWITH {self.source}\n\t{l}\n\tRETURN {path.end} as {self.out}' for l, path in zip(lines, self.paths)])
        query = f"""CALL {{\n{lines}\n}}"""
        return query


class BranchHandler:
    def __init__(self):
        self.graph = OrderedDiGraph()
        self.class_counter = defaultdict(int)

    def new(self, action: Action, parents: List['Branch'], variables: List[CypherVariable], hierarchy: CypherVariable, name: str = None):
        parent_set = set(parents)
        successors = {s for p in parents for s in self.graph.successors(p) if set(self.graph.predecessors(s)) == parent_set}
        candidates = {s for s in successors if s.action == action}
        assert len(candidates) <= 1
        if candidates:
            return successors.pop()
        if name is None:
            self.class_counter[action.__class__] += 1
            name = action.__class__.__name__ + str(self.class_counter[action.__class__])
        instance = Branch(self, action, parents, variables=variables, hierarchy=hierarchy, name=name)
        self.graph.add_node(instance, action=action, name=name)
        for parent in parents:
            self.graph.add_edge(parent, instance)
        return instance

    def begin(self, label):
        action = StartingPoint(label)
        return self.new(action, [], [], action.hierarchy)

    def relevant_graph(self, branch):
        return nx.subgraph_view(self.graph, lambda n: nx.has_path(self.graph, n, branch) or n is branch)

    def iterdown(self, branch):
        for n in self.relevant_graph(branch):
            yield n

    def iterup(self, branch):
        nodes = [n for n in self.relevant_graph(branch)]
        for n in nodes[::-1]:
            yield n


class Branch:
    def __init__(self, handler: BranchHandler, action: Action, parents: List['Branch'], hierarchy: CypherVariable,
                 variables: List[CypherVariable], name: str = None):
        self.handler = handler
        self.action = action
        self.parents = parents
        self.name = name
        self.hierarchy = hierarchy
        self.variables = variables

    def __hash__(self):
        return reduce(xor, map(hash, self.parents + self.variables + [self.hierarchy, self.action, self.handler]))

    def __eq__(self, other):
        return self.handler == other.handler and self.action == other.action and set(self.parents) == set(other.parents) \
               and self.hierarchy == other.hierarchy and set(self.variables) == set(other.variables)

    def __repr__(self):
        return f'<Branch {self.name}>'

    def iterdown(self):
        return self.handler.iterdown(self)

    def iterup(self):
        return self.handler.iterup(self)

    def traverse(self, *paths: TraversalPath) -> 'Branch':
        """
        Extend the branch to a new node(s) by walking along the paths
        If more than one path is given then we take the union of all the resultant nodes.
        """
        action = Traversal(self.hierarchy, *paths)
        return self.handler.new(action, [self], variables=[], hierarchy=action.out)

    def align(self, *branches: 'Branch') -> 'Branch':
        """
        Join two branches into one, keeping the highest cardinality.
        This is used to to directly compare arrays:
            * ob1.runs == ob2.runs  (unequal sizes are zipped up together)
            * ob1.runs == run1  (array to single comparisons are left as is)
            * run1 == run2  (single to single comparisons are left as is)
        zip ups and unwinds take place relative to the branch's shared ancestor
        """
        action = Alignment(self, branches)
        branches = list(branches)
        branches.append(self)
        hierarchy = self.handler.deepest_common_hierarchy(self, *branches)
        return self.handler.new(action, branches, variables=action.variables, hierarchy=hierarchy)

    def collect(self, singular: List['Branch'], multiple: List['Branch']) -> 'Branch':
        """
        Join branches into one, reducing the cardinality to this branch.
        `singular` contains branches that will be coalesced (i.e. only the first result is taken)
        `multiple` contains branches that will be collected (i.e. all results are presented in a list)
        This is used in predicate filters:
            ob1.runs[any(ob1.runs.l1singlespectra.snr > 10)]
            0. branch `ob1.runs` is created
            1. branch `ob1.runs.l1singlespectra.snr > 10` is created
            2. branch `ob1.runs.l1singlespectra.snr > 10` is collected with respect to `ob1.runs`
            3. A filter is applied on the collection at the `ob1.runs` level
        After a collection, only
        """
        action = Collection(self, singular, multiple)
        branches = [self] + singular + multiple
        return self.handler.new(action, branches, variables=action.variables, hierarchy=self.hierarchy)

    def assign(self, operation, varnames: List[str], input_variables: List[CypherVariable]) -> 'Branch':
        """
        Adds a new variable to the namespace
        e.g. y = x*2 uses extant variable x to define a new variable y which is then subsequently accessible
        """
        action = Assignment(operation, self, varnames, input_variables)
        return self.handler.new(action, [self], variables=self.variables + action.output_variables, hierarchy=self.hierarchy)

    def filter(self, operation, filtered_variable, input_variables) -> 'Branch':
        """
        Reduces the cardinality of the branch by using a WHERE clause.
        .filter can only use available variables
        """
        action = Filter(self, filtered_variable)
        return self.handler.new(action, [self], variables=self.variables, hierarchy=self.hierarchy)

    def slice(self, slc: slice):
        """
        Paginate the query using start and stop
        This will use the Cypher commands SKIP and LIMIT
        """
        action = Slice(slc)
        return self.handler.new(action, [self], variables=self.variables, hierarchy=self.hierarchy)

    def returns(self, variables) -> 'Branch':
        """
        Terminate the query and add a return statement
        """
        action = Return(self, variables)
        return self.handler.new(action, [self], variables=[], hierarchy=self.hierarchy)


if __name__ == '__main__':
    from weaveio.opr3 import OurData
    data = OurData('data', port=7687, write=False)
    handler = BranchHandler()

    branch = handler.begin('OB')
    branch = branch.traverse(TraversalPath('->', 'Exposure', '->', 'Run', '->', 'Observation'))
    branch = branch.traverse(TraversalPath('<-', 'RawFile'), TraversalPath('<-', 'File'))

    with CypherQuery('ignore') as query:
        for stage in branch.iterdown():
            query.add_statement(stage.action)
    cypher, _ = query.render_query()
    print(cypher)