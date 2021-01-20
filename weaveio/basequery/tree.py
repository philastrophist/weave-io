from collections import defaultdict
from functools import reduce, wraps
from operator import xor
from typing import Union, List, Dict, Optional

import networkx as nx
from networkx import OrderedDiGraph

from weaveio.writequery.base import BaseStatement, CypherVariable, CypherQuery, DerivedCypherVariable, CypherVariableItem, CypherData


def typeerror_is_false(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            return False
    return inner


def sort_rooted_dag(graph):
    for n, d in graph.in_degree():
        if d == 0:
            break
    return nx.algorithms.traversal.dfs_tree(graph, n)


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
        if self.__class__ != other.__class__:
            return False
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
        self._path = path
        self.nodes = []
        self.steps = []
        self.path = []
        self.end = CypherVariable(str(path[-1]))
        self.repr_path = ''.join(path)
        for i, entry in enumerate(path[:-1]):
            if not i % 2:  # even number
                step = Step(entry)
                self.steps.append(step)
                self.path.append(step)
            else:
                self.nodes.append(str(entry))
                self.path.append(f'(:{entry})')

    def __len__(self):
        return len(self.nodes) + 1

    def __str__(self):
        end = f'({self.end}:{self.end.namehint})'
        return ''.join(map(str, self.path)) + end

    def __repr__(self):
        return f'<TraversalPath({str(self.repr_path)})>'

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self._path == other._path

    def __hash__(self):
        return hash(self._path)


class Action(BaseStatement):
    compare = None
    shape = None

    def to_cypher(self):
        raise NotImplementedError

    def __getitem__(self, item: CypherVariable):
        if isinstance(item, CypherVariableItem):
            return self.transformed_variables[item.parent].get(item.args)
        return self.transformed_variables[item]

    def __init__(self, input_variables: List[CypherVariable], output_variables: List[CypherVariable],
                 hidden_variables: List[CypherVariable] = None, transformed_variables: Dict[CypherVariable, CypherVariable] = None, target: CypherVariable = None):
        super(Action, self).__init__(input_variables, output_variables, hidden_variables)
        self.transformed_variables = {} if transformed_variables is None else transformed_variables
        self.target = target

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        base = set(self.input_variables) == set(other.input_variables) and self.__class__ is other.__class__
        for c in self.compare:
            selfthing = getattr(self, c, None)
            otherthing = getattr(other, c, None)
            base &= selfthing == otherthing
        return base

    def __hash__(self):
        base = reduce(xor, map(hash, [tuple(self.input_variables), self.__class__.__name__]))
        for c in self.compare:
            base ^= hash(getattr(self, c))
        return base

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return f'<{str(self)}>'


class EntryPoint(Action):
    compare = []

    def __init__(self):
        super().__init__([], [])

    def to_cypher(self):
        return ''

    def __str__(self):
        return 'EntryPoint'


class StartingPoint(Action):
    compare = ['labels']

    def __init__(self, *labels):
        self.labels = labels
        self.hierarchy = CypherVariable(self.labels[-1])
        super().__init__([], [self.hierarchy], target=self.hierarchy)

    def to_cypher(self):
        s = f"OPTIONAL MATCH ({self.hierarchy}:{self.labels[0]})"
        if len(self.labels) == 1:
            return s
        condition = ' OR '.join([f'{self.hierarchy}:{l}' for l in self.labels[1:]])
        return s + f" WHERE {condition}"

    def __str__(self):
        return '|'.join(self.labels)


class DataReference(Action):
    compare = ['hashes']

    def __init__(self, *data):
        import numpy as np
        self.hashes = reduce(xor, [hash(np.array(a).tobytes()) for a in data])
        ins = [CypherData(datum, delay=True) for datum in data]
        super().__init__(ins, [])

    def to_cypher(self):
        return '// added data here'

    def __str__(self):
        return 'DataReference'



# class OrderBy(Action):
#
#     def __init__(self, base_hierarchies, ):
#
#     def to_cypher(self):
#         collections = f'collect({self.sort_node}) as {self.collected_variable}'
#         with_statement = ', '.join(map(str, self.base_hierarchies + collections))
#         call_statement = f'with {self.collected_variable} UNWIND {self.collected_variable} as {self.sort_node} RETURN {self.sort_node}' \
#                          f' ORDER BY {self.sort_node}.{self.sort_property}'
#         return f"WITH {with_statement} + CALL {{ {call_statement} }}"


class Traversal(Action):
    """
    Traverse from one hierarchy level to another. This extends the branch and
    potentially increases the cardinality.
    Given a `source` branch and one or more `paths` of form (minnumber, maxnumber, direction),
     traverse to the nodes described by the `paths`.
    For example:
        >>> Traversal(branch, TraversalPath(['->', 'Exposure', '->', 'OB', '->', 'OBSpec']))
        results in `OPTIONAL MATCH (run)-[]->(:Exposure)-[]->(:OB)-[]->(obspec:OBSpec)`
    To traverse multiple paths at once, we use unions in a subquery
    """
    compare = ['paths', 'source']

    def __init__(self, source: CypherVariable, *paths: TraversalPath, name=None):
        if name is None:
            name = ''.join(p.end.namehint for p in paths)
        if len(paths) > 1:
            self.out = CypherVariable(name)
            outs = [p.end for p in paths] + [self.out]
        else:
            self.out = paths[0].end
            outs = [self.out]
        super(Traversal, self).__init__([source], outs, target=self.out)
        self.source = source
        self.paths = paths

    def to_cypher(self):
        lines = [f'OPTIONAL MATCH ({self.source}){p}' for p in self.paths]
        lines = '\n\nUNION\n\n'.join([f'\tWITH {self.source}\n\t{l}\n\tRETURN DISTINCT {path.end} as {self.out}' for l, path in zip(lines, self.paths)])
        return f"""CALL {{\n{lines}\n}}"""

    def __str__(self):
        return f'{self.source.namehint}->{self.out.namehint}'


class Return(Action):
    compare = ['branch', 'varnames']  # just compare input_variables

    def __init__(self, branch: 'Branch', *varnames):
        self.branch = branch
        self.varnames = varnames
        super(Return, self).__init__([branch.hierarchies[-1]], [])

    def to_cypher(self):
        proj = ', '.join([f'{self.branch.hierarchies[-1].get(v)}' for v in self.varnames])
        return f"RETURN {proj}"

    def __str__(self):
        return f'return {self.varnames}'


class Collection(Action):
    shape = 'rect'
    compare = ['_singles', '_multiples', '_reference']

    def __init__(self, reference: 'Branch', singles: List['Branch'], multiples: List['Branch']):
        self._singles = tuple(singles)
        self._multiples = tuple(multiples)
        self._reference = reference

        self.references = reference.hierarchies
        self.insingle_hierarchies = [x.hierarchies[-1] for x in singles]
        self.insingle_variables = [v for x in singles for v in x.variables]
        self.inmultiple_hierarchies = [x.hierarchies[-1] for x in multiples]
        self.inmultiple_variables = [v for x in multiples for v in x.variables]

        self.outsingle_hierarchies = [CypherVariable(s.namehint) for s in self.insingle_hierarchies]
        self.outsingle_variables = [CypherVariable(s.namehint) for s in self.insingle_variables]
        self.outmultiple_hierarchies = [CypherVariable(s.namehint+'_list') for s in self.inmultiple_hierarchies]
        self.outmultiple_variables = [CypherVariable(s.namehint+'_list') for s in self.inmultiple_variables]
        inputs = self.insingle_hierarchies + self.insingle_variables + self.inmultiple_variables + self.inmultiple_hierarchies
        outputs = self.outsingle_hierarchies + self.outsingle_variables + self.outmultiple_variables + self.outmultiple_hierarchies
        super().__init__(inputs + self.references, outputs, [], transformed_variables={i: o for i, o in zip(inputs, outputs)})

    def to_cypher(self):
        base = [f'{r}' for r in self.references + ['time0']]
        single_hierarchies = [f'coalesce({i}) as {o}' for i, o in zip(self.insingle_hierarchies, self.outsingle_hierarchies)]
        multiple_hierarchies = [f'collect({i}) as {o}' for i, o in zip(self.inmultiple_hierarchies, self.outmultiple_hierarchies)]
        single_variables = [f'coalesce({i}) as {o}' for i, o in zip(self.insingle_variables, self.outsingle_variables)]
        multiple_variables = [f'collect({i}) as {o}' for i, o in zip(self.inmultiple_variables, self.outmultiple_variables)]
        return 'WITH ' + ', '.join(base + single_hierarchies + single_variables + multiple_hierarchies + multiple_variables)

    def __str__(self):
        return f'collect'


class Operation(Action):
    compare = ['string_function', 'hashable_inputs']

    def __init__(self, *string_functions: str, namehint=None, **inputs):
        self.string_functions = string_functions
        self.outputs = [CypherVariable(namehint) for _ in string_functions]
        self.inputs = inputs
        self.hashable_inputs = tuple(self.inputs.items())
        super().__init__(list(inputs.values()), self.outputs, target=self.outputs[0])

    def to_cypher(self):
        assignments = [f"{func.format(**self.inputs)} as {out}" for func, out in zip(self.string_functions, self.outputs)]
        return f"WITH *, {', '.join(assignments)}"

    def __str__(self):
        return ', '.join(self.string_functions)


class Filter(Operation):
    shape = 'diamond'
    def __init__(self, string_function, **inputs):
        super().__init__(string_function, **inputs)

    def to_cypher(self):
        return f"WHERE {self.string_function.format(**self.inputs)}"


# class Alignment(Action):
#     compare = ['branches', 'reference']
#     shape = 'house'
#
#     def __init__(self, reference: 'Branch', *branches: 'Branch'):
#         """
#         Collects and unwinds all variables/hierarchies that came after the reference branch
#         Persists all variables/hierarchies that came before the reference branch
#         """
#         self.reference = reference
#         self.branches = branches
#         base = tuple() if reference is None else (self.reference, )
#         ref_vars = [] if reference is None else reference.variables + reference.hierarchies
#
#         ins = []
#         before, after = set(), []
#         for branch in branches + base:
#             ins += branch.variables + branch.hierarchies
#             before |= {v for v in branch.variables + branch.hierarchies if v in ref_vars}
#             after += [v for v in branch.variables + branch.hierarchies if v not in ref_vars]
#         hidden = [CypherVariable(x.namehint+'_collected') for x in after]
#         outs = [CypherVariable(x.namehint+'_aligned') for x in after]
#         self.indexer = CypherVariable('i')
#         self.after = []
#         self.hidden = []
#         self.outs = []  # remove correlated duplicates
#         for a, h, o in zip(after, hidden, outs):
#             if a not in self.after:
#                 self.after.append(a)
#                 self.hidden.append(h)
#                 self.outs.append(o)
#         self.before = list(before)
#         self.output_variables = [o for a, o in zip(after, outs) if any(a in b.variables for b in branches)]
#         self.output_variables += reference.variables
#         self.output_hierarchies = [o for a, o in zip(after, outs) if any(a in b.hierarchies for b in branches)]
#         self.output_hierarchies += reference.hierarchies
#         transformed = {a: o for a, o in zip(after, outs)}
#         transformed.update({r: r for r in ref_vars})
#         super().__init__(ins, self.outs, self.hidden + [self.indexer], transformed)
#
#     def to_cypher(self):
#         base = ['time0'] + [str(b) for b in self.before]
#         base += [f'collect({i}) as {h}' for i, h in zip(self.after, self.hidden)]
#         unwind = f'UNWIND range(0, apoc.coll.max([x in {self.hidden} | size(x)])-1) as {self.indexer}'
#         get = [f'{h}[{self.indexer}] as {o}' for h, o in zip(self.hidden, self.outs)]
#         if len(self.after):
#             return f"WITH {', '.join(base)}\n{unwind}\nWITH *, {', '.join(get)}"
#         return f"WITH {', '.join(base)}"
#
#     def __str__(self):
#         return 'align'

class Alignment(Action):
    compare = ['reference', 'branches']

    def __init__(self, reference, *branches):
        self.reference = reference
        self.branches = branches
        super(Alignment, self).__init__([], [])

    def to_cypher(self):
        return ''

    def __str__(self):
        return 'align'


class Slice(Action):
    compare = []

    def __init__(self, slc):
        self.slc = slc
        self.skip = slc.start
        self.limit = slc.stop - slc.start
        super().__init__([], [], [])

    def __str__(self):
        return f'{self.slc}'

    def to_cypher(self):
        return f'WITH * SKIP {self.skip} LIMIT {self.limit}'


class Results(Action):
    compare = ['branches']

    def __init__(self, branch_attributes):
        self.branch_attributes = branch_attributes
        self.branches = tuple(branch_attributes.keys())
        ins = [j for i in self.branch_attributes.values() for j in i]
        super(Results, self).__init__(ins, [], [], {}, None)

    def to_cypher(self):
        return 'RETURN {}'.format(', '.join(map(str, self.input_variables)))

    def __str__(self):
        names = [i.namehint for i in self.input_variables]
        return 'return {}'.format(', '.join(names))



class BranchHandler:
    def __init__(self):
        self.graph = OrderedDiGraph()
        self.class_counter = defaultdict(int)
        self.entry = self.new(EntryPoint(), [], [], None, [], [], [])
        self.data_objects = {}

    def new(self, action: Action, accessible_parents: List['Branch'], inaccessible_parents: List['Branch'],
            current_hierarchy: Optional[CypherVariable], current_variables: List[CypherVariable],
            variables: List[CypherVariable], hierarchies: List[CypherVariable], name: str = None):
        parents = accessible_parents + inaccessible_parents
        parent_set = set(parents)
        successors = {s for p in parents for s in self.graph.successors(p) if set(self.graph.predecessors(s)) == parent_set}
        candidates = {s for s in successors if s.action == action}
        assert len(candidates) <= 1
        if candidates:
            return successors.pop()
        if name is None:
            self.class_counter[action.__class__] += 1
            name = action.__class__.__name__ + str(self.class_counter[action.__class__])
        instance = Branch(self, action, accessible_parents, inaccessible_parents, current_hierarchy, current_variables,
                          variables=variables, hierarchies=hierarchies, name=name)
        attrs = {}
        if action.shape is not None:
            attrs['shape'] = action.shape
        self.graph.add_node(instance, action=action, name=name, **attrs)
        for parent in parents:
            self.graph.add_edge(parent, instance, accessible=parent in accessible_parents)
        self.current_hierarchy = current_hierarchy
        return instance

    def begin(self, *labels):
        action = StartingPoint(*labels)
        return self.new(action, [self.entry], [], action.hierarchy, [action.hierarchy], [], [action.hierarchy])

    def relevant_graph(self, branch):
        return nx.subgraph_view(self.graph, lambda n: nx.has_path(self.graph, n, branch) or n is branch)

    def deepest_common_ancestor(self, *branches: 'Branch'):
        common = set()
        for i, branch in enumerate(branches):
            ancestors = set(nx.algorithms.dag.ancestors(self.graph, branch))
            if i == 0:
                common = ancestors
            else:
                common &= ancestors
        if not len(common):
            return None
        distances = [(sum(nx.shortest_path_length(self.graph, ancestor, b) for b in branches), ancestor) for ancestor in common]
        return distances[distances.index(min(distances, key=lambda x: x[0]))][1]


def plot(graph, fname):
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(graph)
    A.layout('dot')
    A.draw(fname)


class Branch:
    def __init__(self, handler: BranchHandler, action: Action,
                 accessible_parents: List['Branch'], inaccessible_parents: List['Branch'],
                 current_hierarchy: Optional[CypherVariable], current_variables: List[CypherVariable],
                 hierarchies: List[CypherVariable],
                 variables: List[CypherVariable], name: str = None):
        """
        A branch is an object that represents all the Actions (query statements) in a query.
        It contains both a node (Action) and references to all actions (other Branches) preceeding it (parents).
        If a branch is created in the same way more than once, only one object is actually instantiated, this is to optimize performance
        when writing to cypher query language. The user shouldn't care about this quirk.
        :param handler: The handler object which oversees query uniqueness for a particular dataset.
        :param action: The action that this branch will execute at the end. These are generally (but not always) Cypher statements.
        :param parents: The branches that must be executed before this one (i.e. the dependencies of this branch)
        """
        self.handler = handler
        self.action = action
        self.parents = accessible_parents + inaccessible_parents
        self.accessible_parents = accessible_parents
        self.inaccessible_parents = inaccessible_parents
        self.current_hierarchy = current_hierarchy
        self.current_variables = current_variables
        self.name = name
        self.hierarchies = hierarchies
        self.variables = variables
        self._relevant_graph = None
        self._accessible_graph = None
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = reduce(xor, map(hash, self.parents + [self.current_hierarchy, tuple(self.current_variables), self.action, self.handler]))
        return self._hash

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self.handler == other.handler and self.action == other.action and set(self.parents) == set(other.parents) \
               and self.current_hierarchy == other.current_hierarchy and set(self.current_variables) == set(other.current_variables)

    def __repr__(self):
        return f'<Branch {self.name}: {str(self.action)}>'

    @property
    def relevant_graph(self):
        if self._relevant_graph is None:
            self._relevant_graph = self.handler.relevant_graph(self)
        return self._relevant_graph

    @property
    def accessible_graph(self):
        if self._accessible_graph is None:
            g = nx.subgraph_view(self.relevant_graph, filter_edge=lambda a, b: self.relevant_graph.edges[(a, b)]['accessible'])
            self._accessible_graph = nx.subgraph_view(g, filter_node=lambda x: nx.has_path(g, x, self))
        return self._accessible_graph

    def iterdown(self, graph=None, start=None):
        if graph is None:
            graph = self.relevant_graph
        if start is None:
            start = self.handler.entry
        return nx.algorithms.traversal.dfs_tree(graph, start)

    def find_hierarchies(self):
        hierarchies = []
        for branch in self.iterdown(self.accessible_graph):
            if branch.current_hierarchy is not None:
                hierarchies.append(branch.current_hierarchy)
        return hierarchies

    def find_variables(self):
        variables = []
        for branch in self.iterdown(self.accessible_graph):
            variables += branch.current_variables
        return variables

    def find_hierarchy_branches(self):
        branches = []
        for branch in self.iterdown(self.accessible_graph):
            if branch.current_hierarchy is not None:
                branches.append(branch)
        return branches

    def add_data(self, *data) -> 'Branch':
        action = DataReference(*data)
        return self.handler.new(action, [self], [], current_hierarchy=None, current_variables=action.input_variables,
                         variables=self.variables+[action.input_variables], hierarchies=self.hierarchies)

    def traverse(self, *paths: TraversalPath) -> 'Branch':
        """
        Extend the branch from the most recent hierarchy to a new hierarchy(s) by walking along the paths
        If more than one path is given then we take the union of all the resultant nodes.
        """
        action = Traversal(self.find_hierarchies()[-1], *paths)
        return self.handler.new(action, [self], [], current_hierarchy=action.out, current_variables=[action.out],
                                variables=self.variables, hierarchies=self.hierarchies+[action.out])

    def align(self, branch: 'Branch') -> 'Branch':
        """
        Join branches into one, keeping the highest cardinality.
        This is used to to directly compare arrays:
            * ob1.runs == ob2.runs  (unequal sizes are zipped up together)
            * ob1.runs == run1  (array to single comparisons are left as is)
            * run1 == run2  (single to single comparisons are left as is)
        zip ups and unwinds take place relative to the branch's shared ancestor
        """
        action = Alignment(self, branch)
        return self.handler.new(action, [self, branch], [], None, current_variables=[],
                                variables=self.variables, hierarchies=self.hierarchies)

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
        variables = action.outsingle_variables + action.outmultiple_variables
        hierarchies = action.outsingle_hierarchies + action.outmultiple_hierarchies
        return self.handler.new(action, [self], singular + multiple, None, variables + hierarchies,
                                variables=self.variables + variables, hierarchies=self.hierarchies + hierarchies)

    def operate(self, *string_functions, namehint=None, **inputs) -> 'Branch':
        """
        Adds a new variable to the namespace
        e.g. y = x*2 uses extant variable x to define a new variable y which is then subsequently accessible
        """
        # missing = [k for k, v in inputs.items() if getattr(v, 'parent', v) not in self.variables + self.hierarchies]
        # if missing:
        #     raise ValueError(f"inputs {missing} are not in scope for {self}")
        op = Operation(*string_functions, namehint=namehint, **inputs)
        return self.handler.new(op, [self], [], None, op.output_variables,
                                variables=self.variables + op.output_variables, hierarchies=self.hierarchies)

    def filter(self, logical_string, **boolean_variables: CypherVariable) -> 'Branch':
        """
        Reduces the cardinality of the branch by using a WHERE clause.
        .filter can only use available variables
        """
        # missing = [k for k, v in boolean_variables.items() if getattr(v, 'parent', v) not in self.variables + self.hierarchies]
        # if missing:
        #     raise ValueError(f"inputs {missing} are not in scope for {self}")
        action = Filter(logical_string, **boolean_variables)
        return self.handler.new(action, [self], [], None, [],
                                variables=self.variables, hierarchies=self.hierarchies)

    def results(self, branch_attributes: Dict['Branch', List[CypherVariable]]):
        """
        Returns the rows of results.
        All other branches which are not this branch will have their results collected
        """
        branch_attributes = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in branch_attributes.items()}
        action = Results(branch_attributes)
        return self.handler.new(action, [self], [], None, [], self.variables, self.hierarchies)
