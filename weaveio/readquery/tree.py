from collections import defaultdict
from functools import reduce, wraps
from operator import xor, or_
from textwrap import dedent
from typing import Union, List, Tuple, Dict
from warnings import warn

from networkx import OrderedDiGraph
import networkx as nx

from weaveio.writequery.base import BaseStatement, CypherVariable, CypherQuery, Returns


def typeerror_is_false(func):
    @wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            return False
    return inner


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

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self._path == other._path

    def __hash__(self):
        return hash(self._path)


class Action(BaseStatement):
    compare = None

    def __init__(self, input_variables: List[CypherVariable], output_variables: List[CypherVariable]):
        super(Action, self).__init__(input_variables, output_variables, [])

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        base = set(self.input_variables) == set(other.input_variables) \
               and self.__class__ is other.__class__
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


class StartingPoint(Action):
    compare = ['label']

    def __init__(self, label):
        self.label = label
        self.hierarchy = CypherVariable(self.label)
        super().__init__([], [self.hierarchy])

    def to_cypher(self):
        return f"MATCH ({self.hierarchy}:{self.label})"

    def __str__(self):
        return f'{self.label}'


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


class BranchHandler:
    def __init__(self):
        self.graph = OrderedDiGraph()
        self.class_counter = defaultdict(int)

    def new(self, action: Action, parents: List['Branch'], variables: List[CypherVariable], hierarchies: List[CypherVariable], name: str = None):
        parent_set = set(parents)
        successors = {s for p in parents for s in self.graph.successors(p) if set(self.graph.predecessors(s)) == parent_set}
        candidates = {s for s in successors if s.action == action}
        assert len(candidates) <= 1
        if candidates:
            return successors.pop()
        if name is None:
            self.class_counter[action.__class__] += 1
            name = action.__class__.__name__ + str(self.class_counter[action.__class__])
        instance = Branch(self, action, parents, variables=variables, hierarchies=hierarchies, name=name)
        self.graph.add_node(instance, action=action, name=name)
        for parent in parents:
            self.graph.add_edge(parent, instance)
        return instance

    def begin(self, label):
        action = StartingPoint(label)
        return self.new(action, [], [], [action.hierarchy])

    def relevant_graph(self, branch):
        return nx.subgraph_view(self.graph, lambda n: nx.has_path(self.graph, n, branch) or n is branch)

    def iterdown(self, branch):
        for n in self.relevant_graph(branch):
            yield n

    def iterup(self, branch):
        nodes = [n for n in self.relevant_graph(branch)]
        for n in nodes[::-1]:
            yield n


def plot(graph, fname):
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(graph)
    A.layout('dot')
    A.draw(fname)


class Collection(Action):
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
        self.collected_variables = {i: o for i, o in zip(inputs, outputs)}
        super().__init__(inputs + self.references, outputs)

    def __getitem__(self, item: CypherVariable):
        return self.collected_variables[item]

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

    def __init__(self, string_function: str, **inputs):
        self.string_function = string_function
        self.output = CypherVariable('operation')
        self.inputs = inputs
        self.hashable_inputs = tuple(self.inputs.items())
        super().__init__(list(inputs.values()), [self.output])

    def to_cypher(self):
        return f"WITH *, {self.string_function.format(**self.inputs)} as {self.output_variables[0]}"

    def __str__(self):
        return self.string_function


class Filter(Operation):
    def __init__(self, string_function, **inputs):
        super().__init__(string_function, **inputs)

    def to_cypher(self):
        return f"WHERE {self.string_function.format(**self.inputs)}"


class Branch:
    def __init__(self, handler: BranchHandler, action: Action, parents: List['Branch'], hierarchies: List[CypherVariable],
                 variables: List[CypherVariable], name: str = None):
        self.handler = handler
        self.action = action
        self.parents = parents
        self.name = name
        self.hierarchies = hierarchies
        self.variables = variables

    def __hash__(self):
        return reduce(xor, map(hash, self.parents + self.variables + [tuple(self.hierarchies), self.action, self.handler]))

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self.handler == other.handler and self.action == other.action and set(self.parents) == set(other.parents) \
               and self.hierarchies == other.hierarchies and set(self.variables) == set(other.variables)

    def __repr__(self):
        return f'<Branch {self.name}: {str(self.action)}>'

    def iterdown(self):
        return self.handler.iterdown(self)

    def iterup(self):
        return self.handler.iterup(self)

    def traverse(self, *paths: TraversalPath) -> 'Branch':
        """
        Extend the branch from the most recent hierarchy to a new hierarchy(s) by walking along the paths
        If more than one path is given then we take the union of all the resultant nodes.
        """
        action = Traversal(self.hierarchies[-1], *paths)
        return self.handler.new(action, [self], variables=self.variables, hierarchies=self.hierarchies+[action.out])

    def align(self, *branches: 'Branch') -> 'Branch':
        """
        Join branches into one, keeping the highest cardinality.
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
        return self.handler.new(action, branches, variables=self.variables, hierarchies=self.hierarchies)

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
        variables = action.outsingle_variables + action.outmultiple_variables
        return self.handler.new(action, branches, variables=self.variables + variables, hierarchies=self.hierarchies)

    def operate(self, string_function, **inputs) -> 'Branch':
        """
        Adds a new variable to the namespace
        e.g. y = x*2 uses extant variable x to define a new variable y which is then subsequently accessible
        """
        missing = [k for k, v in inputs.items() if getattr(v, 'parent', v) not in self.variables + self.hierarchies]
        if missing:
            raise ValueError(f"inputs {missing} are not in scope for {self}")
        op = Operation(string_function, **inputs)
        return self.handler.new(op, [self], variables=self.variables + op.output_variables, hierarchies=self.hierarchies)

    def filter(self, logical_string, **boolean_variables: CypherVariable) -> 'Branch':
        """
        Reduces the cardinality of the branch by using a WHERE clause.
        .filter can only use available variables
        """
        missing = [k for k, v in boolean_variables.items() if getattr(v, 'parent', v) not in self.variables + self.hierarchies]
        if missing:
            raise ValueError(f"inputs {missing} are not in scope for {self}")
        action = Filter(logical_string, **boolean_variables)
        return self.handler.new(action, [self], variables=self.variables, hierarchies=self.hierarchies)

    def slice(self, slc: slice):
        """
        Paginate the query using start and stop
        This will use the Cypher commands SKIP and LIMIT
        """
        action = Slice(slc)
        return self.handler.new(action, [self], variables=self.variables, hierarchy=self.hierarchy)

    def returns(self, *variables) -> 'Branch':
        """
        Terminate the query and add a return statement
        """
        action = Return(self, *variables)
        return self.handler.new(action, [self], variables=[], hierarchy=self.hierarchy)


if __name__ == '__main__':
    from weaveio.opr3 import OurData
    data = OurData('data', port=7687, write=False)
    handler = BranchHandler()

    # obs[any(obs.runs.spectra.snr > 10) | any(obs.runs.spectra.observations.seeing > 0) & all(obs.targets[obs.targets.survey.name == 'WL'].ra > 0)]
    obs = handler.begin('OB')
    spectra = obs.traverse(TraversalPath('->', 'run', '->', 'spectrum'))
    observations = spectra.traverse(TraversalPath('->', 'observation'))
    targets = obs.traverse(TraversalPath('->', 'target'))
    surveys = targets.traverse(TraversalPath('->', 'survey'))

    snr = spectra.operate("{snr} > 2", snr=spectra.hierarchies[-1].get('snr'))
    seeing = observations.operate("{seeing} > 0", seeing=observations.hierarchies[-1].get('seeing'))
    surveyname = surveys.operate("{name} = 'WL'", name=surveys.hierarchies[-1].get('surveyname'))
    targets = targets.collect([surveyname], [])
    targets = targets.filter('{name}', name=targets.action[surveyname.action.output])
    ra = targets.operate('{ra} > 0', ra=targets.hierarchies[-1].get('ra'))

    collected = obs.collect([], [snr, seeing, ra])
    filtered = collected.filter('any(x in {snr} where x) OR any(x in {seeing} where x) AND all(x in {ra} where x)', snr=collected.action[snr.action.output],
                                seeing=collected.action[seeing.action.output], ra=collected.action[ra.action.output])

    plot(handler.graph, '/opt/project/querytree.png')

    ordering = list(nx.algorithms.dag.topological_sort(handler.graph))
    print(ordering)
    with CypherQuery('ignore') as query:
        for stage in ordering:
            query.add_statement(stage.action)
    cypher, _ = query.render_query()
    print(cypher)