from functools import reduce
from operator import xor

from networkx import OrderedDiGraph
import networkx as nx


class Action:
    def __init__(self, input_variables, output_variables, cardinality_delta):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.cardinality_delta = cardinality_delta

    def __eq__(self, other):
        return set(self.input_variables) == set(other.input_variables) \
               and set(self.output_variables) == set(other.output_variables) and \
               self.cardinality_delta == other.cardinality_delta

    def __hash__(self):
        return reduce(xor, map(hash, [self.input_variables, self.output_variables, self.cardinality_delta]))


class BranchHandler:
    def __init__(self):
        self.graph = OrderedDiGraph()

    def new(self, action, *parents, name: str = None):
        parent_set = set(parents)
        successors = {s for p in parents for s in self.graph.successors(p) if set(self.graph.predecessors(s)) == parent_set}
        candidates = {s for s in successors if s.action == action}
        assert len(candidates) <= 1
        if candidates:
            return successors.pop()
        instance = Branch(self, action, *parents, name=name)
        self.graph.add_node(instance, action=action, name=name)
        for parent in parents:
            self.graph.add_edge(parent, instance)
        return instance

    def relevant_graph(self, branch):
        return nx.subgraph_view(self.graph, lambda n: nx.has_path(self.graph, n, branch))

    def iterdown(self, branch):
        for n in self.relevant_graph(branch):
            yield n

    def iterup(self, branch):
        nodes = [n for n in self.relevant_graph(branch)]
        for n in nodes[::-1]:
            yield n


class Branch:
    def __init__(self, handler: BranchHandler, action: Action, *parents: 'Branch', name: str = None):
        self.handler = handler
        self.action = action
        self.parents = parents
        self.name = name

    def __hash__(self):
        return reduce(xor, map(hash, self.parents + (self.action, self.handler)))

    def __eq__(self, other):
        return self.handler == other.handler and self.action == other.action and set(self.parents) == set(other.parents)

    def __repr__(self):
        return f'<Branch {self.name}>'

    def iterdown(self):
        return self.handler.iterdown(self)

    def iterup(self):
        return self.handler.iterup(self)

    def traverse(self, label) -> 'Branch':
        raise NotImplementedError

    def join(self, branch) -> 'Branch':
        raise NotImplementedError

    def merge(self, branch) -> 'Branch':
        raise NotImplementedError

    def assign(self) -> 'Branch':
        raise NotImplementedError

    def filter(self) -> 'Branch':
        raise NotImplementedError

    def returns(self) -> 'Branch':
        raise NotImplementedError


if __name__ == '__main__':
    handler = BranchHandler()
    a = handler.new(1, name='a')
    b = handler.new(2, a, name='b')
    c = handler.new(3, b, name='c')
    d = handler.new(4, b, name='d')
    e = handler.new(5, d, name='e')
    f = handler.new(6, d, e, name='f')
    g = handler.new(7, b, f, name='g')
    g2 = handler.new(7, b, f, name='g2')

    for i in handler.iterup(g):
        print(i)
