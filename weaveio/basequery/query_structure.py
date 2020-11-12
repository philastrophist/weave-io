from collections import defaultdict
from itertools import count
from copy import deepcopy
from itertools import zip_longest
import networkx as nx

from weaveio.basequery.query_objects import Condition


class _NodeReference:
    def __init__(self, graph: 'QueryTree', nx_label: int):
        self.graph = graph
        self.nx_label = nx_label

    @property
    def nx_node(self):
        return self.graph.g.nodes[self.nx_label]

    def instance_number(self):
        d = defaultdict(int)
        for n in self.graph.g.nodes:
            if n == self.nx_label:
                return d[n]
            d[n] +=1
        else:
            raise KeyError(f"{self.nx_label}")

    def __eq__(self, other):
        if self.nx_label != other.nx_label:
            return False
        return self.graph.graph_equal(other.graph, join=self)


class HierarchyNodeReference(_NodeReference):
    def __init__(self, graph: 'QueryTree', nx_label: int):
        super().__init__(graph, nx_label)
        assert self.nx_node['ntype'] == 'hierarchy'

    @property
    def name(self):
        data = self.nx_node
        return f'{data["label"]}{self.instance_number()}'

    def __repr__(self):
        return f'({self.nx_node["label"]}: {self.name})'

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(f"{self} has no attribute {item}")
        return NodeProperty(self, item)


class ConditionNodeReference(_NodeReference):
    def __init__(self, graph: 'QueryTree', nx_label: int):
        super().__init__(graph, nx_label)
        assert self.nx_node['ntype'] == 'condition'

    @property
    def name(self):
        data = self.nx_node
        return f'{data["label"]}'

    def __repr__(self):
        return f'{self.name}'


nodetypes = {'hierarchy': HierarchyNodeReference, 'condition': ConditionNodeReference}

def NodeReference(graph: 'QueryTree', nx_label: int):
    return nodetypes[graph.g.nodes[nx_label]['ntype']](graph, nx_label)


class NodeProperty:
    def __init__(self, node, property_name):
        self.node = node
        self.property_name = property_name

    def __repr__(self):
        return f'{self.node.name}.{self.property_name}'


class Filter:
    def __init__(self, condition, subject):
        self.condition = condition
        self.subject = subject

    def __str__(self):
        return f"{self.subject}[{self.condition}]"


class AlignmentError(Exception):
    pass


def draw_query(graph: 'QueryTree'):
    g = graph.g
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    layout = {n: (data['column'], data['row']) for n, data in g.nodes(data=True)}
    nx.draw_networkx_nodes(g, layout)
    nx.draw_networkx_labels(g, layout, labels={n: str(NodeReference(graph, n)) for n in g.nodes})

    for edge in g.edges:
        if g.edges[edge]['etype'] == 'condition':
            ax.annotate("",
                        xy=layout[edge[0]], xycoords='data',
                        xytext=layout[edge[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="<-", color="0.5",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=-0.3",
                                        ),
                        )
        else:
            nx.draw_networkx_edges(g, layout, [edge])
    # ax.set_ylim(2, self.row-2)
    maxcolumn = max(c for _, c in g.nodes.data('column'))
    ax.set_xlim(-2, maxcolumn + 2)
    plt.draw()
    return fig


class QueryTree:
    """
    A QueryTree is a collection of nodes, hierarchy-edges, condition-edges, and transform-edges
        each node represents some definite returnable value
        each hierarchy-edge represents a link in the database `r<--e = (r:run)<--(e:exposure)`
        each condition-edge represents a filter from one node
            A condition-edge must go around a hierarchy-edge like:
                `a-[condition]->a'` would condition like `a[Condition(a)]`
                or
                `a-[condition]->a',a-->b-[condition]->a'` would condition like `a[Condition(a) | Condition(a.bs)]
    These are all stored in a nx.DiGraph, where each node has a index number to determine order
    """

    def __init__(self, source_label: str):
        self.node_counter = 0
        self.g = nx.OrderedMultiDiGraph()
        self.add_hierarchy(source_label, source=True)

    def graph_equal(self, other, join: NodeReference = None):
        if join is not None:
            try:
                before = nx.subgraph(self.g, nx.algorithms.ancestors(self.g, join.nx_label))
                before_other = nx.subgraph(other.g, nx.algorithms.ancestors(other.g, join.nx_label))
            except KeyError:
                return False
        else:
            before, before_other = self.g, other.g
        return nx.algorithms.is_isomorphic(before, before_other,
                nx.algorithms.isomorphism.categorical_node_match(['label', 'ntype'], [None, None, None]),
                nx.algorithms.isomorphism.categorical_edge_match(['etype'], [None]))

    def add_hierarchy(self, label, source=False):
        self.g.add_node(self.node_counter, label=label, column=0, row=-self.node_counter,
                        ntype='hierarchy')
        if not source:
            self.g.add_edge(self.current_node.nx_label, self.node_counter, etype='hierarchy')
        self.current_node = NodeReference(self, self.node_counter)
        self.node_counter += 1
        return self.current_node

    def available_nodes(self) -> _NodeReference:
        yield self.current_node
        for n in nx.descendants(self.g, self.current_node.nx_label):
            if len(list(self.g.successors(n))) == 0:
                yield NodeReference(self, n)

    def draw_query(self):
        return draw_query(self)

    def add_branch(self, other: 'QueryTree'):
        # check the paths shared the same history before current node
        if not self.graph_equal(other, self.current_node):
            raise AlignmentError(f"{other} does not share the same path with {self} before {self.current_node}")
        descendants = list(nx.descendants(other.g, self.current_node.nx_label))
        cropped = nx.subgraph(other.g, descendants + [self.current_node.nx_label])
        relabel = {d: d - self.current_node.nx_label - 1 + self.node_counter for d in cropped.nodes}
        del relabel[self.current_node.nx_label]
        self.node_counter += len(relabel)
        nx.relabel_nodes(cropped, relabel, copy=False)
        self.g = nx.compose(self.g, cropped)  # join graphs
        for d in descendants:
            self.g.nodes[d]['column'] += 1

    def filter_by_condition(self, condition: Condition):
        """
        This joins offshot branches back to their parent with a condition
        The condition can only rely on the current hierarchy node,
        or any terminating nodes of branches from the current hierarchy node
        """
        available_nodes = list(self.available_nodes())
        for node in condition.nodes:
            if node not in available_nodes:
                raise ValueError(f'Cannot join {node} to graph based on {self.current_node}')
        filtered = Filter(condition=condition, subject=self.current_node)
        self.g.add_node(self.node_counter, label=filtered, column=0, row=-self.node_counter,
                        ntype='condition')
        self.g.add_edge(self.current_node.nx_label, self.node_counter, etype='condition')
        self.current_node = NodeReference(self, self.node_counter)
        self.node_counter += 1
        return self.current_node


    def add_function(self, function):
        pass


if __name__ == '__main__':

    q1 = QueryTree('OB')
    q1.add_hierarchy('Exposure')
    q1.add_hierarchy('Run')  # ob.exposures.runs

    q2 = QueryTree('OB')
    q2.add_hierarchy('Exposure')
    q2.add_hierarchy('Run')
    config2 = q2.add_hierarchy('Config')  # ob.exposures.runs.configs

    q3 = QueryTree('OB')
    q3.add_hierarchy('Exposure')
    q3.add_hierarchy('Run')
    config3 = q3.add_hierarchy('Config')  # ob.exposures.runs.configs


    # q2.filter_by_condition(Condition(config2.camera, '=', 'red'))
    # q1.add_branch(q2)
    #
    # q2.add_branch(q
    # q1.filter_by_condition(Condition(config3.camera, '=', 'blue'))




    fig = q2.draw_query()
    import matplotlib.pyplot as plt
    plt.savefig('query.pdf')