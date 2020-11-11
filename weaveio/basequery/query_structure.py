from itertools import zip_longest
from typing import List
import networkx as nx

from weaveio.basequery.query_objects import Node, NodeProperty, Path, Condition, Generator


class Filter:
    def __init__(self, condition, subject):
        self.condition = condition
        self.subject = subject

    def __str__(self):
        return f"{self.subject}[{self.condition}]"


class AlignmentError(Exception):
    pass


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

    def __init__(self, source: Node):
        self.g = nx.OrderedMultiDiGraph()
        self.g.add_node(source, column=0, row=0)
        self.row = -1
        self.current_node = source

    def hanging_nodes(self):
        return [n for n in self.g.nodes if len(list(self.g.successors(n))) == 0]

    def draw_query(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        layout = {n: (data['column'], data['row']) for n, data in self.g.nodes(data=True)}
        nx.draw_networkx_nodes(self.g, layout, with_labels=True)
        nx.draw_networkx_labels(self.g, layout, labels={n: str(n) for n in self.g.nodes})
        # nx.draw(self.g, ax=ax, pos=layout, with_labels=True)
        for edge in self.g.edges:
            if self.g.edges[edge]['etype'] == 'filtered':
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
                nx.draw_networkx_edges(self.g, layout, [edge])
        # ax.set_ylim(2, self.row-2)
        maxcolumn = max(c for _, c in self.g.nodes.data('column'))
        ax.set_xlim(-2, maxcolumn+2)
        plt.draw()
        return fig

    def add_hierarchy(self, to: Node):
        self.g.add_node(to, column=0, row=self.row)
        self.g.add_edge(self.current_node, to, etype='hierarchy')
        self.row -= 1
        self.current_node = to

    def add_branch(self, other: 'QueryTree'):
        for i, (a, b) in enumerate(zip_longest(other.g.nodes, self.g.nodes)):
            if a == self.current_node:
                break
            elif a != b:
                raise AlignmentError(f"{other} does not align with {self} on {self.current_node}")
        successors = list(other.g.successors(a))
        cropped = nx.subgraph(other.g, successors + [self.current_node])
        for n in successors:
            cropped.nodes[n]['column'] += 1
        self.row -= 1
        self.g = nx.compose(self.g, cropped)


    def filter_by_condition(self, condition: Condition):
        """
        You can only add conditions on free nodes
        1. Add resultant node (the one that will have been filtered)
        2. Add edge from source to resultant
        3. Add edges from
        """
        hanging_nodes = self.hanging_nodes()
        filtered = Filter(condition=condition, subject=self.current_node)
        self.g.add_node(filtered, column=0, row=self.row)
        self.row -= 1
        for node in condition.nodes:
            if node not in hanging_nodes:
                raise ValueError(f'Cannot join {node} to graph based on {self.current_node}')
            self.g.add_edge(node, filtered, etype='filtered')
        self.g.add_edge(self.current_node, filtered, etype='hierarchy')



    def add_function(self, function):
        pass


if __name__ == '__main__':

    g1 = Generator()
    ob1, exposures1, runs1 = g1.nodes('OB', 'Exposure', 'Run')
    q1 = QueryTree(ob1)
    q1.add_hierarchy(exposures1)
    q1.add_hierarchy(runs1)  # ob.exposures.runs

    g2 = Generator()
    ob2, exposures2, runs2, configs2 = g2.nodes('OB', 'Exposure', 'Run', 'Config')
    q2 = QueryTree(ob2)
    q2.add_hierarchy(exposures2)
    q2.add_hierarchy(runs2)
    q2.add_hierarchy(configs2)  # ob.exposures.runs.configs

    g3 = Generator()
    ob3, exposures3, runs3, configs3 = g3.nodes('OB', 'Exposure', 'Run', 'Config')
    q3 = QueryTree(ob3)
    q3.add_hierarchy(exposures3)
    q3.add_hierarchy(runs3)
    q3.add_hierarchy(configs3)  # ob.exposures.runs.configs


    q1.add_branch(q2)
    q1.filter_by_condition(Condition(configs2.camera, '=', 'red'))

    q1.add_branch(q3)
    q1.filter_by_condition(Condition(configs3.camera, '=', 'blue'))







    fig = q1.draw_query()
    import matplotlib.pyplot as plt
    plt.savefig('query.pdf')