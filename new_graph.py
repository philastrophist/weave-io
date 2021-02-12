import networkx as nx
from collections import Counter, defaultdict
from networkx.drawing.nx_agraph import to_agraph


def plot(graph, fname):
    A = to_agraph(graph)
    A.layout('dot')
    A.draw(fname)


class Graph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.start = Start(self)
        self.add_node(self.start)

    def add_node(self, node, *parents, accessible=True, **attr):
        assert isinstance(node, Node)
        super(Graph, self).add_node(node)
        for p in parents:
            self.add_edge(p, node, accessible=accessible, **attr)

    def add_edge(self, a, b, accessible=True, **attr):
        return super(Graph, self).add_edge(a, b, accessible=accessible, **attr)

    def add_hierarchy(self, parent, name):
        h = Hierarchy(name, self, parent)
        self.add_node(h, parent)
        return h

    def add_operation(self, parent, name):
        assert isinstance(name, str)
        o = Operation(name, self, parent)
        self.add_node(o, parent)
        return o

    def relevant_graph(self, node):
        return nx.subgraph_view(self, lambda n: nx.has_path(self, n, node) or n is node)

    def accessible_graph(self, node):
        graph = self.relevant_graph(node)
        g = nx.subgraph_view(graph, filter_edge=lambda a, b: graph.edges[(a, b)]['accessible'])
        return nx.subgraph_view(g, filter_node=lambda x: nx.has_path(g, x, node))

    def find_first_hierarchy_parent(self, node):
        accessible = self.accessible_graph(node)
        if not len(accessible):
            return self.start
        reverse = accessible.reverse()
        for n in nx.shortest_path(reverse, node, self.start):
            if isinstance(n, Hierarchy) and n != node:
                break
        else:
            return self.start
        return n

    def add_filter(self, name, to_filter, by=None):
        if by is not None:
            by = self.add_collect(by, to_filter)
        wrt = self.find_first_hierarchy_parent(to_filter)
        f = Filter(name, self, to_filter, by)
        if by is not None:
            self.add_node(f, to_filter, by)
        else:
            self.add_node(f, to_filter)
        self.add_edge(wrt, f, accessible=True)
        return f

    def on_same_branch(self, a, b):
        return b in self.accessible_graph(a) or a in self.accessible_graph(b)

    def find_shared_parent(self, a, b):
        aorder = nx.shortest_path(self.accessible_graph(a), self.start, a)[::-1]
        border = nx.shortest_path(self.accessible_graph(b), self.start, b)[::-1]
        return [ai for ai, bi in zip(aorder, border) if ai == bi][-1]

    def add_align(self, a, b):
        if self.on_same_branch(a, b):
            x = Operation('align-branch', self, a, b)
            self.add_node(x, a, b)
            return x
        shared = self.find_shared_parent(a, b)
        a = self.add_collect(a, wrt=shared)
        b = self.add_collect(b, wrt=shared)
        align = Align(self, a, b, shared)
        self.add_node(align, a, b, accessible=False)
        self.add_edge(shared, align, accessible=True)
        return align

    def add_collect(self, node, wrt):
        c = Collect(self, node, wrt)
        self.add_node(c, node, wrt, accessible=False)
        self.edges[wrt, c]['accessible'] = True
        return c

    def add_aggregation(self, node, wrt, how):
        a = Aggregation(how, self, node, wrt)
        self.add_node(a, node, wrt)
        self.edges[wrt, a]['accessible'] = True
        return a


class Node:
    counter = defaultdict(int)

    def __init__(self, name, graph, *parents):
        Node.counter[name] += 1
        i = Node.counter[name]
        if i == 1:
            name = name
        else:
            name = name + f'{{{i}}}'
        self.name = name
        self.graph = graph
        self.parents = parents

    def __str__(self):
        return self.name


class Hierarchy(Node):
    pass


class Filter(Node):
    pass


class Operation(Node):
    pass


class Align(Node):
    def __init__(self, graph, *parents):
        super(Align, self).__init__('align', graph, *parents)


class Aggregation(Node):
    pass


class Collect(Aggregation):
    def __init__(self, graph, *parents):
        super(Collect, self).__init__('collect', graph, *parents)



class Start(Hierarchy):
    def __init__(self, graph):
        super().__init__('data', graph)



if __name__ == '__main__':
    graph = Graph()
    obs = graph.add_hierarchy(graph.start, 'ob')
    obid = graph.add_operation(obs, 'obid == 3130')
    ob = graph.add_filter('ob[obid == 3130]', obs, obid)
    l1spectra = graph.add_hierarchy(ob, 'l1spectra')
    l1spectra0 = graph.add_filter('l1spectra[0]', l1spectra)
    l1spectra0_snr = graph.add_operation(l1spectra0, 'snr')

    obs_l1spectra = graph.add_hierarchy(obs, 'l1spectra')
    camera = graph.add_hierarchy(obs_l1spectra, 'camera')
    is_red = graph.add_operation(camera, '== red')
    is_blue = graph.add_operation(camera, '== blue')

    red_spectra = graph.add_filter('l1spectra[red]', obs_l1spectra, is_red)
    blue_spectra = graph.add_filter('l1spectra[blue]', obs_l1spectra, is_blue)

    red_snr = graph.add_operation(red_spectra, 'snr')
    blue_snr = graph.add_operation(blue_spectra, 'snr')

    aligned_snr = graph.add_align(red_snr, blue_snr)
    red_better = graph.add_operation(aligned_snr, '>')
    all_red_better = graph.add_aggregation(red_better, obs, 'all')
    best_obs = graph.add_filter('obs[all(red.snr > blue > snr]', obs, all_red_better)
    snrs = graph.add_operation(best_obs, 'snr')

    aligned = graph.add_align(snrs, l1spectra0_snr)
    graph.add_operation(aligned, '>')


    for n in nx.topological_sort(graph):
        print(n)
    plot(graph, 'new-graph.png')