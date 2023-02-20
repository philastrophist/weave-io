from typing import List
import networkx as nx


def detect_d_graph(g: nx.DiGraph, path: List[str]):
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        if c.startswith(f'({b}s)'):
            if (a, c) in g.edges:
                return True
    return False


def traverse_all(g, a, b):
    """
    Return paths top to bottom where paths that go via collections rule out other paths that also
    go via their individuals
    i.e
        for  ________
            /        \
            a-->b-->(bs)-->c
        a-->b-->(bs)-->c is ruled out in preference for a-->(bs)-->c
    """
    for path in nx.shortest_simple_paths(g, a, b):
        if not detect_d_graph(g, path):
            yield path


def traverse_direct(g, a, b):
    previous = []
    for i, path in enumerate(traverse_all(g, a, b)):
        if i == 0 or len(path) == len(previous):
            yield path
            previous = path


if __name__ == '__main__':
    import graphviz
    from networkx.drawing.nx_pydot import to_pydot


    def plot(g, fname, directory=None, ftype='pdf'):
        graphviz.Source(to_pydot(g).to_string()).render(fname, directory, format=ftype)


    G = nx.DiGraph()
    G.add_edges_from([('survey', 'targprog'),
                      ('survey', 'catalogue'),
                      ('survey', '(targprogs)1'),
                      ('survey', '(targprogs)2'),
                      ('targprog', '(targprogs)1'),
                      ('targprog', '(targprogs)2'),
                      ('(targprogs)1', 'catalogue'),
                      ('(targprogs)2', 'surveytarget'),
                      ('catalogue', 'surveytarget'),
                      # ('survey', 'surveytarget'),
                      # ('weavetarget', 'surveytarget'),
                      # ('surveytarget', 'fibretarget'),
                      # ('fibretarget', 'L1Single'),
                      # ('weavetarget', 'L1Supertarget'),
                      # ('L1Single', 'L1Template'),
                      # ('L1Supertarget', 'L1Template'),
                      # ('weavetarget', 'L1Singles'),
                      # ('L1Singles', 'L1Supertarget'),
                      # ('L1Single', 'L1Singles'),
                      # ('L1Template', 'Redrock')
                      ])
    plot(G, 'render')

    for path in traverse_direct(G, 'survey', 'surveytarget'):
        print(path)