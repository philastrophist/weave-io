import networkx as nx
import numpy as np
from graphviz import Source

def graph2pdf(graph, ftitle):
    dot = nx.nx_pydot.to_pydot(graph)
    dot.set_strict(False)
    # dot.obj_dict['attributes']['splines'] = 'ortho'
    dot.obj_dict['attributes']['nodesep'] = '0.5'
    dot.obj_dict['attributes']['ranksep'] = '0.75'
    dot.obj_dict['attributes']['overlap'] = False
    dot.obj_dict['attributes']['penwidth'] = 18
    dot.obj_dict['attributes']['concentrate'] = False
    Source(dot).render(ftitle, cleanup=True, format='pdf')
#
lightblue = '#69A3C3'
lightgreen = '#71C2BF'
red = '#D08D90'
orange = '#DFC6A1'
purple = '#a45fed'
pink = '#d50af5'


hierarchy = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
abstract_hierarchy = {'type': 'abstract_hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
factor = {'type': 'factor', 'style': 'filled', 'fillcolor': orange, 'shape': 'box', 'edgecolor': orange}
identity = {'type': 'factor', 'style': 'filled', 'fillcolor': purple, 'shape': 'box', 'edgecolor': purple}
product = {'type': 'factor', 'style': 'filled', 'fillcolor': pink, 'shape': 'box', 'edgecolor': pink}
l1file = {'type': 'l1file', 'style': 'filled', 'fillcolor': lightblue, 'shape': 'box', 'edgecolor': lightblue}
l2file = {'type': 'l2file', 'style': 'filled', 'fillcolor': lightgreen, 'shape': 'box', 'edgecolor': lightgreen}
choice = {'type': 'choice', 'style': 'filled', 'fillcolor': red, 'shape': 'diamond', 'edgecolor': red}
rawfile = l1file


class Edge:
    def __init__(self, name):
        self.name = name


class MultiEdge(Edge):
    def __init__(self, name, maxnumber=None, minnumber=1):
        super().__init__(name)
        self.maxnumber = maxnumber
        self.minnumber = minnumber
        if maxnumber is None:
            self.label = f'>{minnumber}'
        else:
            self.label = f"{minnumber} - {maxnumber}" if minnumber != maxnumber else f"{maxnumber}"



RELATIONS = {
    'OBSpec': (hierarchy, ['ObsTemp', 'ProgTemp', 'TargetSet', 'Title', 'OBID']),
    'OB realisation': (abstract_hierarchy, ['OBSpec', 'Repeat']),
    'Exposure': (abstract_hierarchy, ['OB realisation', 'Order']),
    'Run': (hierarchy, ['Exposure', 'VPH', 'RunID']),
    'Raw': (rawfile, ['Run']),

    'L1 single': (l1file, ['Run']),
    'L1 stack': (l1file, ['OB realisation', 'VPH']),
    'L1 superstack': (l1file, ['VPH', 'OBSpec']),
    'L1 supertarget': (l1file, ['Mode', 'Arm config', 'Target', 'Binning']),

    'L2 single': (l2file, ['Exposure']),  # join the two runs together
    'L2 stack/superstack': (l2file, ['TargetSet', 'Mode', MultiEdge('Arm config', 3), 'Binning']),
    'L2 supertarget': (l2file, ['Mode', MultiEdge('Arm config', 3), 'Target', 'Binning']),

    'ObsTemp': (factor, []),
    'ProgTemp': (hierarchy, ['Mode', MultiEdge('Arm config', 2, 1), 'Binning']),
    'Mode': (factor, []),

    'Target': (abstract_hierarchy, ['Target_name']),
    'TargetSet': (hierarchy, [MultiEdge('Target')]),

    # 'Red arm config': (abstract_hierarchy, ['Red arm', 'Resolution', 'Binning']),
    # 'Blue arm config': (abstract_hierarchy, ['Blue arm', 'Blue VPH', 'Resolution', 'Binning']),
    'Arm config': (abstract_hierarchy, ['VPH', 'Resolution']),

    # 'Red arm': (hierarchy, []),
    # 'Blue arm': (hierarchy, []),

    # 'Arm config': (choice, ['Red arm config', 'Blue arm config']),
    # 'Arm': (choice, ['Red arm', 'Blue arm']),

    'Resolution': (factor, []),
    'Title': (factor, []),
    'Repeat': (factor, []),
    'Order': (factor, []),
    'Binning': (factor, []),
    'VPH': (factor, []), # green,blue,red. Can have ['B', 'G'] to do an "or"

    'OBID': (identity, []),
    'RunID': (identity, []),
    'Target_name': (identity, []),
}


def find_highest_common_descendant(graph, a, b):
    """
    Find the highest common descendants in the directed, acyclic graph of node a and b.

    Arguments:
    ----------
        graph: networkx.DiGraph instance
            directed, acyclic, graph
        a, b:
            node IDs

    Returns:
    --------
        hcd: [node 1, ..., node n]
            list of highest common descendants nodes (can be more than one)
    """
    assert nx.is_directed_acyclic_graph(graph), "Graph has to be acyclic and directed."
    # get ancestors of both (intersection)
    common_descendants = list(nx.descendants(graph, a) & nx.descendants(graph, b))
    # get sum of path lengths
    sum_of_path_lengths = np.zeros((len(common_descendants)))
    for ii, c in enumerate(common_descendants):
        sum_of_path_lengths[ii] = nx.shortest_path_length(graph, a, c) \
                                  + nx.shortest_path_length(graph, b, c)
    minima, = np.where(sum_of_path_lengths == np.min(sum_of_path_lengths))

    return [common_descendants[ii] for ii in minima]



if __name__ == '__main__':
    graph = nx.DiGraph()
    for to_node, (attr, from_nodes) in RELATIONS.items():
        graph.add_node(to_node, **attr)
    for to_node, (attr, from_nodes) in RELATIONS.items():
        for from_node in from_nodes:
            label = '1' if isinstance(from_node, str) else  from_node.label
            minnumber = 1 if isinstance(from_node, str) else from_node.minnumber
            maxnumber = 1 if isinstance(from_node, str) else from_node.maxnumber
            from_node = from_node if isinstance(from_node, str) else from_node.name
            graph.add_edge(from_node, to_node, color=graph.nodes[to_node]['edgecolor'],
                           headlabel=label)#, labeldistance=0)
    graph2pdf(graph, 'graph')

