import itertools

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
abstract_hierarchy = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
factor = {'type': 'factor', 'style': 'filled', 'fillcolor': orange, 'shape': 'box', 'edgecolor': orange}
identity = {'type': 'id', 'style': 'filled', 'fillcolor': purple, 'shape': 'box', 'edgecolor': purple}
product = {'type': 'factor', 'style': 'filled', 'fillcolor': pink, 'shape': 'box', 'edgecolor': pink}
l1file = {'type': 'file', 'style': 'filled', 'fillcolor': lightblue, 'shape': 'box', 'edgecolor': lightblue}
l2file = {'type': 'file', 'style': 'filled', 'fillcolor': lightgreen, 'shape': 'box', 'edgecolor': lightgreen}
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
    'OBSpec': (hierarchy, ['ObsTemp', 'ProgTemp', 'TargetSet', 'OBTitle']),
    'OBRealisation': (abstract_hierarchy, ['OBSpec', 'OBRepeat', 'OBID']),
    'Exposure': (abstract_hierarchy, ['OBRealisation', 'ExpMJD']),
    'Run': (hierarchy, ['Exposure', 'VPH', 'RunID']),

    'Raw': (rawfile, ['Run']),
    'L1 single': (l1file, ['Run']),
    'L1 stack': (l1file, ['OBRealisation', 'VPH']),
    'L1 superstack': (l1file, ['VPH', 'OBSpec']),
    'L1 supertarget': (l1file, ['Mode', 'Arm config', 'Target', 'Binning']),


    'L2 single': (l2file, ['Exposure']),  # join the two runs together
    'L2 stack/superstack': (l2file, ['TargetSet', 'Mode', MultiEdge('Arm config', 3), 'Binning']),
    'L2 supertarget': (l2file, ['Mode', MultiEdge('Arm config', 3), 'Target', 'Binning']),

    'ObsTemp': (hierarchy, ['MaxSeeing', 'MinTrans', 'MinElev', 'MinMoon', 'MaxSky']),
    'ProgTemp': (hierarchy, ['Mode', MultiEdge('Arm config', 2, 1), 'Binning']),
    'Mode': (factor, []),

    'Target': (abstract_hierarchy, ['TargetName']),
    'TargetSet': (hierarchy, [MultiEdge('Target')]),
    'Arm config': (abstract_hierarchy, ['VPH', 'Resolution']),
    'Resolution': (factor, []),
    'OBTitle': (factor, []),
    'OBRepeat': (factor, []),
    'ExpMJD': (factor, []),
    'Binning': (factor, []),
    'VPH': (factor, []), # green,blue,red. Can have ['B', 'G'] to do an "or"

    'MaxSeeing': (factor, []),
    'MinTrans': (factor, []),
    'MinElev': (factor, []),
    'MinMoon': (factor, []),
    'MaxSky': (factor, []),

    'OBID': (identity, []),
    'RunID': (identity, []),
    'TargetName': (identity, []),
}

DATA = {
    'single_r0001.fits': ['L1 single', {
        'RunID': 'r0001',
        'VPH': 'red',
        'ExpMJD': 0,
        'OBRepeat': 0,
        'MaxSeeing': 'S', 'MinTrans': 'T', 'MinElev': 'A', 'MinMoon': 'M', 'MaxSky': 'B',
        'OBTitle': 'MyOB0',
        'OBID': 0,
        'TargetName': ['target1', 'target2', 'target3', 'target4'],
        'Mode': 'MOS',
        'Resolution': 'L',
        'Binning': 2
    }],
    'single_r0002.fits': ['L1 single', {
        'RunID': 'r0002',
        'VPH': 'blue',
        'ExpMJD': 0,
        'OBRepeat': 0,
        'MaxSeeing': 'S', 'MinTrans': 'T', 'MinElev': 'A', 'MinMoon': 'M', 'MaxSky': 'B',
        'OBTitle': 'MyOB0',
        'OBID': 0,
        'Target': ['target1', 'target2', 'target3', 'target4'],
        'Mode': 'MOS',
        'Resolution': 'L',
        'Binning': 2,
    }],
    'single_r0003.fits': ['L1 single', {
        'RunID': 'r0003',
        'VPH': 'green',
        'ExpMJD': 0,
        'OBRepeat': 0,
        'MaxSeeing': 'S', 'MinTrans': 'T', 'MinElev': 'A', 'MinMoon': 'M', 'MaxSky': 'B',
        'OBTitle': 'MyOB1',
        'OBID': 1,
        'Target': ['target5', 'target6', 'target7', 'target8'],
        'Mode': 'MOS',
        'Resolution': 'H',
        'Binning': 2,
    }],
    'single_r0004.fits': ['L1 single', {
        'RunID': 'r0004',
        'VPH': 'green',
        'ExpMJD': 0,
        'OBRepeat': 0,
        'OBID': 2,
        'MaxSeeing': 'S', 'MinTrans': 'T', 'MinElev': 'A', 'MinMoon': 'M', 'MaxSky': 'B',
        'OBTitle': 'MyOB2',
        'Target': ['target5', 'target6', 'target7', 'target8'],
        'Mode': 'MOS',
        'Resolution': 'H',
        'Binning': 2
    }],
    'l1supertarget_8.fits': ['L1 supertarget', {
        'RunID': 'r0003',
        'VPH': 'red',
        'ExpMJD': 0,
        'OBRepeat': 0,
        'MaxSeeing': 'S', 'MinTrans': 'T', 'MinElev': 'A', 'MinMoon': 'M', 'MaxSky': 'B',
        'OBTitle': 'MyOB3',
        'OBID': 3,
        'Target': 'target8',
        'Mode': 'MOS',
        'Resolution': 'H',
        'Binning': 2
    }],
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


def traverse_backwards(graph: nx.DiGraph, start_node):
    nodes = [start_node]
    for node in nodes:
        graph.predecessors(node)


if __name__ == '__main__':
    relation_graph = nx.DiGraph()
    for to_node, (attr, from_nodes) in RELATIONS.items():
        relation_graph.add_node(to_node, **attr)
    for to_node, (attr, from_nodes) in RELATIONS.items():
        for from_node in from_nodes:
            label = '1' if isinstance(from_node, str) else  from_node.label
            minnumber = 1 if isinstance(from_node, str) else from_node.minnumber
            maxnumber = 1 if isinstance(from_node, str) else from_node.maxnumber
            from_node = from_node if isinstance(from_node, str) else from_node.name
            relation_graph.add_edge(from_node, to_node, color=relation_graph.nodes[to_node]['edgecolor'],
                                    headlabel=label)
    # G = nx.subgraph_view(relation_graph, lambda n: relation_graph.nodes[n]['type'] in ['hierarchy', 'file'])
    G = relation_graph
    graph2pdf(G, 'relation_graph')

    # # instance_graph = nx.DiGraph()
    # for fname, (ftype_name, factors_ids) in DATA.items():
    #     factors_ids = {k: vs if isinstance(vs , list) else [vs] for k, vs in factors_ids.items()}
    #     factors = {k: vs for k, vs in factors_ids.items() if relation_graph.nodes[k]['type'] == 'factor'}
    #     ids = {k: vs for k, vs in factors_ids.items() if relation_graph.nodes[k]['type'] == 'id'}
    #
    #     ancestors = list(nx.ancestors(relation_graph, ftype_name))
    #     subgraph = nx.subgraph(relation_graph, ancestors)  # type: nx.DiGraph
    #     instance_graph = nx.create_empty_copy(subgraph) # type: nx.DiGraph
    #     for factor_name, factor_values in factors.items():
    #         for factor_value in factor_values:
    #             n = f"{factor_name}-{factor_value}"
    #             instance_graph.add_node(n, **subgraph.nodes[factor_name])
    #             instance_graph.add_edge(factor_name, n)
    #     for id_name, id_values in ids.items():
    #         hierarchy = list(subgraph.successors(id_name))[0]
    #         for id_value in id_values:
    #             n = f"{hierarchy}-{id_value}"
    #             instance_graph.add_node(n, **subgraph.nodes[hierarchy])
    #     while True:
    #         for node, attrs in subgraph.nodes(data=True):
    #             if attrs['type'] == 'hierarchy':
    #                 if all(subgraph.predecessors(node):
    #
    #
    #
    #
    #
    #
    #
    #     break



    # graph2pdf(subgraph, 'instance_graph')



