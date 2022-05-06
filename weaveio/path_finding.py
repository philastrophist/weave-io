import networkx as nx

def shortest_simple_paths(graph, source, target):
    try:
        yield from nx.shortest_simple_paths(graph, source, target)
    except nx.NetworkXNoPath:
        yield []


def _find_path(graph, a, b, force_single):
    singles = nx.subgraph_view(graph, filter_edge=lambda u, v: graph.edges[u, v]['singular'])
    single_paths = [i for i in [next(shortest_simple_paths(singles, a, b)), next(shortest_simple_paths(singles, b, a))[::-1]] if i]
    if single_paths:
        return min(single_paths, key=len)
    elif force_single:
        raise nx.NetworkXNoPath('No path found between {} and {}'.format(a, b))
    restricted = nx.subgraph_view(graph, filter_edge=lambda u, v: 'relation' in graph.edges[u, v])
    paths = [i for i in [next(shortest_simple_paths(restricted, a, b)), next(shortest_simple_paths(restricted, b, a))[::-1]] if i]
    if paths:
        return min(paths, key=len)
    raise nx.NetworkXNoPath('No path found between {} and {}'.format(a, b))

def find_path(graph, a, b, force_single):
    path = _find_path(graph, a, b, force_single)
    if len(path) == 1:
        return [path[0], path[0]]
    return path