import math
from itertools import zip_longest

import networkx as nx

def shortest_simple_paths(graph, source, target, weight=None):
    try:
        yield from nx.shortest_simple_paths(graph, source, target, weight)
    except nx.NetworkXNoPath:
        yield []


def _find_singular_simple_hierarchy_path(graph, start, end, optionals=False):
    """
    Assuming the graph contains only edges from top to bottom which are labelled singular and not optional.
    x-[singular]->y "y has 1 x", "x can have many y"
    In this case, we always look for start<-end first otherwise look for end<-start (and reverse the path)
    """
    def filt(u, v):
        edge = graph.edges[u, v]
        return edge['singular'] and (not edge['optional'] if optionals else True)  # this only works if all paths have equal weight otherwise
    g = nx.subgraph_view(graph, filter_edge=filt)
    try:
        path = next(nx.shortest_simple_paths(g, end, start, 'optional'))[::-1]
    except nx.NetworkXNoPath:
        path = next(nx.shortest_simple_paths(g, start, end, 'optional'))
    return path


def find_singular_simple_hierarchy_path(graph, start, end):
    """
    If there is a path without optionals then take it, otherwise we want the fewest optional edges
    """
    return _find_singular_simple_hierarchy_path(graph, start, end, optionals=False)



def _find_path(graph, a, b, force_single):
    singles = nx.subgraph_view(graph, filter_edge=lambda u, v: graph.edges[u, v]['singular'])
    # first try to find a singular path
    single_paths = [i for i in [next(shortest_simple_paths(singles, a, b)), next(shortest_simple_paths(singles, b, a))[::-1]] if i]
    if single_paths:
        return min(single_paths, key=len)
    elif force_single:
        raise nx.NetworkXNoPath('No path found between {} and {}'.format(a, b))
    # then try to find a non-singular path with one direction, either -> or <-
    # we dont want ()->()<-() when it's multiple
    restricted = nx.subgraph_view(graph, filter_edge=lambda u, v: graph.edges[u, v]['actual_number'] > 0)
    forward = next(shortest_simple_paths(restricted, a, b, 'actual_number'))
    forward_weight = nx.path_weight(restricted, forward, 'actual_number') or math.inf
    reversed = restricted.reverse()
    backward = next(shortest_simple_paths(reversed, a, b, 'actual_number'))
    backward_weight = nx.path_weight(reversed, backward, 'actual_number') or math.inf
    if not forward and not backward:
        raise nx.NetworkXNoPath('No path found between {} and {}'.format(a, b))
    if backward_weight < forward_weight:
        return backward
    return forward

def shortest_simple_paths_with_weight(graph, a, b, weight):
    for path in shortest_simple_paths(graph, a, b, weight):
        yield path, nx.path_weight(graph, path, weight)

def find_paths(graph, a, b, force_single):
    """
    generates all unidirectional paths from a->b or b<-a with the same shortest length.
    Conditions:
        1. Cannot mix -> and <-, since that would allow nonsense like l2single->aps<-l2stack
        2. Prefer paths with as few "optional" edges as possible
        3. Prefer paths that are single in some direction.
    In the case of generic traversals e.g. `ingested.camera`, there are multiple paths
    """
    # should decide how to include optional edges
    real = nx.subgraph_view(graph, filter_edge=lambda u, v: graph.edges[u, v]['actual_number'] > 0 and 'relation' in graph.edges[u, v])
    flipped = nx.subgraph_view(graph, filter_edge=lambda *e: e[::-1] in real.edges)
    real_paths = shortest_simple_paths_with_weight(real, a, b, weight='actual_number')
    flipped_paths = shortest_simple_paths_with_weight(flipped, a, b, weight='actual_number')
    paths = ((p, w) for paths in zip_longest(real_paths, flipped_paths, fillvalue=(None, None)) for p, w in paths if p)
    buffer = []
    buffer_weight = None
    for path, weight in paths:
        length = len(path)
        if length == buffer_weight or buffer_weight is None:
            buffer.append(path)
            buffer_weight = length
        elif length > buffer_weight:
            break
        elif length < buffer_weight:
            buffer = [path]
            buffer_weight = length
    if buffer:
        return buffer
    raise nx.NetworkXNoPath(f"No path found between {a} and {b}")
    # equivalent paths will be removed later at the arrow stage








def find_path(graph, a, b, force_single):
    paths = find_paths(graph, a, b, force_single)
    return paths[0]
    # nonoptional = nx.subgraph_view(graph, filter_edge=lambda *e: not graph.edges[e]['optional'])
    # try:
    #     path = _find_path(nonoptional, a, b, force_single)
    # except nx.NetworkXNoPath:
    #     path = _find_path(graph, a, b, force_single)
    # if len(path) == 1:
    #     return [path[0], path[0]]
    # return path