from typing import List, TYPE_CHECKING

import textdistance
import networkx as nx

if TYPE_CHECKING:
    from .base import BaseQuery
    from .objects import ObjectQuery

__all__ = ['attributes', 'objects', 'explain', 'find']


def attributes(query: 'BaseQuery', plural=False, directly_owned_only=False) -> List[str]:
    from ..data import Data
    from .objects import ObjectQuery
    if isinstance(query, Data):
        return list(query.plural_factors.keys())
    if not isinstance(query, ObjectQuery):
        raise TypeError(f"{query} is not an ObjectQuery or Data")
    if directly_owned_only:
        return sorted([i.lower() for i in query._data.class_hierarchies[query._obj].products_and_factors])
    o = query._data.class_hierarchies[query._obj]
    def _test(n):
        if n in G:
            return nx.has_path(G, o, n)
        return False
    G = query._data.hierarchy_graph.singular
    reachable = set(G.subgraph_view(_test).nodes) - {o}
    attrs = {f'{n.singular_name}.{i.lower()}' for n in reachable for i in n.products_and_factors}
    if plural:
        G = query._data.hierarchy_graph.plural
        reachable = set(G.subgraph_view(_test).nodes) - {o}
        attrs |= {f'{n.plural_name}.{i.lower()}' for n in reachable for i in n.products_and_factors}
    return sorted([i.lower() for i in o.products_and_factors]) + sorted(attrs)

def _objects(query: 'BaseQuery'):
    o = query._data.class_hierarchies[query._obj]
    def _test(n):
        if n in G:
            return nx.has_path(G, o, n)
        return False
    G = query._data.hierarchy_graph.singular
    singular = set(G.subgraph_view(_test).nodes) - {o}
    G = query._data.hierarchy_graph.plural
    plural = set(G.subgraph_view(_test).nodes) - {o}
    return singular - {o}, plural - {o}

def objects(query: 'BaseQuery', plural=True) -> List[str]:
    from ..data import Data
    if isinstance(query, Data):
        return list(query.plural_hierarchies.keys())
    ss, ps = _objects(query)
    if not plural:
        ps = []
    rels = list(query._data.class_hierarchies[query._obj].relative_names.keys())
    return sorted([s.singular_name.lower() for s in ss] + rels) + sorted([p.plural_name.lower() for p in ps])

def explain(query: 'BaseQuery') -> None:
    from ..data import Data
    if isinstance(query, Data):
        print(f"Database connection {query}")
    if query._obj is None:
        raise TypeError(f"{query} is not a ObjectQuery")
    print(f"{query._obj}:")
    print(f"\t {query._data.class_hierarchies[query._obj].__doc__.strip()}")

def find(query: 'BaseQuery', guess: str, n=10) -> List[str]:
    from ..data import Data
    from .objects import AttributeQuery
    if isinstance(query, Data):
        return query.find_names(guess)
    if isinstance(query, AttributeQuery):
        raise TypeError(f"You cannot search for things in an attribute query. Try using find on an object instead.")
    ss, ps = _objects(query)
    objs = [s.singular_name for s in ss] + [p.plural_name for p in ps]
    factors = [i for s in ss for i in s.products_and_factors] + [query._data.plural_name(i) for p in ps for i in p.products_and_factors]
    sources = objs + factors
    inorder = sorted(sources, key=lambda x: textdistance.jaro_winkler(guess, x), reverse=True)
    return [i.lower() for i in inorder[0:n]]