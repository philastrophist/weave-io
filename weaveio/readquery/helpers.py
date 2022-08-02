from typing import List, Tuple, Set

import textdistance

from .base import BaseQuery

__all__ = ['attributes', 'objects', 'explain', 'find']


def attributes(query: BaseQuery) -> List[str]:
    from ..data import Data
    if isinstance(query, Data):
        return list(query.plural_factors.keys())
    return query._data.class_hierarchies[query._obj].products_and_factors

def _objects(query: BaseQuery):
    h = query._data.class_hierarchies[query._obj]
    singular = {e[1] for *e, d in query._data.hierarchy_graph.edges(h, data=True) if d.get('singular', False)}
    singular |= {e[1] for *e, d in query._data.hierarchy_graph.in_edges(h, data=True) if d.get('singular', False)}
    plural = {e[1] for *e, d in query._data.hierarchy_graph.edges(h, data=True) if not d.get('singular', True)}
    plural |= {e[1] for *e, d in query._data.hierarchy_graph.in_edges(h, data=True) if not d.get('singular', True)}
    return singular, plural

def objects(query: BaseQuery) -> List[str]:
    from ..data import Data
    if isinstance(query, Data):
        return list(query.plural_hierarchies.keys())
    ss, ps = _objects(query)
    return [s.singular_name for s in ss] + [p.plural_name for p in ps]

def explain(query: BaseQuery) -> None:
    from ..data import Data
    if isinstance(query, Data):
        print(f"Database connection {query}")
    print(f"{query._obj}:")
    print(f"\t {query._data.class_hierarchies[query._obj].__doc__.strip()}")

def find(query: BaseQuery, guess: str, n=10) -> List[str]:
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
    return inorder[0:n]