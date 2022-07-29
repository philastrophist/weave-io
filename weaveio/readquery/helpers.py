from typing import List

from .base import BaseQuery


def attributes(query: BaseQuery) -> List[str]:
    from ..data import Data
    if isinstance(query._data, Data):
        return list(query._data.singular_factors.keys())
    return query._data.class_hierarchies[query._obj].products_and_factors

def objects(query: BaseQuery) -> List[str]:
    from ..data import Data
    if isinstance(query._data, Data):
        return list(query._data.singular_hierarchies.keys())
    h = query._data.class_hierarchies[query._obj]
    neighbors = [i.singular_name for i in query._data.hierarchy_graph.neighbors(h)]
    return neighbors

def explain(query: BaseQuery) -> None:
    print(f"{query._obj}:")
    print(f"\t {query._data.class_hierarchies[query._obj].__doc__.strip()}")

def find(query: BaseQuery, guess: str) -> List[str]:
    return query._data.find_names(guess)