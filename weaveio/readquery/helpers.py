from typing import List

from .base import BaseQuery


def attributes(query: BaseQuery) -> List[str]:
    return query._data.class_hierarchies[query._obj].products_and_factors

def explain(query: BaseQuery) -> None:
    print(f"{query._obj}:")
    print(f"\t {query._data.class_hierarchies[query._obj].__doc__.strip()}")
