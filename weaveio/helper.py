import textwrap

import networkx as nx

from weaveio.basequery.hierarchy import HierarchyFrozenQuery
from weaveio.hierarchy import Graphable, GraphableMeta, Hierarchy


def _convert_obj(obj, data=None):
    if isinstance(obj, str):
        obj = obj.lower()
        try:
            obj = data.singular_hierarchies[obj]
        except KeyError:
            obj = data.plural_hierarchies[obj]
    elif isinstance(obj, HierarchyFrozenQuery):
        data = obj.handler.data
        obj = obj.hierarchy_type
    elif not isinstance(obj, (Graphable, GraphableMeta)):
        raise TypeError(f"{obj} is not a recognised type of object. Use a name (string), a query, or the type directly")
    return obj, data


def attributes(obj, data=None):
    obj, data = _convert_obj(obj, data)
    return sorted(set(obj.products_and_factors))


def objects(obj, data=None):
    obj, data = _convert_obj(obj, data)
    neighbors = list(data.relation_graph.predecessors(obj)) + list(data.relation_graph.successors(obj))
    relations = []
    for b in neighbors:
        try:
            data.find_hierarchy_paths(obj, b, plural=False)
        except (nx.NetworkXNoPath, KeyError):
            relations.append(b.plural_name)
        else:
            relations.append(b.singular_name)
    relations.sort()
    return relations


def explain(obj, data=None):
    obj, data = _convert_obj(obj, data)
    objs = objects(obj, data)
    attrs = attributes(obj, data)
    print('===========', obj.singular_name, '===========')
    if obj.__doc__:
        print('\n'.join(textwrap.wrap(textwrap.dedent('\n'.join(obj.__doc__.split('\n'))))))
        print()
    if obj.__bases__ != (Hierarchy, ):
        print(f"A {obj.singular_name} is a type of {obj.__bases__[0].singular_name}")
    if obj.idname is not None:
        print(f"A {obj.singular_name} has a unique id called '{obj.idname}'")
    else:
        print(f"A {obj.singular_name} has no unique id that can be used")
    if obj.identifier_builder:
        print(f"{obj.plural_name} are identified by {tuple(obj.identifier_builder)}")
    print(f'one {obj.singular_name} is linked to:')
    for o in objs:
        if data.is_plural_name(o):
            print('\t- many', o)
        else:
            print('\t- 1', o)
    print(f'a {obj.singular_name} directly owns these attributes:')
    for a in attrs:
        print('\t-', a)
    print('======================' + '='*len(obj.singular_name))