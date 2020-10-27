from collections import defaultdict
from typing import Dict, Any, Union, List, Type

from weaveio.file import File
from weaveio.hierarchy import Hierarchy




def parse_apoc_tree(root_hierarchy: Type['Hierarchy'], root_id: Any, tree: Dict[str, Any],
                    data) -> Union['Hierarchy', List['Hierarchy']]:
    hierarchies = data.class_hierarchies
    subclassed = {k.__name__: len(k.__subclasses__()) for k in hierarchies.values()}
    inputs = defaultdict(list)
    for key, value in tree.items():
        if key.startswith('_') or key == 'id':
            continue
        elif isinstance(value, list):  # preceeding relationship
            value.sort(key=lambda x: x.pop(f'{key}.order', 0))
            for entry in value:
                names = [n for n in entry['_type'].split(':') if n in hierarchies]
                names.sort(key=lambda x: subclassed[x])
                name = names[0]
                # if issubclass(hierarchies[name], File):
                #     h = hierarchies[name](fname=entry['id'])
                h = parse_apoc_tree(hierarchies[name], entry['id'], entry, data)
                requirements = root_hierarchy.requirement_names()
                if h.singular_name in requirements:
                    inputs[h.singular_name] = h
                elif h.plural_name in requirements:
                    inputs[h.plural_name].append(h)
                else:
                    raise TypeError(f"{root_hierarchy} does not require {h}. This should not happen")
        elif isinstance(value, (int, float, str)):
            inputs[key.lower()] = value
        else:
            raise ValueError(f"Invalid json schema")
    h = root_hierarchy(**inputs)
    h.identifier = root_id
    h.add_parent_data(data)
    return h