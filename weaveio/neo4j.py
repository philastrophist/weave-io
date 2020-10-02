from typing import Dict, Any, Union, List, Type

from weaveio.hierarchy import Hierarchy
from weaveio.file import File




def parse_apoc_tree(root_hierarchy: Type[Hierarchy], root_id: Any, tree: Dict[str, Any],
                    data) -> Union[Hierarchy, List[Hierarchy]]:
    hierarchies = data.class_hierarchies
    subclassed = {k.__name__: len(k.__subclasses__()) for k in hierarchies.values()}
    inputs = {}
    for key, value in tree.items():
        if key.startswith('_') or key == 'id':
            continue
        elif isinstance(value, list):  # preceeding relationship
            value.sort(key=lambda x: x.pop(f'{key}.order', 0))
            for entry in value:
                if 'Factor' in entry['_type']:
                    name = entry['_type'].replace('Factor', '').strip(':').lower()
                    inputs[name] =  entry['value']
                else:
                    names = [n for n in entry['_type'].split(':') if n in hierarchies]
                    names.sort(key=lambda x: subclassed[x])
                    name = names[0]
                    h = parse_apoc_tree(hierarchies[name], entry['id'], entry, data)
                    if isinstance(h, File):
                        continue   # TODO: need to properly put files in the graph
                    if h.singular_name in inputs:
                        inputs[h.plural_name] = [inputs.pop(h.singular_name), h]
                    elif h.plural_name in inputs:
                        inputs[h.plural_name].append(h)
                    else:
                        inputs[h.singular_name] = h
        elif isinstance(value, (int, float, str)):
            inputs[key.lower()] = value
        else:
            raise ValueError(f"Invalid json schema")
    h = root_hierarchy(**inputs)
    h.identifier = root_id
    h.add_parent_data(data)
    return h