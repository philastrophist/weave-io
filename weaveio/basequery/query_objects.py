from collections import defaultdict
from copy import deepcopy
from typing import Union, List

from weaveio.utilities import quote


class Copyable:
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Aliasable(Copyable):
    def __init__(self, name, alias=None):
        self.name = name
        self.alias = alias

    @property
    def alias_name(self):
        return self.name if self.alias is None else self.alias

    @property
    def context_string(self):
        if self.name == self.alias or self.alias is None:
            return self.name
        return f"{self.name} as {self.alias}"


class Node(Aliasable):
    def __init__(self, label=None, name=None, alias=None, **properties):
        super(Node, self).__init__(name, alias)
        self.label = label
        self.properties = properties

    @property
    def node(self):
        return self

    def identify(self, idvalue):
        self.properties['id'] = idvalue

    def stringify(self, mentioned_nodes):
        if self in mentioned_nodes:
            return f'({self.name})'
        mentioned_nodes.append(self)
        return str(self)

    def __hash__(self):
        return hash(''.join(map(str, [self.label, self.name, self.alias, self.properties])))

    def __repr__(self):
        name = '' if self.name is None else self.name
        label = '' if self.label is None else f':{self.label}'
        if self.properties:
            properties = ''
            for k, v in self.properties.items():
                properties += f'{k}: {quote(v)}'
            return f"({name}{label} {{{properties}}})"
        return f"({name}{label})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.label == other.label) and \
               ((self.name == other.name) or (self.name is None and other.name is None)) and \
               list(self.properties.items()) == list(other.properties.items())

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(f"{self} has no attribute {item}")
        return NodeProperty(self, item)


class NodeProperty(Aliasable):
    def __init__(self, node, property_name, alias=None):
        if alias is None:
            alias = f'{node.name}_{property_name}'
        super(NodeProperty, self).__init__(f"{node.name}.{property_name}", alias)
        self.node = node
        self.property_name = property_name

    def stringify(self, mentioned_nodes):
        n = self.node.stringify(mentioned_nodes)
        s = f"{n}.{self.property_name}"
        return s

    def __repr__(self):
        return f"{self.stringify([])}"

    def __eq__(self, other):
        return self.node == other.node and self.property_name == other.property_name


class Collection(Aliasable):
    def __init__(self, obj: Union[Node, NodeProperty], alias: str):
        super().__init__(f'collect({obj.name})', alias)
        self.obj = obj
        self.node = obj.node


class Path(Copyable):
    def __init__(self, *path: Union[Node, str]):
        if len(path) == 1:
            self.nodes = path
            self.directions = []
        elif not len(path) % 2 and len(path) > 0:
            raise RuntimeError(f"Path expects input as [Node, <-, Node, <-, Node]")
        else:
            self.nodes, self.directions = path[::2], path[1::2]
        self.path = path

    def reversed(self):
        return Path(*['<--' if i == '-->' else '-->' if i == '<--' else i for i in self.path[::-1]])

    def __repr__(self):
        s = ''.join(map(str, self.path))
        return s

    def stringify(self, mentioned_nodes):
        s = ''.join([p if isinstance(p, str) else p.stringify(mentioned_nodes) for p in self.path])
        return s

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.nodes[item]
        else:
            return self.nodes[self.names.index(item)]

    @property
    def names(self):
        return [n.name for n in self.nodes]

    def __len__(self):
        return len(self.nodes)


class Unwind(Copyable):
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def stringify(self, mentioned_nodes):
        if self in mentioned_nodes:
            return self.name
        mentioned_nodes.append(self.name)
        return f"${self.name} as {self.name}"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.name == other.name)


class Generator:
    def __init__(self):
        self.node_counter = defaultdict(int)
        self.property_name_counter = defaultdict(int)

    def node(self, label=None, name=None, **properties):
        if name is None:
            self.node_counter[str(label)] += 1
            name = ''.join([str(label).lower(), str(self.node_counter[str(label)] - 1)])
        return Node(label, name, **properties)

    def data(self, values: List) -> Unwind:
        label = 'user_data'
        name = label + str(self.node_counter[str(label)])
        self.node_counter[str(label)] += 1
        return Unwind(name, values)

    def nodes(self, *labels):
        return [self.node(l) for l in labels]

    def property_list(self, property_name):
        self.property_name_counter[property_name] += 1
        return ''.join([property_name, str(self.property_name_counter[property_name] - 1)])


class Condition(Copyable):
    def __init__(self, a, comparison, b):
        self.a = a
        self.comparison = comparison
        self.b = b

    @property
    def nodes(self):
        if isinstance(self.a, Condition):
            yield from self.a.nodes
        else:
            yield getattr(self.a, 'node', self.a)
        if isinstance(self.b, Condition):
            yield from self.b.nodes
        else:
            yield getattr(self.a, 'node', self.a)


    def stringify(self):
        a = self.a.stringify() if isinstance(self.a, Condition) else getattr(self.a, 'name', quote(str(self.a)))
        b = self.b.stringify() if isinstance(self.b, Condition) else getattr(self.b, 'name', quote(str(self.b)))
        return f"({a} {self.comparison} {b})"

    def __repr__(self):
        return self.stringify()

    def __and__(self, other):
        return Condition(self, 'and', other)

    def __or__(self, other):
        return Condition(self, 'or', other)

    def __eq__(self, other):
        return Condition(self, '==', other)

    def __ne__(self, other):
        return Condition(self, '<>', other)