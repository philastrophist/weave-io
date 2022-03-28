import os
from copy import deepcopy
from functools import partial
from time import sleep
from typing import Type, Callable

import pytest
from py2neo import Subgraph

from weaveio.graph import Graph
from weaveio.hierarchy import Hierarchy, Multiple, Optional, One2One
from weaveio.schema import diff_hierarchy_schema_node, write_schema, AttemptedSchemaViolation, read_schema, SchemaNode, hierarchy_type_tree, hierarchy_dependency_tree


class A(Hierarchy):
    idname = 'id'
    factors = ['a', 'aa']
    parents = []
    children = []

class B(A):
    idname = 'id'
    factors = []
    parents = [A]
    children = []

class C(Hierarchy):
    idname = 'id'
    factors = ['c']
    parents = [Multiple(B, 1, 2)]
    children = []

class D(Hierarchy):
    idname = 'id'
    factors = ['d']
    parents = []
    children = [One2One(A), Optional(C)]

class E(C):
    idname = 'id'
    factors = ['e']
    parents = [Optional(C)]
    children = [One2One('E')]

class F(E):
    is_template = True

class G(F):
    pass

class H(Hierarchy):
    factors = ['hh']
    indexes = ['hh']

class I(Hierarchy):
    identifier_builder = ['g', 'i']
    factors = ['i', 'ii']
    parents = [G]
    children = [Multiple(H, 1, 1)]

class J(Hierarchy):
    idname = 'id'
    children = [Multiple(H)]


entire_hierarchy = [A, B, C, D, E, F, G, H, I]

def decompose_parents_children_into_names(x):
    if isinstance(x, str):
        return x
    if isinstance(x, type):
        return x.__name__
    if isinstance(x, Multiple):
        if isinstance(x, Optional):
            return Multiple(x.node.__name__, 0, 1, x.constrain, x.relation_idname)
        if x.minnumber == 1 and x.maxnumber == 1 and len(x.constrain) == 0 and x.relation_idname is None:
            return x.node.__name__
        return x.__class__(x.node.__name__, x.minnumber, x.maxnumber, x.constrain, x.idname)


def assert_class_equality(a, b):
    for attr in ['__name__', 'idname']:
        assert getattr(a, attr) == getattr(b, attr), f'{attr} not matched for {a} and {b}'
    for attr in ['parents', 'children', '__bases__']:  # order doesn't matter
        assert set(map(decompose_parents_children_into_names, getattr(a, attr))) == \
               set(map(decompose_parents_children_into_names, getattr(b, attr))), \
            f'{attr} not matched for {a} and {b}'
    for attr in ['factors', 'identifier_builder', 'indexes']:  # order doesn't matter
        if not (getattr(a, attr) is None and getattr(b, attr) is None):
            assert False
        assert set(getattr(a, attr)) == set(getattr(b, attr)), f'{attr} not matched for {a} and {b}'


def nodes_with_parents_or_children_that_meet_condition(condition: Callable, children_or_parents: str = None, *further):
    if children_or_parents is None:
        children_or_parents = ['children', 'parents']
    else:
        children_or_parents = [children_or_parents]
    any_yielded = False
    for node in entire_hierarchy:
        for attr in children_or_parents:
            for target_i, target_node in enumerate(getattr(node, attr)):
                if condition(target_node):
                    _name = getattr(target_node, '__name__', getattr(target_node.node, '__name__', target_node.node))
                    id = f"{node.__name__}.{attr}[{target_i}] == {_name}"
                    yield pytest.param(node, attr, target_node, target_i, *further, id=id)
                    any_yielded = True
    if not any_yielded:
        raise RuntimeError(f"No node in the hierarchy can be used to run this test with condition: {further}")


def subnode_meets_condition(subnode, *conditions):
    if isinstance(subnode, Multiple):
        for condition in conditions:
            if not eval(f'subnode' + condition):
                return False
        return True
    return False

has_maxnumber_above_1 = lambda x: subnode_meets_condition(x, '.maxnumber is not None', '.maxnumber > 1')
has_maxnumber_1 = lambda x: subnode_meets_condition(x, '.maxnumber is not None', '.maxnumber == 1')
has_maxnumber_None = lambda x: subnode_meets_condition(x, '.maxnumber is None')
has_maxnumber_0 = lambda x: subnode_meets_condition(x, '.maxnumber is not None', '.maxnumber == 0')
has_minnumber_0 = lambda x: subnode_meets_condition(x, '.minnumber is not None', '.minnumber == 0')
has_minnumber_1 = lambda x: subnode_meets_condition(x, '.minnumber is not None', '.minnumber == 1')

def copy_class(X, replace_base=None) -> Type[Hierarchy]:
    """
    copies a class and changes it's bases to 'replace_base' if the names are the same.
    This is done recursively
    """
    if not issubclass(X, Hierarchy):
        return X
    if replace_base is not None:
        if X.__name__ == replace_base.__name__:
            return replace_base
    new_bases = tuple(copy_class(b, replace_base) for b in X.__bases__)
    if new_bases == X.__bases__:
        return X
    return type(X.__name__, new_bases, deepcopy(dict(X.__dict__)))


def replace_class_in_type_hierarchy(hierarchies, replace):
    """
    Replaces all mentions of a hierarchy with the same name as `replace` with `replace` itself
    """
    hier_list = [copy_class(hier, replace) for hier in hierarchy_type_tree(hierarchies)]
    for list_type in ['parents', 'children']:
        for hier in hierarchy_dependency_tree(hier_list):
            for i, h in enumerate(getattr(hier, list_type)):
                if isinstance(h, Multiple):
                    h.node = copy_class(h.node, replace)
                else:
                    getattr(hier, list_type)[i] = copy_class(h, replace)
    return hier_list


@pytest.fixture(scope='function')
def graph():
    graph = Graph(name='playground', host='127.0.0.1', port=7687,
                  user=os.environ['WEAVEIO_USER'], password=os.environ['WEAVEIO_PASS'], write=True)
    if len(graph.execute('MATCH (n) return n').to_table()):
        raise ValueError(f"Cannot start doing tests on a non-empty database")
    yield graph
    graph.execute('MATCH (n) detach delete n')


def test_push_dryrun_makes_no_changes(graph):
    write_schema(graph, entire_hierarchy, dryrun=True)
    assert len(graph.execute('MATCH (n) return n').to_table()) == 0


def test_push_one_to_empty(graph):
    write_schema(graph, [A])
    assert len(graph.execute('MATCH (n) return n').to_table()) == 1


def test_same_node_no_change(graph):
    """Pushing the same hierarchy makes no changes"""
    write_schema(graph, entire_hierarchy)
    subgraph1 = graph.execute('MATCH (n)-[r]-(m) return *').to_subgraph()  # type: Subgraph
    write_schema(graph, entire_hierarchy)
    subgraph2 = graph.execute('MATCH (n)-[r]-(m) return *').to_subgraph()  # type: Subgraph
    assert subgraph1 == subgraph2


@pytest.mark.parametrize('attr', ['idname', 'singular_name', 'plural_name'])
@pytest.mark.parametrize('node', entire_hierarchy)
def test_changing_string_attributes_is_not_allowed(graph, attr, node):
    write_schema(graph, entire_hierarchy)
    new_node = copy_class(node)
    setattr(new_node, attr, 'changed')
    with pytest.raises(AttemptedSchemaViolation, match=f'proposed {attr} .+ is different from the original'):
        write_schema(graph, replace_class_in_type_hierarchy(entire_hierarchy, new_node))


def test_changing_relidname_is_not_allowed(graph):
    assert False


@pytest.mark.parametrize('attr', ['factors', 'parents', 'children', 'identifier_builder', 'indexes'])
@pytest.mark.parametrize('node', entire_hierarchy)
def test_shortening_attributes_is_not_allowed(graph, attr, node):
    if len(getattr(node, attr) or []) == 0:
        return
    write_schema(graph, entire_hierarchy)
    new_node = copy_class(node)
    setattr(new_node, attr, getattr(new_node, attr)[:-1])  # shorten it
    with pytest.raises(AttemptedSchemaViolation, match=f'{attr}'):
        write_schema(graph, replace_class_in_type_hierarchy(entire_hierarchy, new_node))


def test_lengthening_factors_is_allowed(graph):
    write_schema(graph, entire_hierarchy)
    newI = copy_class(I)
    newI.factors += ['new']
    write_schema(graph, entire_hierarchy[:-1] + [newI])
    assert graph.execute('MATCH (n:I {name:"I"}) return n.factors').evaluate() == newI.factors


@pytest.mark.parametrize('node', entire_hierarchy)
@pytest.mark.parametrize('attr', ['children', 'parents'])
def test_adding_nonoptional_parents_is_not_allowed(graph, node, attr):
    write_schema(graph, entire_hierarchy)
    new_node = copy_class(node)
    class NewClass(Hierarchy):
        idname = 'id'
    setattr(new_node, attr, getattr(new_node, attr) + [NewClass])
    with pytest.raises(AttemptedSchemaViolation, match=f'{attr}'):
        write_schema(graph, replace_class_in_type_hierarchy(entire_hierarchy, new_node)+[NewClass])


@pytest.mark.parametrize('node', entire_hierarchy)
@pytest.mark.parametrize('attr', ['children', 'parents'])
@pytest.mark.parametrize('use_Optional', [True, False])
def test_adding_optional_parents_is_allowed(graph, node, attr, use_Optional):
    write_schema(graph, entire_hierarchy)
    new_node = copy_class(node)
    class NewClass(Hierarchy):
        idname = 'id'
    if use_Optional:
        new = Optional(NewClass)
    else:
        new = Multiple(NewClass, 0)
    setattr(new_node, attr, getattr(new_node, attr) + [new])
    with pytest.raises(AttemptedSchemaViolation, match=f'{attr}'):
        write_schema(graph, replace_class_in_type_hierarchy(entire_hierarchy, new_node) + [NewClass])


params = [
    list(nodes_with_parents_or_children_that_meet_condition(has_maxnumber_above_1, None, 'max >1 -> None', True)),
    list(nodes_with_parents_or_children_that_meet_condition(has_maxnumber_1, None, 'max 1 -> None', False)),
    list(nodes_with_parents_or_children_that_meet_condition(has_maxnumber_None, None, 'max None -> 2', False)),
    list(nodes_with_parents_or_children_that_meet_condition(has_maxnumber_1, None, 'max 1 -> 2', False)),
    list(nodes_with_parents_or_children_that_meet_condition(has_minnumber_0, None, 'min 0 -> 1', False)),
    list(nodes_with_parents_or_children_that_meet_condition(has_minnumber_1, None, 'min 1 -> 2', False)),
    list(nodes_with_parents_or_children_that_meet_condition(has_minnumber_1, None, 'min 1 -> 0', True))
]
@pytest.mark.parametrize('node,attr,target_node,target_i,condition,allowed', [p for plist in params for p in plist])
def test_changing_bound_multiples(graph, node, attr, target_node, target_i, condition, allowed):
    write_schema(graph, entire_hierarchy)
    new_node = copy_class(node)
    new_target = deepcopy(target_node)
    new_target.maxnumber = None
    getattr(new_node, attr)[target_i] = new_target
    new_hierarchy = replace_class_in_type_hierarchy(entire_hierarchy, new_node)
    if not allowed:
        with pytest.raises(AttemptedSchemaViolation):
            write_schema(graph, new_hierarchy)
    else:
        write_schema(graph, new_hierarchy)
        hiers = {v.__name__: v for v in read_schema(graph)}
        retrieved_attrs = getattr(hiers[target_node.__name__], attr)[target_i]
        assert any(i for i in retrieved_attrs if i == new_target)  # make there is one match


def test_changing_one2one_to_another_is_not_allowed(graph):
    assert False


def test_changing_singular_or_multiple_to_one2one_is_not_allowed(graph):
    assert False


def test_changing_to_template_is_not_allowed(graph):
    assert False


def test_changing_from_template_is_not_allowed(graph):
    assert False



def test_push_entire_hierarchy(graph):
    write_schema(graph, entire_hierarchy)


def test_pull_hierarchy_matches_creator(graph):
    write_schema(graph, entire_hierarchy)
    read_hierarchy = {v.__name__: v for v in read_schema(graph)}
    original = {v.__name__: v for v in entire_hierarchy}
    assert set(read_hierarchy.keys()) == set(original.keys())
    for k in original.keys():
        assert_class_equality(read_hierarchy[k], original[k])


def test_template_hierarchies_dont_have_deps_written(graph):
    write_schema(graph, entire_hierarchy)
    nodes = graph.execute('match (n: SchemaNode {name: "F"}) '
                          'optional match (n)-[r]-(m) where not r:IS_TYPE_OF '
                          'return n, count(r)').to_table()
    assert len(nodes) == 1
    assert nodes[0][1] == 0


def test_template_hierarchies_are_recomposed_at_read_from_other_hierarchies(graph):
    assert False


def test_writing_the_read_graph_back(graph):
    write_schema(graph, entire_hierarchy)
    graph.execute('MATCH (n)-[r]-(m) return *').to_subgraph()  # type: Subgraph
    read_hierarchy1 = {v.__name__: v for v in read_schema(graph)}
    write_schema(graph, read_hierarchy1.values())
    read_hierarchy2 = {v.__name__: v for v in read_schema(graph)}
    for k, v in read_hierarchy2.items():
        assert_class_equality(read_hierarchy1[k], v)

