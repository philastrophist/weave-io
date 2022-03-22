import os

import pytest
from py2neo import Subgraph

from weaveio.graph import Graph
from weaveio.hierarchy import Hierarchy, Multiple, Optional, One2One
from weaveio.schema import diff_hierarchy_schema_node, write_schema, AttemptedSchemaViolation, read_schema


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
    parents = [Multiple(B)]
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
    idname = 'id'

class I(Hierarchy):
    idname = 'id'
    parents = [G]
    children = [H]

entire_hierarchy = [A, B, C, D, E, F, G, H, I]


@pytest.fixture(scope='function')
def graph():
    graph = Graph(name='playground', host='127.0.0.1', port=7687,
                 user=os.environ['WEAVEIO_USER'], password=os.environ['WEAVEIO_PASS'], write=True)
    if len(graph.execute('MATCH (n) return n').to_table()):
        raise ValueError(f"Cannot start doing tests on a non-empty database")
    yield graph
    graph.execute('MATCH (n) detach delete n')


def test_push_dryrun_makes_no_changes(graph):
    write_schema(graph, [A], dryrun=True)
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


def test_changing_idname_is_not_allowed(graph):
    write_schema(graph, entire_hierarchy[:1])
    class A(Hierarchy):
        idname = 'id_changed'
        factors = ['a', 'aa']
        parents = []
        children = []
    with pytest.raises(AttemptedSchemaViolation, match=f'proposed idname {A.idname} is different from the original'):
        write_schema(graph, [A])


def test_changing_singular_name_is_not_allowed(graph):
    pass


def test_changing_plural_name_is_not_allowed(graph):
    pass


def test_shortening_factors_is_not_allowed(graph):
    pass


def test_lengthening_factors_is_allowed(graph):
    pass


def test_removing_parents_is_not_allowed(graph):
    pass


def test_removing_children_is_not_allowed(graph):
    pass


def test_adding_optional_parents_is_allowed(graph):
    pass


def test_adding_optional_children_is_allowed(graph):
    pass


def test_adding_multiple_parents_with_min0_is_allowed(graph):
    pass


def test_adding_multiple_children_with_min0_is_allowed(graph):
    pass


def test_push_entire_hierarchy(graph):
    write_schema(graph, entire_hierarchy)


def test_pull_hierarchy_matches_creator(graph):
    write_schema(graph, entire_hierarchy)
    read_hierarchy = read_schema(graph)
    assert read_hierarchy == entire_hierarchy  # including template ones


def test_template_hierarchies_dont_have_deps_written(graph):
    write_schema(graph, entire_hierarchy)
    nodes = graph.execute('match (n: SchemaNode {name: "F"}) '
                          'optional match (n)-[r]-(m) where not r:IS_TYPE_OF '
                          'return n, count(r)').to_table()
    assert len(nodes) == 1
    assert nodes[0][1] == 0


def test_template_hierarchies_are_recomposed_at_read_from_other_hierarchies(graph):
    pass


