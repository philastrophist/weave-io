import pytest

from weaveio.basequery.query import Condition, EmptyCondition, Node, NodeProperty


def test_condition_stringify_nodetype_scalar():
    c = Condition(Node('label', 'name').property, '=', 'a')
    assert str(c) == "(name.property = 'a')"


def test_condition_stringify_nodetype_nodetype():
    c = Condition(Node('label', 'name').property, '=', Node('label2', 'name2').property2)
    assert str(c) == "(name.property = name2.property2)"

