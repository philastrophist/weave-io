from dataclasses import dataclass
from functools import wraps
from string import printable, ascii_letters
from typing import Dict, List, Any

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.stateful import initialize, RuleBasedStateMachine, rule, invariant, precondition
from hypothesis.strategies import builds, data

from weaveio.graph import Graph
from weaveio.writequery import CypherQuery, merge_node
from weaveio.writequery.tests.conftest import write_database

baseproperty_strategy = st.none() | st.integers() | st.text() | st.floats(allow_nan=True, allow_infinity=True)
neo4jproperty_strategy = baseproperty_strategy | st.lists(baseproperty_strategy)
key_strategy = st.text(min_size=1, alphabet=printable)
label_strategy = st.text(ascii_letters, min_size=1)
labels_strategy = st.lists(label_strategy, min_size=1, unique=True)
properties_strategy = st.dictionaries(key_strategy, neo4jproperty_strategy)
identproperties_strategy = st.dictionaries(key_strategy, baseproperty_strategy)


@dataclass
class Node:
    labels: List[str]
    properties: Dict[str, Any]
    identproperties: Dict[str, Any]


@dataclass
class Relation:
    reltype: str
    properties: Dict[str, Any]
    identproperties: Dict[str, Any]


node_strategy = builds(Node, labels=labels_strategy, properties=properties_strategy, identproperties=identproperties_strategy)
rel_strategy = builds(Relation, reltype=label_strategy, properties=properties_strategy, identproperties=identproperties_strategy)


def labels_from_labels(labels: List, how, sampler):
    if how == False:
        return labels
    elif how == 'entire':  # must be entirely different
        return sampler.draw(labels_strategy.filter(lambda xs: not any(x in labels for x in xs)))
    elif how == 'crop':  # return the same list up until an index
        if len(labels) == 1:
            return labels  # odd, but we need to ignore it here
        index = sampler.draw(st.integers(min_value=0, max_value=len(labels) - 2))
        return labels[:index]
    elif how == 'extend':  # just add more labels
        return labels + sampler.draw(labels_strategy.filter(lambda xs: not any(x in labels for x in xs)
                                                                       and len(xs) > 1))
    elif how == 'crop&extend':
        if len(labels) == 1:
            return labels
        return labels[:-1] + [sampler.draw(label_strategy.filter(lambda x: x not in labels[:-1]))]
    else:
        raise ValueError(f"Method to change labels {how} is unknown")


def properties_from_properties(properties: Dict, how, sampler, properties_strategy_type, property_strategy_type):
    new = properties.copy()
    keys = list(properties.keys())
    if how == False:
        return properties
    elif how == 'entirekeys':  # must be entirely different
        return sampler.draw(properties_strategy_type.filter(lambda d: not any(k in properties for k in d)))
    elif how == 'crop':  # return the same list up until an index
        if len(properties) == 0:
            return {}  # this situation is handled by 'extend'
        indexes = sampler.draw(st.lists(st.integers(min_value=0, max_value=len(keys) - 1), unique=True, min_size=1, max_size=len(keys)))
        for i in indexes:
            del new[keys[i]]
        return new
    elif how == 'addkeys':  # just add more labels
        strat = properties_strategy_type.filter(lambda x: not any(k in properties for k in x) and len(x))
        additions = sampler.draw(strat)
        new.update(additions)
        return new
    elif how == 'overwritekeys':
        if len(properties) == 0:
            return {}  # this situation is handled by 'extend'
        n = sampler.draw(st.integers(min_value=1, max_value=len(properties)))
        whichkeys = sampler.draw(st.lists(st.integers(min_value=0, max_value=len(keys)-1), min_size=n, max_size=n, unique=True))
        newkeys = [keys[i] for i in whichkeys]
        oldvalues = [properties[k] for k in newkeys]
        newvalues = sampler.draw(st.lists(property_strategy_type.filter(lambda x: x not in oldvalues),
                                          min_size=n, max_size=n)
                                 )
        for k, v in zip(newkeys, newvalues):
            new[k] = v
        return new
    else:
        raise ValueError(f"Method to change properties {how} is unknown")


def create_node_from_node(node, sampler, different_labels, different_properties, different_identproperties):
    new_properties = properties_from_properties(node.properties, different_properties, sampler,
                                                properties_strategy, neo4jproperty_strategy)
    new_identproperties = properties_from_properties(node.identproperties, different_identproperties,
                                                     sampler, identproperties_strategy, baseproperty_strategy)
    new_labels = labels_from_labels(node.labels, different_labels, sampler)
    return Node(new_labels, identproperties=new_identproperties, properties=new_properties)


# @given(labels=labels_strategy.filter(lambda x: len(x) > 1), sampler=data())
# @mark.parametrize('different_labels', [False, 'entire', 'crop', 'extend', 'crop&extend'])
# def test_labels_from_labels(labels, different_labels, sampler):
#     newlabels = labels_from_labels(labels, different_labels, sampler)
#     note(f"newlabels = {newlabels}")
#     note(f"oldlabels = {labels}")
#     assert len(set(newlabels)) == len(newlabels)
#     if different_labels == False:
#         assert newlabels == labels
#     elif different_labels == 'entire':
#         assert not any(l in labels for l in newlabels)
#     elif different_labels == 'crop':
#         assert newlabels == labels[:len(newlabels)]
#     elif different_labels == 'extend':
#         assert newlabels[:len(labels)] == labels
#         assert len(newlabels) > len(labels)
#     elif different_labels == 'crop&extend':
#         assert newlabels[0] == labels[0]  # at least one is the same
#
#
# def is_null(x):
#     try:
#         return np.all(np.isnan(x))
#     except TypeError:
#         pass
#     return x is None
#
#
# def is_equal_or_null(x, y):
#     return x == y or (is_null(x) and is_null(y) and (type(x) == type(y)))
#
#
# @given(properties=properties_strategy, sampler=data())
# @mark.parametrize('different_properties', [False, 'entirekeys', 'crop', 'addkeys', 'overwritekeys'])
# def test_properties_from_properties(properties, different_properties, sampler):
#     newprops = properties_from_properties(properties, different_properties, sampler,
#                                           properties_strategy, neo4jproperty_strategy)
#     note(f"newprops = {newprops}")
#     note(f"oldprops = {properties}")
#     if different_properties == False:
#         assert properties == newprops
#     elif different_properties == 'entirekeys':
#         assert not any(k in properties for k in newprops)
#     elif different_properties == 'crop':
#         assert len(newprops) < len(properties) or len(properties) == 0
#         assert all(is_equal_or_null(properties[k], v) for k, v in newprops.items())
#     elif different_properties == 'addkeys':
#         assert all(is_equal_or_null(newprops[k], v) for k, v in properties.items())
#         assert len(newprops) > len(properties)
#     elif different_properties == 'overwritekeys':
#         assert len(newprops) == len(properties)
#         if len(properties) > 0:
#             # at least one different
#             assert any(not is_equal_or_null(newprops[k], v) for k, v in properties.items())
#
#
# @pytest.yield_fixture(scope="module")
# def database():
#     print('create database')
#     yield
#     print('destroy database')
#
#
#
# @settings(max_examples=100)
# @given(node1=node_strategy, sampler=data())
# @pytest.mark.parametrize('different_labels', [False, 'entire', 'crop', 'extend', 'crop&extend'])
# @pytest.mark.parametrize('different_properties', [False, 'entirekeys', 'crop', 'addkeys', 'overwritekeys'])
# @pytest.mark.parametrize('different_identproperties', [False, 'entirekeys', 'crop', 'addkeys', 'overwritekeys'])
# @pytest.mark.parametrize('collision_manager', ['overwrite', 'ignore', 'track&flag'])
# def test_merge(node1, sampler, database, different_labels, different_properties, different_identproperties, collision_manager):
#     print('clear')
#     node2 = create_node_from_node(node1, sampler, different_labels, different_properties, different_identproperties)
#     print('merge')
#     if different_labels == False and different_identproperties == False:
#         labels = node1.labels
#         properties = node1.properties.copy()
#         properties.update(node1.identproperties)
#         rows = database.execute(f'MATCH (n) WHERE NOT n:Collision RETURN n').to_table()
#         assert len(rows) == 1
#         node = rows[0][0]
#         assert node['labels'] == labels
#         assert node['properties'] == properties
#         if collision_manager == 'track&flag':
#             rows = database.execute(f'MATCH (n:Collision) RETURN n').to_table()

def invariant_test(method):
    return invariant()(precondition(lambda self: self.test_begins)(method))


def factory(different_labels, different_identproperties, different_properties, collision_manager):
    @settings(max_examples=100, stateful_step_count=1)
    class Merger(RuleBasedStateMachine):
        def __init__(self):
            self.different_labels = different_labels
            self.different_identproperties = different_identproperties
            self.different_properties = different_properties
            self.collision_manager = collision_manager
            self.test_begins = False
            super(Merger, self).__init__()

        @rule(node1=node_strategy, sampler=data())
        def setup(self, node1, sampler):
            node2 = create_node_from_node(node1, sampler, self.different_labels, self.different_properties,
                                          self.different_identproperties)
            # with CypherQuery() as query:
            #     a = merge_node(node1.labels, node1.identproperties, node1.properties, collision_manager=self.collision_manager)
            #     b = merge_node(node2.labels, node2.identproperties, node2.properties, collision_manager=self.collision_manager)
            # cypher, params = query.render_query()
            # self.graph.execute(cypher, **params)
            self.test_begins = True

        @invariant_test
        def test_it(self):
            pass

        @invariant_test
        def test_it2(self):
            assert False

    return Merger.TestCase


for different_labels in [False, 'entire', 'crop', 'extend', 'crop&extend']:
    for different_properties in  [False, 'entirekeys', 'crop', 'addkeys', 'overwritekeys']:
        for different_identproperties in [False, 'entirekeys', 'crop', 'addkeys', 'overwritekeys']:
            for collision_manager in ['track&flag', 'ignore', 'overwrite']:
                suffix = f'{different_labels},{different_identproperties,}{different_properties},{collision_manager}'
                pytest.mark.parametrize()
                test_case = factory(different_labels, different_identproperties, different_properties, collision_manager)
                globals()[f'TestMerger[{suffix}]'] = test_case
