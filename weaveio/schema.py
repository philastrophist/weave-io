import logging
from collections import deque
from typing import Type, List, Union, Tuple, Dict, Set
from py2neo import Relationship, Node, Subgraph
from tqdm import tqdm

from weaveio.graph import Graph
from weaveio.hierarchy import Graphable, Hierarchy, Multiple, One2One


class AttemptedSchemaViolation(Exception):
    pass


def get_all_class_bases(cls: Type[Graphable]) -> List[Type[Graphable]]:
    new = []
    for b in cls.__bases__:
        if b is Graphable or not issubclass(b, Graphable):
            continue
        new.append(b)
        new += get_all_class_bases(b)
    return new


def get_labels_of_schema_hierarchy(hierarchy: Union[Type[Hierarchy], Multiple], as_set=False):
    if isinstance(hierarchy, Multiple):
        hierarchy = hierarchy.node
    bases = get_all_class_bases(hierarchy)
    labels = [i.__name__ for i in bases]
    labels.append(hierarchy.__name__)
    labels.append('SchemaNode')
    if as_set:
        return frozenset(labels)
    return labels


def push_py2neo_schema_subgraph_cypher(subgraph: Subgraph) -> Tuple[str, Dict]:
    """
    Each node label is unique, so we just merge by label and then update properties
    for relationships we merge with empty relation and then update the properties
    """
    cypher = []
    params = {}
    for i, node in enumerate(subgraph.nodes):
        cypher.append(f"MERGE (n{i}{node.labels}) WITH * WHERE size(labels(n{i})) = {len(node.labels)}")
        for k, v in node.items():
            params[f'n{i}{k}'] = v
            cypher.append(f"SET n{i}.{k} = $n{i}{k}")
    for i, rel in enumerate(subgraph.relationships):
        a = list(subgraph.nodes).index(rel.start_node)
        b = list(subgraph.nodes).index(rel.end_node)
        cypher.append(f"MERGE (n{a})-[r{i}:{list(rel.types())[0]}]->(n{b})")
        for k, v in rel.items():
            params[f'r{i}{k}'] = v
            cypher.append(f"SET r{i}.{k} = $r{i}{k}")
    return "\n".join(cypher), params


def diff_hierarchy_schema_node(graph: Graph, hierarchy: Type[Hierarchy]):
    """
    Given a hierarchy, return py2neo subgraph that can be pushed to update the schema.
    If the new hierarchy is not backwards compatible, an exception will be raised

    A hierarchy can only be merged into the schema if existing data will still match the schema.
    i.e. if all below are true:
        has the same idname
        has the same factors or more
        has the same parents and children
            (additional parents and children can only be specified if they are optional)

    In the schema:
        (a)-[:is_parent_of]->(b) indicates that (b) requires a parent (a) at instantiation time
        (a)-[:has_child]->(b) indicates that (a) requires a child (b) at instantiation time
    """
    # structure is [{labels}, is_optional, (minn, maxn), rel_idname, is_one2one]
    actual_parents = [(get_labels_of_schema_hierarchy(p, True), False, (1, 1), None, False) if not isinstance(p, Multiple)
                      else (get_labels_of_schema_hierarchy(p.node, True), p.minnumber == 0,
                            (p.minnumber, p.maxnumber), p.relation_idname, isinstance(p, One2One)) for p in hierarchy.parents]
    actual_children = [(get_labels_of_schema_hierarchy(p, True), False, (1, 1), None, False) if not isinstance(p, Multiple)
                       else (get_labels_of_schema_hierarchy(p.node, True), p.minnumber == 0,
                             (p.minnumber, p.maxnumber), p.relation_idname, isinstance(p, One2One)) for p in hierarchy.children]
    cypher = f"""
    MATCH (n:{hierarchy.__name__}:SchemaNode)
    OPTIONAL MATCH (n)-[child_rel:HAS_CHILD]->(child:SchemaNode)
    with n, child, collect(child_rel) as child_rels
    OPTIONAL MATCH (parent:SchemaNode)-[parent_rel:IS_PARENT_OF]->(n)
    with n, child, child_rels, parent, collect(parent_rel) as parent_rels
    RETURN n, parent, child, parent_rels, child_rels
    """
    labels = get_labels_of_schema_hierarchy(hierarchy)
    results = graph.execute(cypher).to_table()
    if len(results) == 0:  # node is completely new
        parents = []
        children = []
        rels = []
        found_node = Node(*labels,
                          factors=hierarchy.factors, idname=hierarchy.idname,
                          singular_name=hierarchy.singular_name, plural_name=hierarchy.plural_name)
        for struct in actual_parents:
            props = dict(optional=struct[1], minnumber=struct[2][0], maxnumber=struct[2][1],
                         idname=struct[3])
            parent = Node(*struct[0])  # extant parent, specify labels so it matches not creates
            rels.append(Relationship(parent, 'IS_PARENT_OF', found_node, **props))  # new rel, should not exists, create
            if struct[4]:  # is_one2one, so add a reflected relation as well
                rels.append(Relationship(found_node, 'IS_PARENT_OF', parent, **props))
            parents.append(parent)
        for struct in actual_children:
            props = dict(optional=struct[1], minnumber=struct[2][0], maxnumber=struct[2][1],
                         idname=struct[3])
            child = Node(*struct[0])  # extant child, specify labels so it matches not creates
            rels.append(Relationship(found_node, 'HAS_CHILD', child, **props))  # new rel, should not exists, create
            if struct[4]:  # is_one2one, so add a reflected relation as well
                rels.append(Relationship(child, 'HAS_CHILD', found_node, **props))
            children.append(child)
    else:
        found_node = results[0][0]
        actual_parents = set(actual_parents)
        actual_children = set(actual_children)

        # gather extant info
        found_factors = set(found_node.get('factors', []))
        found_parents = {(frozenset(r[1].labels), rel['optional'], (rel['minnumber'], rel['maxnumber']), rel['idname']) for r in results if r[1] is not None for rel in r[3]}
        found_children = {(frozenset(r[2].labels), rel['optional'], (rel['minnumber'], rel['maxnumber']), rel['idname']) for r in results if r[2] is not None for rel in r[4]}

        # see if hierarchy is different in anyway
        different_idname = found_node.get('idname') != hierarchy.idname
        different_singular_name = found_node.get('singular_name') != hierarchy.singular_name
        different_plural_name = found_node.get('plural_name') != hierarchy.plural_name
        missing_factors = set(hierarchy.factors) - found_factors
        different_labels = set(labels).symmetric_difference(found_node.labels)

        # parents are different?
        missing_parents = set(actual_parents - found_parents)
        new_parents = {p for p in found_parents - actual_parents if not p[1]}  # allowed if the new ones are optional

        # children are different?
        missing_children = found_children - actual_children  # missing from new definition
        new_children = {p for p in actual_children - found_children if not p[1]}  # allowed if the new ones are optional

        if different_idname or missing_factors or different_labels or missing_parents or \
                new_parents or missing_children or new_children or different_singular_name or different_plural_name:
            msg = f'Cannot add new hierarchy {hierarchy} because the {hierarchy.__name__} already exists' \
                  f' and the proposed definition of {hierarchy} is not backwards compatible. ' \
                  f'The differences are listed below:\n'
            if different_idname:
                msg += f'- proposed idname {hierarchy.idname} is different from the original {found_node.get("idname")}\n'
            if different_singular_name:
                msg += f'- proposed singular_name {hierarchy.singular_name} is different from the original {found_node.get("singular_name")}\n'
            if different_plural_name:
                msg += f'- proposed plural_name {hierarchy.plural_name} is different from the original {found_node.get("plural_name")}\n'
            if different_labels:
                msg += f'- proposed inherited types {different_labels} are different from {labels}\n'
            if missing_factors:
                msg += f'- factors {missing_factors} are missing from proposed definition\n'
            if new_parents:
                msg += f'- new parents with labels {[set(p[0]) - {"SchemaNode"} for p in new_parents]} are not optional (and therefore arent backwards compatible)\n'
            if new_children:
                msg += f'- new children with labels {[set(p[0]) - {"SchemaNode"} for p in new_children]} are not optional (and therefore arent backwards compatible)\n'
            if missing_parents:
                msg += f'- parents with labels {[set(p[0]) - {"SchemaNode"} for p in missing_parents]} are missing from the new definition\n'
            if missing_children:
                msg += f'- children with labels {[set(p[0]) - {"SchemaNode"} for p in missing_children]} are missing from the new definition\n'
            msg += f'any flagged children or parents may have inconsistent min/max number'
            raise AttemptedSchemaViolation(msg)

        nodes, rels = [], []
        if set(hierarchy.factors) != found_factors:
            found_node['factors'] = hierarchy.factors  # update
            nodes.append(found_node)
        if found_children.symmetric_difference(actual_children):
            children = [(Node(*labels), dict(optional=optional, minnumber=minn, maxnumber=maxn, idname=idname))
                        for labels, optional, (minn, maxn), idname in actual_children]
            nodes += children
        else:
            children = []
        if found_parents.symmetric_difference(actual_parents):
            parents = [(Node(*labels), dict(optional=optional, minnumber=minn, maxnumber=maxn, idname=idname))
                       for labels, optional, (minn, maxn), idname in actual_parents]
            nodes += parents
        else:
            parents = []
        # repeat the relationship if it is multiple
        rels = [Relationship(found_node, 'HAS_CHILD', c, **props) for c, props in children for _ in range((props['maxnumber'] > 1 or props['maxnumber'] is None) + 1)] \
               + [Relationship(p, 'IS_PARENT_OF', found_node, **props) for p, props in parents for _ in range((props['maxnumber'] > 1 or props['maxnumber'] is None) + 1)]
    return Subgraph(parents + children + [found_node], rels)


def write_schema(graph, hierarchies, dryrun=False):
    """
    writes to the neo4j schema graph for use in optimising queries
    this should always be done before writing data
    """
    # sort available hierarchies meaning non-dependents first
    hierarchies = deque(sorted(list(hierarchies),
                               key=lambda x: len(x.children) + len(x.parents) + len(x.products)))
    done = []
    executions = []
    nmisses = 0
    while hierarchies:
        if nmisses == len(hierarchies):
            raise AttemptedSchemaViolation(f"Dependency resolution impossible, proposed schema elements may have cyclic dependencies:"
                                           f"{list(hierarchies)}")
        hier = hierarchies.popleft()  # type: Type[Hierarchy]
        hier.instantate_nodes()
        if hier.is_template:
            continue  # don't need to add templates, they just provide labels
        dependencies = [d.node if isinstance(d, Multiple) else d for d in hier.parents + hier.children]
        dependencies = filter(lambda d: d != hier, dependencies)  # allow self-references
        if not all(d in done for d in dependencies):
            logging.info(f"{hier} put at the back because it requires dependencies which have not been written yet")
            hierarchies.append(hier)  # do it after the dependencies are done
            nmisses += 1
            continue
        executions.append(push_py2neo_schema_subgraph_cypher(diff_hierarchy_schema_node(graph, hier)))
        nmisses = 0  # reset counter
        done.append(hier)
    if not dryrun:
        for cypher, params in tqdm(executions, desc='schema updates'):
            graph.execute(cypher, **params)
    return True


def read_schema(graph) -> Set[Type[Hierarchy]]:
    nondependents = graph.execute('MATCH (n) WHERE not exists((n)<-[:IS_PARENT_OF]-()) AND not exists((n)-[:HAS_CHILD]->()) RETURN n').to_subgraph()

