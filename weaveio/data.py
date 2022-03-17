import logging
import re
import time
from collections import deque
from functools import reduce
from itertools import product
from operator import and_
from pathlib import Path
from typing import Union, List, Tuple, Type, Dict, Set
from warnings import warn

import networkx as nx
import pandas as pd
import py2neo
from py2neo import ClientError, DatabaseError, Relationship, Node, Subgraph
from tqdm import tqdm

from weaveio.basequery.actions import TraversalPath
from weaveio.basequery.common import AmbiguousPathError
from weaveio.basequery.handler import Handler, defaultdict
from weaveio.basequery.tree import BranchHandler
from weaveio.file import File, HDU
from weaveio.graph import Graph
from weaveio.hierarchy import Multiple, Hierarchy, Graphable, One2One, collect
from weaveio.utilities import make_plural, make_singular
from weaveio.writequery import Unwind, merge_node, match_node, match_pattern_node, match_relationship

CONSTRAINT_FAILURE = re.compile(r"already exists with label `(?P<label>[^`]+)` and property "
                                r"`(?P<idname>[^`]+)` = (?P<idvalue>[^`]+)$", flags=re.IGNORECASE)

def process_neo4j_error(data: 'Data', file: File, msg):
    matches = CONSTRAINT_FAILURE.findall(msg)
    if not len(matches):
        return  # cannot help
    label, idname, idvalue = matches[0]
    # get the node properties that already exist
    extant = data.graph.neograph.evaluate(f'MATCH (n:{label} {{{idname}: {idvalue}}}) RETURN properties(n)')
    fname = data.graph.neograph.evaluate(f'MATCH (n:{label} {{{idname}: {idvalue}}})-[*]->(f:File) return f.fname limit 1')
    idvalue = idvalue.strip("'").strip('"')
    file.data = data
    obj = [i for i in data.hierarchies if i.__name__ == label][0]
    instance_list = getattr(file, obj.plural_name)
    new = {}
    if not isinstance(instance_list, (list, tuple)):  # has an unwind table object
        new_idvalue = instance_list.identifier
        if isinstance(new_idvalue, Unwind):
            # find the index in the table and get the properties
            filt = (new_idvalue.data == idvalue).iloc[:, 0]
            for k in extant.keys():
                if k == 'id':
                    k = idname
                value = getattr(instance_list, k, None)
                if isinstance(value, Unwind):
                    table = value.data.where(pd.notnull(value.data), 'NaN')
                    new[k] = str(table[k][filt].values[0])
                else:
                    new[k] = str(value)
        else:
            # if the identifier of this object is not looping through a table, we cant proceed
            return
    else:  # is a list of non-table things
        found = [i for i in instance_list if i.identifier == idvalue][0]
        for k in extant.keys():
            value = getattr(found, k, None)
            new[k] = value
    comparison = pd.concat([pd.Series(extant, name='extant'), pd.Series(new, name='to_add')], axis=1)
    filt = comparison.extant != comparison.to_add
    filt &= ~comparison.isnull().all(axis=1)
    where_different = comparison[filt]
    logging.exception(f"The node (:{label} {{{idname}: {idvalue}}}) tried to be created twice with different properties.")
    logging.exception(f"{where_different}")
    logging.exception(f"filenames: {fname}, {file.fname}")


def get_all_subclasses(cls: Type[Graphable]) -> List[Type[Graphable]]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


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


def find_children_of(parent):
    hierarchies = get_all_subclasses(Hierarchy)
    children = set()
    for h in hierarchies:
        if len(h.parents):
            if any(p is parent if isinstance(p, type) else p.node is parent for p in h.parents):
                children.add(h)
    return children


class IndirectAccessError(Exception):
    pass


class MultiplicityError(Exception):
    pass


def shared_base_class(*classes):
    if len(classes):
        all_classes = list(reduce(and_, [set(get_all_class_bases(cls)+[cls]) for cls in classes]))
        all_classes.sort(key=lambda x: len(get_all_class_bases(x)), reverse=True)
        if all_classes:
            return all_classes[0]
    return Hierarchy


def is_multiple_edge(graph, x, y):
    return not graph.edges[(x, y)]['multiplicity']


class AttemptedSchemaViolation(Exception):
    pass


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



class Data:
    filetypes = []

    @staticmethod
    def hierarchies_from_filetype(*filetype: Type[File]) -> Set[Type[Hierarchy]]:
        hierarchies = set()
        all_hierarchies = set(get_all_subclasses(Hierarchy))
        todo = set(filetype)
        while todo:
            thing = todo.pop()
            if isinstance(thing, Multiple):
                thing = thing.node
            if thing in hierarchies:
                continue  # if already done
            if thing.is_template:
                if not issubclass(thing, File):
                    todo.update(get_all_subclasses(thing))
                continue  # but dont add it directly yet
            todo.update(thing.produces)
            todo.update(thing.hdus.values())
            todo.update(thing.parents)
            bases = get_all_class_bases(thing)
            bases.append(thing)
            todo.update([i for i in all_hierarchies if any(b.singular_name in i.belongs_to for b in bases) and not i.is_template])  # children
            hierarchies.add(thing)
            hierarchies.update(get_all_class_bases(thing))
        return hierarchies


    def __init__(self, rootdir: Union[Path, str] = '/beegfs/car/weave/weaveio/',
                 host: str = '127.0.0.1', port=7687, write=False, dbname='neo4j',
                 password='weavepassword', user='weaveuser', verbose=False):
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.branch_handler = BranchHandler()
        self.handler = Handler(self)
        self.host = host
        self.port = port
        self.write_allowed = write
        self.dbname = dbname
        self._graph = None
        self.password = password
        self.user = user
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.relation_graphs = []
        for i, f in enumerate(self.filetypes):
            fs = self.filetypes[:i+1]
            self.relation_graphs.append(self.make_relation_graph(self.hierarchies_from_filetype(*fs), fs))
        self.relation_graph = self.relation_graphs[-1]
        self.traversal_graph = self.make_traversal_graph()
        self.hierarchies = list(self.relation_graph.nodes)
        self.hierarchies += list({template for h in self.hierarchies for template in get_all_class_bases(h) if template.is_template})
        self.class_hierarchies = {h.__name__: h for h in self.hierarchies}
        self.singular_hierarchies = {h.singular_name: h for h in self.hierarchies}  # type: Dict[str, Type[Hierarchy]]
        self.plural_hierarchies = {h.plural_name: h for h in self.hierarchies if h.plural_name != 'graphables'}
        self.factor_hierarchies = defaultdict(list)
        for h in self.hierarchies:
            for f in getattr(h, 'products_and_factors', []):
                self.factor_hierarchies[f.lower()].append(h)
            if h.idname is not None:
                self.factor_hierarchies[h.idname].append(h)
        self.factor_hierarchies = dict(self.factor_hierarchies)  # make sure we always get keyerrors when necessary!
        self.factors = set(self.factor_hierarchies.keys())
        self.plural_factors =  {make_plural(f.lower()): f.lower() for f in self.factors}
        self.singular_factors = {f.lower() : f.lower() for f in self.factors}
        self.singular_idnames = {h.idname: h for h in self.hierarchies if h.idname is not None}
        self.plural_idnames = {make_plural(k): v for k,v in self.singular_idnames.items()}

    def diff_hierarchy_schema_node(self, hierarchy: Type[Hierarchy]):
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
        results = self.graph.execute(cypher).to_table()
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
                if struct[4]: # is_one2one, so add a reflected relation as well
                    rels.append(Relationship(found_node, 'IS_PARENT_OF', parent, **props))
                parents.append(parent)
            for struct in actual_children:
                props = dict(optional=struct[1], minnumber=struct[2][0], maxnumber=struct[2][1],
                             idname=struct[3])
                child = Node(*struct[0])  # extant child, specify labels so it matches not creates
                rels.append(Relationship(found_node, 'HAS_CHILD', child, **props))  # new rel, should not exists, create
                if struct[4]: # is_one2one, so add a reflected relation as well
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
            rels = [Relationship(found_node, 'HAS_CHILD', c, **props) for c, props in children for i in range((props['maxnumber'] > 1 or props['maxnumber'] is None)+1)] \
                   + [Relationship(p, 'IS_PARENT_OF', found_node, **props) for p, props in parents for i in range((props['maxnumber'] > 1 or props['maxnumber'] is None)+1)]
        return Subgraph(parents+children+[found_node], rels)


    def write_schema(self, dryrun=False):
        """
        writes to the neo4j schema graph for use in optimising queries
        this should always be done before writing data
        """
        # sort available hierarchies meaning non-dependents first
        hierarchies = deque(sorted(self.class_hierarchies.values(),
                                   key=lambda x: len(x.children)+len(x.parents)+len(x.products)))
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
            executions.append(push_py2neo_schema_subgraph_cypher(self.diff_hierarchy_schema_node(hier)))
            nmisses = 0  # reset counter
            done.append(hier)
        if not dryrun:
            for cypher, params in tqdm(executions, desc='schema updates'):
                print(cypher)
                self.graph.execute(cypher, **params)
        return True





    def write(self, collision_manager='track&flag'):
        if self.write_allowed:
            return self.graph.write(collision_manager)
        raise IOError(f"You have not allowed write operations in this instance of data (write=False)")

    def is_unique_factor(self, name):
        return len(self.factor_hierarchies[name]) == 1

    @property
    def graph(self):
        if self._graph is None:
            d = {}
            if self.password is not None:
                d['password'] = self.password
            if self.user is not None:
                d['user'] = self.user
            self._graph = Graph(host=self.host, port=self.port, name=self.dbname, write=self.write, **d)
        return self._graph

    def make_traversal_graph(self):
        graph = self.relation_graph.reverse(copy=True)
        return graph

    def make_relation_graph(self, hierarchies, filetypes):
        reference = hierarchies
        hierarchies = hierarchies.copy()
        graph = nx.DiGraph()
        while hierarchies:
            hierarchy = hierarchies.pop()
            if hierarchy.is_template:
                continue
            graph.add_node(hierarchy)
            for parent in set(hierarchy.parents + hierarchy.produces):
                multiplicity = isinstance(parent, Multiple) and not isinstance(parent, One2One)
                is_one2one = isinstance(parent, One2One)
                if multiplicity:
                    if parent.maxnumber == parent.minnumber:
                        number = parent.maxnumber
                        numberlabel = f'={number}'
                    else:
                        number = None
                        if (parent.minnumber is None or parent.minnumber == 0) and parent.maxnumber is None:
                            numberlabel = 'any'
                        elif (parent.minnumber is None or parent.minnumber == 0) and parent.maxnumber is not None:
                            numberlabel = f'<= {parent.maxnumber}'
                        elif (parent.minnumber is not None and parent.minnumber > 0) and parent.maxnumber is None:
                            numberlabel = f'>={parent.minnumber}'
                        else:
                            numberlabel = f'{parent.minnumber} - {parent.maxnumber}'
                else:
                    number = 1
                    numberlabel = f'={number}'
                if isinstance(parent, Multiple):
                    parent = parent.node
                reverse = parent in hierarchy.produces
                data = dict(multiplicity=multiplicity, number=number, label=numberlabel)
                if parent.is_template:
                    to_do = set(get_all_subclasses(parent))
                    data['oneway'] = True
                    data['label'] += ' (oneway)'
                else:
                    to_do = {parent}
                for p in to_do:
                    if p not in reference:
                        continue
                    if reverse:
                        graph.add_edge(hierarchy, p, **data, real=True)
                        if is_one2one:
                            graph.add_edge(p, hierarchy, **data, real=False)
                    else:
                        graph.add_edge(p, hierarchy, **data, real=True)
                        if is_one2one:
                            graph.add_edge(hierarchy, p, **data, real=False)
                        elif p.singular_name in hierarchy.belongs_to or any(s.singular_name in hierarchy.belongs_to for s in  get_all_subclasses(p)):
                            graph.add_edge(hierarchy, p, multiplicity=True, number=None, label='any', real=False)
        bunch = [i for i in filetypes]
        for filetype in filetypes:
            bunch += list(nx.ancestors(graph, filetype))
            bunch += list(nx.descendants(graph, filetype))
        temp = nx.DiGraph(nx.subgraph(graph, bunch))
        view = nx.subgraph_view(temp, lambda n: not n.is_template)
        return nx.DiGraph(view)

    def make_constraints_cypher(self):
        return {hierarchy: hierarchy.make_schema() for hierarchy in self.hierarchies}

    def apply_constraints(self):
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        templates = []
        equivalencies = []
        for hier, q in tqdm(self.make_constraints_cypher().items(), desc='applying constraints'):
            if q is None:
                templates.append(hier)
            else:
                try:
                    self.graph.neograph.run(q)
                except py2neo.ClientError as e:
                    if '[Schema.EquivalentSchemaRuleAlreadyExists]' in str(e):
                       equivalencies.append(hier)
                       templates.append(hier)
        if len(templates):
            print(f'No index/constraint was made for {templates}')
        if len(equivalencies):
            print(f'EquivalentSchemaRuleAlreadyExists for {equivalencies}')

    def drop_all_constraints(self):
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        self.graph.neograph.run('CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *')

    def get_extant_files(self):
        return self.graph.execute("MATCH (f:File) RETURN DISTINCT f.fname").to_series(dtype=str).values.tolist()

    def raise_collisions(self):
        """
        returns the properties that would have been overwritten in nodes and relationships.
        """
        node_collisions = self.graph.execute("MATCH (c: _Collision) return c { .*}").to_data_frame()
        rel_collisions = self.graph.execute("MATCH ()-[c: _Collision]-() return c { .*}").to_data_frame()
        return node_collisions, rel_collisions

    def read_files(self, *paths: Union[Path, str], raise_on_duplicate_file=False,
                   collision_manager='ignore', batch_size=None, halt_on_error=True,
                   dryrun=False, do_not_apply_constraints=False) -> pd.DataFrame:
        """
        Read in the files given in `paths` to the database.
        `collision_manager` is the method with which the database deals with overwriting data.
        Values of `collision_manager` can be {'ignore', 'overwrite', 'track&flag'}.
        track&flag will have the same behaviour as ignore but places the overlapping data in its own node for later retrieval.
        :return
            statistics dataframe
        """
        if not do_not_apply_constraints:
            self.apply_constraints()
        batches = []
        if len(paths) == 1 and isinstance(paths[0], (tuple, list)):
            paths = paths[0]
        for path in paths:
            path = Path(path)
            matches = [f for f in self.filetypes if f.match_file(self.rootdir, path.relative_to(self.rootdir), self.graph)]
            if len(matches) > 1:
                raise ValueError(f"{path} matches more than 1 file type: {matches} with `{[m.match_pattern for m in matches]}`")
            filetype = matches[0]
            filetype_batch_size = filetype.recommended_batchsize if batch_size is None else batch_size
            slices = filetype.get_batches(path, filetype_batch_size)
            batches += [(filetype, path.relative_to(self.rootdir), slc) for slc in slices]
        elapsed_times = []
        stats = []
        timestamps = []
        if dryrun:
            logging.info(f"Dryrun: will not write to database. However, reading is permitted")
        bar = tqdm(batches)
        for filetype, fname, slc in bar:
            bar.set_description(f'{fname}[{slc.start}:{slc.stop}]')
            try:
                if raise_on_duplicate_file:
                    if len(self.graph.execute('MATCH (f:File {fname: $fname})', fname=fname)) != 0:
                        raise FileExistsError(f"{fname} exists in the DB and raise_on_duplicate_file=True")
                with self.write(collision_manager) as query:
                    filetype.read(self.rootdir, fname, slc)
                cypher, params = query.render_query()
                start = time.time()
                if not dryrun:
                    results = self.graph.execute(cypher, **params)
                    stats.append(results.stats())
                    timestamp = results.evaluate()
                    if timestamp is None:
                        logging.warning(f"This query terminated early due to an empty input table/data. "
                             f"Adjust your `.read` method to allow for empty tables/data")
                    timestamps.append(timestamp)
                elapsed_times.append(time.time() - start)
            except (ClientError, DatabaseError, FileExistsError) as e:
                logging.exception('ClientError:', exc_info=True)
                if halt_on_error:
                    raise e
                print(e)
        if len(batches) and not dryrun:
            df = pd.DataFrame(stats)
            df['timestamp'] = timestamps
            df['elapsed_time'] = elapsed_times
            _, df['fname'], slcs = zip(*batches)
            df['batch_start'], df['batch_end'] = zip(*[(i.start, i.stop) for i in slcs])
        elif dryrun:
            df = pd.DataFrame(columns=['elapsed_time', 'fname', 'batch_start', 'batch_end'])
            df['elapsed_time'] = elapsed_times
        else:
            df = pd.DataFrame(columns=['timestamp', 'elapsed_time', 'fname', 'batch_start', 'batch_end'])
        return df.set_index(['fname', 'batch_start', 'batch_end'])

    def find_files(self, *filetype_names, skip_extant_files=True):
        filelist = []
        if len(filetype_names) == 0:
            filetypes = self.filetypes
        else:
            filetypes = [f for f in self.filetypes if f.singular_name in filetype_names or f.plural_name in filetype_names]
        if len(filetypes) == 0:
            raise KeyError(f"Some or all of the filetype_names are not understood. "
                           f"Allowed names are: {[i.singular_name for i in self.filetypes]}")
        for filetype in filetypes:
            filelist += [i for i in filetype.match_files(self.rootdir, self.graph)]
        if skip_extant_files:
            extant_fnames = self.get_extant_files() if skip_extant_files else []
            filtered_filelist = [i for i in filelist if str(i.relative_to(self.rootdir)) not in extant_fnames]
        else:
            filtered_filelist = filelist
        diff = len(filelist) - len(filtered_filelist)
        if diff:
            print(f'Skipping {diff} extant files (use skip_extant_files=False to go over them again)')
        return filtered_filelist

    def read_directory(self, *filetype_names, collision_manager='ignore', skip_extant_files=True, halt_on_error=False,
                        dryrun=False) -> pd.DataFrame:
        filtered_filelist = self.find_files(*filetype_names, skip_extant_files=skip_extant_files)
        return self.read_files(*filtered_filelist, collision_manager=collision_manager, halt_on_error=halt_on_error,
                                dryrun=dryrun)

    def _validate_one_required(self, hierarchy_name):
        hierarchy = self.singular_hierarchies[hierarchy_name]
        parents = [h for h in hierarchy.parents]
        qs = []
        for parent in parents:
            if isinstance(parent, Multiple):
                mn, mx = parent.minnumber, parent.maxnumber
                b = parent.node.__name__
            else:
                mn, mx = 1, 1
                b = parent.__name__
            mn = 0 if mn is None else mn
            mx = 9999999 if mx is None else mx
            a = hierarchy.__name__
            q = f"""
            MATCH (n:{a})
            WITH n, SIZE([(n)<-[]-(m:{b}) | m ])  AS nodeCount
            WHERE NOT (nodeCount >= {mn} AND nodeCount <= {mx})
            RETURN "{a}", "{b}", {mn} as mn, {mx} as mx, n.id, nodeCount
            """
            qs.append(q)
        if not len(parents):
            qs = [f"""
            MATCH (n:{hierarchy.__name__})
            WITH n, SIZE([(n)<-[:IS_REQUIRED_BY]-(m) | m ])  AS nodeCount
            WHERE nodeCount > 0
            RETURN "{hierarchy.__name__}", "none", 0 as mn, 0 as mx, n.id, nodeCount
            """]
        dfs = []
        for q in qs:
            dfs.append(self.graph.neograph.run(q).to_data_frame())
        df = pd.concat(dfs)
        return df

    def _validate_no_duplicate_relation_ordering(self):
        q = """
        MATCH (a)-[r1]->(b)<-[r2]-(a)
        WHERE TYPE(r1) = TYPE(r2) AND r1.order <> r2.order
        WITH a, b, apoc.coll.union(COLLECT(r1), COLLECT(r2))[1..] AS rs
        RETURN DISTINCT labels(a), a.id, labels(b), b.id, count(rs)+1
        """
        return self.graph.neograph.run(q).to_data_frame()

    def _validate_no_duplicate_relationships(self):
        q = """
        MATCH (a)-[r1]->(b)<-[r2]-(a)
        WHERE TYPE(r1) = TYPE(r2) AND PROPERTIES(r1) = PROPERTIES(r2)
        WITH a, b, apoc.coll.union(COLLECT(r1), COLLECT(r2))[1..] AS rs
        RETURN DISTINCT labels(a), a.id, labels(b), b.id, count(rs)+1
        """
        return self.graph.neograph.run(q).to_data_frame()

    def validate(self):
        duplicates = self._validate_no_duplicate_relationships()
        print(f'There are {len(duplicates)} duplicate relations')
        if len(duplicates):
            print(duplicates)
        duplicates = self._validate_no_duplicate_relation_ordering()
        print(f'There are {len(duplicates)} relations with different orderings')
        if len(duplicates):
            print(duplicates)
        schema_violations = []
        for h in tqdm(list(self.singular_hierarchies.keys())):
            schema_violations.append(self._validate_one_required(h))
        schema_violations = pd.concat(schema_violations)
        print(f'There are {len(schema_violations)} violations of expected relationship number')
        if len(schema_violations):
            print(schema_violations)
        return duplicates, schema_violations

    def find_factor_paths(self, starting_point: Type[Hierarchy], factor_name: str,
                          plural: bool) -> Tuple[Dict[Type[Hierarchy], Set[TraversalPath]], Type[Hierarchy], str]:
        """
        1. Identify all hierarchies that contain the factor under plural constraint
        2. Get paths to those hierarchies with the plural constraint
        3. Discard the hierarchies which don't have paths
        4.
        """
        if factor_name in starting_point.products_and_factors:
            return {starting_point: set()}, starting_point, factor_name
        if factor_name in self.singular_hierarchies:  # if its really a hierarchy name treat it as an idname
            hier = self.singular_hierarchies[factor_name]
            logging.info(f"Turned '{factor_name}' into '{hier.__name__.lower()}.{hier.idname}'")
            factor_name = hier.idname
            possible = {hier}
        else:
            parts = factor_name.split('.')
            if len(parts) > 1:
                hier, factor_name = parts[-2:]
                possible = {self.singular_hierarchies[hier]}
            else:
                possible = {c for c in get_all_subclasses(Hierarchy) if factor_name in c.products_and_factors and not c.is_template}
        pathset = set()
        for p in possible:
            try:
                paths, ends, _, _ = self.find_hierarchy_paths(starting_point, p, plural)
                for path, end in zip(paths, ends):
                    pathset.add((path, end))
            except nx.NetworkXNoPath:
                pass
        if len(pathset) == 0:
            raise nx.NetworkXNoPath(f'There are no paths from a `{starting_point.singular_name}` to `{factor_name}`. '
                                    f'This might be because `{factor_name}` is plural relative to `{starting_point.singular_name}`. '
                                    f'Try using `{make_plural(factor_name)}` instead')
        paths, ends = zip(*pathset)
        if not plural and len(paths) > 1:
            lengths = map(len, paths)
            min_length = min(lengths)
            paths, ends = zip(*[(p, e) for p, e in zip(paths, ends) if len(p) == min_length])
            if len(paths) > 1:
                raise AmbiguousPathError(f"There is more than one '{factor_name}' with the same distance away from '{starting_point.__name__.lower()}'")
        shared = shared_base_class(*ends)
        if factor_name not in shared.products_and_factors:
            raise AmbiguousPathError(f"{starting_point.__name__.lower()}.{factor_name} refers to multiple objects ({ends}) which have no consistent shared parent")
        pathdict = defaultdict(set)
        for e, p in zip(ends, paths):
            pathdict[e].add(p)
        return pathdict, shared, factor_name

    @staticmethod
    def shortest_path_without_oneway_violation(graph: nx.Graph, a, b, cutoff=50):
        """Iterate over shortest paths until one without reusing a one-way path is found"""
        for count, path in enumerate(nx.shortest_simple_paths(graph, a, b)):
            if count == cutoff:
                raise nx.NetworkXNoPath(f'No viable path between {a} and {b} (exploration timed out)')
            oneway_node = None
            for x, y in zip(path[:-1], path[1:]):
                oneway = graph.edges[(x, y)].get('oneway', False)  # is it oneway?
                if oneway:
                    if x == oneway_node:  # violated
                        break
                    else:
                        oneway_node = y
            else:
                return path
        else:
            raise nx.NetworkXNoPath(f'No viable path between {a} and {b}')

    def multiplicity_of_edge(self, a, b, graph=None):
        if graph is None:
            graph = self.relation_graph
        try:
            if not graph.edges[b, a]['multiplicity']:
                return False
        except KeyError:
            pass
        return True


    def _find_restricted_path(self, graph: nx.DiGraph, a: Type[Hierarchy], b: Type[Hierarchy],
                              plural: bool) -> Tuple[TraversalPath, List[bool], nx.DiGraph]:
        try:
            travel_path = self.shortest_path_without_oneway_violation(graph, a, b)
        except nx.NetworkXNoPath:
            path = self.shortest_path_without_oneway_violation(graph, b, a)
            travel_path = path[::-1]
        if any(self.multiplicity_of_edge(a, b, graph) for a, b in zip(travel_path[:-1], travel_path[1:])) and not plural:
            raise nx.NetworkXNoPath
        multiplicity = []
        number = []
        _direction = []
        one_way = []
        for x, y in zip(travel_path[:-1], travel_path[1:]):
            try:
                edge = graph.edges[(y, x)]
            except KeyError:
                multiplicity.append(True)
                number.append(None)
                one_way.append(False)
            else:
                multiplicity.append(edge['multiplicity'])
                number.append(edge['number'])
                one_way.append(edge.get('oneway', False))
        direction = []
        # now reverse the arrow direction if that direction is not real
        for x, y in zip(travel_path[:-1], travel_path[1:]):
            try:
                real = graph.edges[(x, y)].get('real', False)
            except KeyError:
                real = not graph.edges[(y, x)].get('real', False)
            arrow = '->' if real else '<-'
            direction.append(arrow)
        total_path = []
        for i, node in enumerate(travel_path[1:]):
            total_path.append(direction[i])
            total_path.append(node.__name__)
        return TraversalPath(*total_path), one_way, graph

    def _find_hierarchy_path(self, a: Type[Hierarchy], b: Type[Hierarchy], plural: bool) -> Tuple[TraversalPath, List[bool], nx.DiGraph]:
        for i, graph in enumerate(self.relation_graphs):
            try:
                return self._find_restricted_path(graph, a, b, plural)
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError) as e:
                continue
        raise nx.NetworkXNoPath(f"The is no path between `{a.singular_name}` and `{b.singular_name}`. "
                                f"This might be because `{b.singular_name}` is plural relative to `{a.singular_name}`. "
                                f"Try using `{b.plural_name}`")

    def find_hierarchy_paths(self, a: Type[Hierarchy], b: Type[Hierarchy],
                             plural: bool) -> Tuple[List[TraversalPath], List[Type[Hierarchy]], Type[Hierarchy], Type[Hierarchy]]:
        if a.is_template:
            a = [i for i in get_all_subclasses(a) if not i.is_template]
        else:
            a = [a]
        if b.is_template:
            b = [i for i in get_all_subclasses(b) if not i.is_template]
        else:
            b = [b]
        paths = set()
        for ai, bi in product(a, b):
            try:
                path, oneways, graph = self._find_hierarchy_path(ai, bi, plural)
                if sum(oneways) and len(oneways) > 1:  # if >1 there are more paths, if 1 then you are directly connected (so dont explore)
                    if sum(oneways) > 1:
                        raise NotImplementedError(f"Resolving paths with more than 1 oneway is not yet implemented: {path.repr_path}")
                    elif oneways[-1] != True:
                        raise NotImplementedError(f"Resolving paths with the oneway not at the end is not yet implemented: {path.repr_path}")
                    else:
                        just_before = set(nx.descendants(graph, ai)) & set(graph.predecessors(bi))
                        for before in just_before:
                            path, _, _ = self._find_restricted_path(graph, ai, before, plural)
                            paths.add((TraversalPath(*path._path, '->', bi.__name__), bi))
                else:
                    paths.add((path, bi))
            except nx.NetworkXNoPath:
                pass
        if len(paths) == 0:
            raise nx.NetworkXNoPath(f'There are no paths from {a} to {b} with the constraint of plural={plural}')
        paths, ends = zip(*paths)
        if not plural and len(paths) > 1:
            lengths = map(len, paths)
            min_length = min(lengths)
            paths, ends = zip(*[(p, e) for p, e in zip(paths, ends) if len(p) == min_length])
            if len(paths) > 1:
                raise AmbiguousPathError(f"There is more than one {b} with the same distance away from {a}")
        return list(paths), list(ends), shared_base_class(*a), shared_base_class(*ends)

    def is_factor_name(self, name):
        try:
            name = self.singular_name(name)
            return self.is_singular_factor(name) or self.is_singular_idname(name)
        except KeyError:
            return False

    def is_singular_idname(self, value):
        return self.is_singular_name(value) and value.split('.')[-1] in self.singular_idnames

    def is_plural_idname(self, value):
        return self.is_plural_name(value) and value.split('.')[-1] in self.plural_idnames

    def is_plural_factor(self, value):
        return self.is_plural_name(value) and value.split('.')[-1] in self.plural_factors

    def is_singular_factor(self, value):
        return self.is_singular_name(value) and value.split('.')[-1] in self.singular_factors

    def plural_name(self, name):
        pattern = name.split('.')
        if any(map(self.is_plural_name, pattern)):
            return name
        return '.'.join(pattern[:-1] + [make_plural(pattern[-1])])

    def singular_name(self, name):
        pattern = name.split('.')
        return '.'.join([make_singular(p) for p in pattern])

    def is_valid_name(self, name):
        if isinstance(name, str):
            pattern = name.split('.')
            if len(pattern) == 1:
                return self.is_plural_name(name) or self.is_singular_name(name)
            return all(self.is_valid_name(p) for p in pattern)
        return False

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        pattern = name.split('.')
        if len(pattern) == 1:
            return name in self.plural_hierarchies or name in self.plural_factors or name in self.plural_idnames
        return all(self.is_plural_name(n) for n in pattern)

    def is_singular_name(self, name):
        pattern = name.split('.')
        if len(pattern) == 1:
            return name in self.singular_hierarchies or name in self.singular_factors or name in self.singular_idnames
        return all(self.is_singular_name(n) for n in pattern)



    def __getitem__(self, address):
        return self.handler.begin_with_heterogeneous().__getitem__(address)

    def __getattr__(self, item):
        return self.handler.begin_with_heterogeneous().__getattr__(item)

    def plot_relations(self, i=-1, show_hdus=True, fname='relations.pdf', include=None):
        from networkx.drawing.nx_agraph import to_agraph
        if not show_hdus:
            G = nx.subgraph_view(self.relation_graphs[i], lambda n: not issubclass(n, HDU))  # True to get rid of templated
        else:
            G = self.relation_graphs[i]
        if include is not None:
            include = [self.singular_hierarchies[i] for i in include]
            include_list = include.copy()
            include_list += [a for i in include for a in nx.ancestors(G, i)]
            include_list += [d for i in include for d in nx.descendants(G, i)]
            G = nx.subgraph_view(G, lambda n: n in include_list)
        A = to_agraph(G)
        A.layout('dot')
        A.draw(fname)
