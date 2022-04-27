import logging
import re
import time
from collections import defaultdict
from functools import reduce
from operator import and_
from pathlib import Path
from typing import Union, List, Tuple, Type, Dict, Set, Callable

import networkx as nx
import pandas as pd
import py2neo
from networkx import NetworkXNoPath, NodeNotFound
from py2neo import ClientError, DatabaseError
from tqdm import tqdm

from .file import File, HDU
from .graph import Graph
from .hierarchy import Multiple, Hierarchy, Graphable, One2One, Optional, Single
from .readquery import Query
from .utilities import make_plural, make_singular
from .writequery import Unwind

CONSTRAINT_FAILURE = re.compile(r"already exists with label `(?P<label>[^`]+)` and property "
                                r"`(?P<idname>[^`]+)` = (?P<idvalue>[^`]+)$", flags=re.IGNORECASE)

def get_all_class_bases(cls: Type[Graphable]) -> List[Type[Graphable]]:
    new = []
    for b in cls.__bases__:
        if b is Graphable or not issubclass(b, Graphable):
            continue
        new.append(b)
        new += get_all_class_bases(b)
    return new

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

def add_relation_graph_edge(graph, parent, child, relation: Multiple):
    """
    if an object of type O requires n parents of type P then this is equivalent to defining that instances of those behave as:
    parent-(n)->object (1 object has n parents of type P)
    it implicitly follows that:
        object--(m)-parent (each of object's parents of type P can be used by an unknown number `m` of objects of type O = many to one)
    if an object of type O requires n children of type C then this is equivalent to defining that instances of those behave as:
        object-(n)->child (1 object has n children of type C)
        it implicitly follows that:
            child-[has 1]->Object (each child has maybe 1 parent of type O)
    """
    relation.instantate_node()
    graph.add_edge(child, parent, singular=relation.maxnumber == 1, optional=relation.minnumber == 0, relation=relation)
    if isinstance(relation, One2One) or relation.node is child:
        graph.add_edge(parent, child, singular=True, optional=True, relation=relation)
    else:
        graph.add_edge(parent, child, singular=False, optional=True, relation=relation)


def make_relation_graph(hierarchies: Set[Type[Hierarchy]]):
    graph = nx.DiGraph()
    for h in hierarchies:
        if h not in graph.nodes:
            graph.add_node(h)
        for child in h.children:
            rel = child if isinstance(child, Multiple) else Single(child)
            child = child.node if isinstance(child, Multiple) else child
            add_relation_graph_edge(graph, h, child, rel)
        for parent in h.parents:
            rel = parent if isinstance(parent, Multiple) else Single(parent)
            parent = parent.node if isinstance(parent, Multiple) else parent
            add_relation_graph_edge(graph, parent, h, rel)
    return graph

def hierarchies_from_hierarchy(hier: Type[Hierarchy], done=None) -> Set[Type[Hierarchy]]:
    if done is None:
        done = []
    hierarchies = set()
    todo = set(hier.parents + hier.children + hier.produces)
    for new in todo:
        if isinstance(new, Multiple):
            new.instantate_node()
            h = new.node
        else:
            h = new
        if h not in done and h is not hier:
            hierarchies.update(hierarchies_from_hierarchy(h, done))
            done.append(h)
    hierarchies.add(hier)
    return hierarchies

def hierarchies_from_hierarchies(*hiers: Type[Hierarchy]) -> Set[Type[Hierarchy]]:
    return reduce(set.union, map(hierarchies_from_hierarchy, hiers))

def make_arrows(path, forward=True, type=None):
    arrow = '' if type is None else f'-[:{type}]-'
    if forward:
        arrow = f"-{arrow}->"
    else:
        arrow = f"<-{arrow}-"
    middle = arrow.join(map('(:{})'.format, [p.__name__ for p in path[1:-1]]))
    if middle:
        return "{}{}{}".format(arrow, middle, arrow)
    return arrow

def path_to_hierarchy(g: nx.DiGraph, from_obj, to_obj, singular) -> Tuple[str, bool]:
    """
    Find path from one obj to another obj with the constraint that the path is singular or not
    raises NetworkXNoPath if there is no path with that constraint
    :returns:
    path: str
    path_yields_singular: bool
    """
    singles = nx.subgraph_view(g, filter_edge=lambda a, b: g.edges[a, b]['singular'])  # type: nx.DiGraph
    try:
        try:
            # singular path, but will be plural later
            return make_arrows(nx.shortest_path(singles, from_obj, to_obj)[::-1], False), True
        except NetworkXNoPath as e:
            if singular:
                raise e # singular searches one get one chance
        try:
            # e.g. run.obs find the path ob->run is
            return make_arrows(nx.shortest_path(singles, to_obj, from_obj)), False
        except NetworkXNoPath:
            pass
        return make_arrows(nx.shortest_path(g, from_obj, to_obj)), False
    except NetworkXNoPath:
        raise NetworkXNoPath(f"A single={singular} link between {from_obj} and {to_obj} doesn't make sense. "
                             f"Go via a another object.")


class Data:
    filetypes = []

    def __init__(self, rootdir: Union[Path, str] = '/beegfs/car/weave/weaveio/',
                 host: str = '127.0.0.1', port=7687, write=False, dbname='neo4j',
                 password='weavepassword', user='weaveuser', verbose=False):
        if verbose:
            logging.basicConfig(level=logging.INFO)
        self.query = Query(self)
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
            self.relation_graphs.append(make_relation_graph(hierarchies_from_hierarchies(*fs)))
        self.hierarchies = {h for g in self.relation_graphs for h in g.nodes}

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


    def path_to_hierarchy(self, from_obj: str, to_obj: str, singular: bool):
        """
        Find path from one obj to another obj with the constraint that the path is singular or not
        raises NetworkXNoPath if there is no path with that constraint
        """
        a, b = map(self.singular_name, [from_obj, to_obj])
        for i, g in enumerate(self.relation_graphs):
            try:
                return path_to_hierarchy(g, self.singular_hierarchies[a], self.singular_hierarchies[b], singular)
            except (NodeNotFound, NetworkXNoPath):
                if i == len(self.relation_graphs)-1:
                    if not singular:
                        to = f"multiple `{self.plural_name(b)}`"
                    else:
                        to = f"only one `{self.singular_name(b)}`"
                    from_ = self.singular_name(a.lower())
                    raise NetworkXNoPath(f"Can't find a link between `{from_}` and {to}. "
                                        f"This may be because it doesn't make sense for `{from_}` to have {to}. "
                                        f"Try checking the cardinalty of your query.")

    def all_links_to_hierarchy(self, hierarchy: Type[Hierarchy], edge_constraint: Callable[[nx.DiGraph, Tuple], bool]) -> Set[Type[Hierarchy]]:
        hierarchy = self.class_hierarchies[self.class_name(hierarchy)]
        g = self.relation_graphs[-1]
        singles = nx.subgraph_view(g, filter_edge=lambda a, b: edge_constraint(g, (a, b)))
        hiers = set()
        for node in singles.nodes:
            if nx.has_path(singles, hierarchy, node):
                hiers.add(node)
        return hiers

    def all_single_links_to_hierarchy(self, hierarchy: Type[Hierarchy]) -> Set[Type[Hierarchy]]:
        return self.all_links_to_hierarchy(hierarchy, lambda g, e: g.edges[e]['singular'])

    def all_multiple_links_to_hierarchy(self, hierarchy: Type[Hierarchy]) -> Set[Type[Hierarchy]]:
        return self.all_links_to_hierarchy(hierarchy, lambda g, e: not g.edges[e]['singular'])

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

    def is_factor_name(self, name):
        if name in self.factor_hierarchies:
            return True
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

    def class_name(self, name):
        if isinstance(name, type):
            return name.__name__
        else:
            return self.singular_hierarchies[self.singular_name(name)].__name__

    def plural_name(self, name):
        if isinstance(name, type):
            name = name.__name__
        pattern = name.lower().split('.')
        if any(map(self.is_plural_name, pattern)):
            return name
        return '.'.join(pattern[:-1] + [make_plural(pattern[-1])])

    def singular_name(self, name):
        if isinstance(name, type):
            name = name.__name__
        pattern = name.lower().split('.')
        return '.'.join([make_singular(p) if self.is_plural_name(p) else p for p in pattern])

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
        return self.query.__getitem__(address)

    def __getattr__(self, item):
        return self.query.__getattr__(item)

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
