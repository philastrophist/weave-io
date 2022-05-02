import logging
import re
import time
from collections import defaultdict
from functools import reduce
from operator import and_
from pathlib import Path
from typing import Union, List, Tuple, Type, Dict, Set, Callable

import graphviz
import networkx as nx
import pandas as pd
import py2neo
import textdistance
from networkx import NetworkXNoPath, NodeNotFound
from networkx.drawing.nx_pydot import to_pydot
from py2neo import ClientError, DatabaseError
from tqdm import tqdm

from .file import File, HDU
from .graph import Graph
from .hierarchy import Multiple, Hierarchy, Graphable, Optional, OneOf
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

def expand_template_relation(relation):
    """
    Returns a list of relations that relate to each non-template class
    e.g.
    >>> expand_template_relation(Multiple(L1StackSpectrum))
    [Multiple(L1SingleSpectrum), Multiple(L1OBStackSpectrum), Multiple(L1SuperstackSpectrum)]
    """
    if not relation.node.is_template:
        return [relation]
    subclasses = [cls for cls in get_all_subclasses(relation.node) if not cls.is_template]
    return [Multiple(subclass, 0, relation.maxnumber, relation.constrain, relation.relation_idname) for subclass in subclasses]


def add_relation_graph_edge(graph, parent, child, relation: Multiple):
    """
    if an object of type O requires n parents of type P then this is equivalent to defining that instances of those behave as:
        P-(n)->O (1 object of type O has n parents of type P)
    it implicitly follows that:
        O--(m)--P (each of object's parents of type P can be used by an unknown number `m` of objects of type O = many to one)
    if an object of type O requires n children of type C then this is equivalent to defining that instances of those behave as:
        O-(n)->C (1 object has n children of type C)
        it implicitly follows that:
            child-[m]->Object (each child has m parents of type O)
    """
    relation.instantate_node()
    child_defines_parents = relation.node is parent
    for relation in expand_template_relation(relation):
        relation.instantate_node()
        # only parent-->child is in the database
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        if child_defines_parents:  # i.e. parents = [...] is set in the class for this object
            # child instance has n of type Parent, parent instance has unknown number of type Child
            parent = relation.node  # reset from new relations
            graph.add_edge(child, parent, singular=relation.maxnumber == 1,
                           optional=relation.minnumber == 0,
                           style=relstyle)
            graph.add_edge(parent, child, singular=False, optional=True, style='dotted', relation=relation)
        else:  # i.e. children = [...] is set in the class for this object
            # parent instance has n of type Child, each child instance has one of type Parent
            child = relation.node  # reset from new relations
            graph.add_edge(parent, child, singular=relation.maxnumber == 1,
                           optional=relation.minnumber == 0,
                           relation=relation, style=relstyle)
            graph.add_edge(child, parent, singular=True, optional=True, style='solid')


def make_relation_graph(hierarchies: Set[Type[Hierarchy]]):
    graph = nx.DiGraph()
    for h in hierarchies:
        if h not in graph.nodes:
            graph.add_node(h)
        for child in h.children:
            rel = child if isinstance(child, Multiple) else OneOf(child)
            child = child.node if isinstance(child, Multiple) else child
            add_relation_graph_edge(graph, h, child, rel)
        for parent in h.parents:
            rel = parent if isinstance(parent, Multiple) else OneOf(parent)
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
    hiers += tuple(Hierarchy._hierarchies.values())
    return reduce(set.union, map(hierarchies_from_hierarchy, hiers))

def make_arrows(path, forwards: List[bool], descriptors=None):
    descriptors = [descriptors]*len(forwards) if not isinstance(descriptors, (list, tuple)) else descriptors
    descriptors = [f":{descriptor}" if descriptor is not None else "" for descriptor in descriptors]
    assert len(forwards) == len(path) - 1
    forward_arrow = '-[{name}{descriptor}]->'
    backward_arrow = '<-[{name}{descriptor}]-'
    nodes = list(map('(:{})'.format, [p.__name__ for p in path]))
    path_list = []
    for i, (node, forward) in enumerate(zip(nodes[1:], forwards)):
        arrow = forward_arrow if forward else backward_arrow
        if i == len(forwards) - 1:
            arrow = arrow.format(name='{name}', descriptor=descriptors[i])
        else:
            arrow = arrow.format(name="", descriptor=descriptors[i])
        path_list.append(arrow)
        path_list.append(node)
    path_list = path_list[:-1]
    return ''.join(path_list)


def all_paths(g, from_obj, to_obj):
    singles = nx.subgraph_view(g, filter_edge=lambda a, b: g.edges[a, b]['singular'])
    try:
        yield from nx.all_simple_paths(singles, from_obj, to_obj)
    except NetworkXNoPath:
        pass
    try:
        yield from (p[::-1] for p in nx.all_simple_paths(singles, to_obj, from_obj))
    except NetworkXNoPath:
        pass
    forward = nx.subgraph_view(g, filter_edge=lambda a, b: 'relation' in g.edges[(a, b)])
    backward = nx.subgraph_view(g, filter_edge=lambda a, b: 'relation' not in g.edges[(a, b)])
    try:
        yield from nx.all_simple_paths(forward, from_obj, to_obj)
    except NetworkXNoPath:
        pass
    try:
        yield from nx.all_simple_paths(backward, from_obj, to_obj)
    except NetworkXNoPath:
        pass

def all_unidirectional_paths(g, from_obj, to_obj, force_singular):
    paths = []
    for path in all_paths(g, from_obj, to_obj):
        path = tuple(path)
        if path in paths:
            continue
        paths.append(path)
        singular = all(g.edges[(a, b)]['singular'] for a, b in zip(path[:-1], path[1:]))
        if force_singular and not singular:
            return
        forwards = ['relation' in g.edges[edge] for edge in zip(path[:-1], path[1:])]
        if all(forwards) or not any(forwards):
            yield path, singular, forwards

def is_a_constrained_path(g, path):
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        if 'relation' in g.edges[(a, b)]:
            constrain = g.edges[(a, b)]['relation'].constrain
        else:
            constrain = g.edges[(b, a)]['relation'].constrain
        if constrain:
            if any(c in path for c in constrain):
                return True
    return False

def paths_to_hierarchy(g: nx.DiGraph, from_obj, to_obj, force_singular):
    """
    Find path from one obj to another obj with the constraint that the path is singular or not
    raises NetworkXNoPath if there is no path with that constraint
    :returns:
    path: str
    path_yields_singular: bool
    """
    for path, singular, forwards in all_unidirectional_paths(g, from_obj, to_obj, force_singular):
        if not is_a_constrained_path(g, path):
            yield path, forwards, singular


def paths_to_hierarchy_with_above(g: nx.DiGraph, from_obj, to_obj, force_singular, descriptor=None):
    singles = nx.subgraph_view(g, filter_edge=lambda a, b: g.edges[a, b]['singular'])
    ancestors = [from_obj] + list(nx.ancestors(singles, from_obj))
    for ancestor in ancestors:
        for path, forwards, singular in paths_to_hierarchy(g, ancestor, to_obj, force_singular):
            if not any(n in path for n in ancestors):
                yield path, forwards, singular


def plot_graph(G, fname, format):
    return graphviz.Source(to_pydot(G).to_string()).render(fname, format=format)


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
        self.relative_names = defaultdict(dict)
        for h in self.hierarchies:
            for name, relation in h.relative_names.items():
                self.relative_names[name][h.__name__] = relation
        self.relative_names = dict(self.relative_names)
        self.plural_relative_names = {make_plural(name): name for name in self.relative_names}

    def paths_to_hierarchy(self, from_obj: str, to_obj: str, singular: bool, descriptor=None):
        """
        Find path from one obj to another obj with the constraint that the path is singular or not
        raises NetworkXNoPath if there is no path with that constraint
        """
        a, b = map(self.singular_name, [from_obj, to_obj])
        paths = paths_to_hierarchy(self.relation_graphs[-1], self.singular_hierarchies[a], self.singular_hierarchies[b], singular)
        found_any = False
        for path, forwards, singular in paths:
            yield make_arrows(path, forwards, descriptor), singular, path
            found_any = True
        if not found_any:
            if not singular:
                to = f"multiple `{self.plural_name(b)}`"
            else:
                to = f"only one `{self.singular_name(b)}`"
            from_ = self.singular_name(a.lower())
            raise NetworkXNoPath(f"Can't find a link between `{from_}` and {to}. "
                                f"This may be because it doesn't make sense for `{from_}` to have {to}. "
                                f"Try checking the cardinalty of your query.")

    def path_to_hierarchy(self, from_obj: str, to_obj: str, singular: bool, descriptor=None):
        generator = self.paths_to_hierarchy(from_obj, to_obj, singular, descriptor)
        arrows, singular, path = next(generator)
        others = '\n'.join(['.'.join([ii.__name__ for ii in i[-1]]) for i in generator])
        if others:
            raise NetworkXNoPath(f"Found multiple paths between `{from_obj}` and `{to_obj}` via these paths:"
                                 f"\n{others}")

        return arrows, singular

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

    def is_product(self, factor_name, hierarchy_name):
        return self.singular_name(factor_name) in self.singular_hierarchies[self.singular_name(hierarchy_name)]

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
            if issubclass(name, Hierarchy):
                return name.plural_name
            else:
                raise TypeError(f'{name} is not a weaveio object or string')
        if name in self.class_hierarchies:
            return self.class_hierarchies[name].plural_name
        name = name.lower()
        if self.is_plural_name(name):
            return name
        if self.is_singular_name(name):
            try:
                return self.singular_factors[name]
            except KeyError:
                try:
                    return self.singular_idnames[name]
                except KeyError:
                    try:
                        return self.relative_names[name]
                    except KeyError:
                        return self.singular_hierarchies[name].singular_name
        if '.' in name:
            pattern = name.lower().split('.')
            if any(map(self.is_plural_name, pattern)):
                return name
            return '.'.join(pattern[:-1] + [self.plural_name(pattern[-1])])
        return make_plural(name)

    def singular_name(self, name):
        if isinstance(name, type):
            if issubclass(name, Hierarchy):
                return name.singular_name
            else:
                raise TypeError(f'{name} is not a weaveio object or string')
        if name in self.class_hierarchies:
            return self.class_hierarchies[name].singular_name
        name = name.lower()
        if self.is_singular_name(name):
            return name
        if self.is_plural_name(name):
            try:
                return self.plural_factors[name]
            except KeyError:
                try:
                    return self.plural_idnames[name]
                except KeyError:
                    try:
                        return self.plural_relative_names[name]
                    except KeyError:
                        return self.plural_hierarchies[name].singular_name
        if '.' in name:
            pattern = name.lower().split('.')
            return '.'.join([self.singular_name(p) for p in pattern])
        return make_singular(name)

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
            return name in self.plural_hierarchies or name in self.plural_factors or\
                   name in self.plural_idnames or name in self.plural_relative_names
        return all(self.is_plural_name(n) for n in pattern)

    def is_singular_name(self, name):
        pattern = name.split('.')
        if len(pattern) == 1:
            return name in self.singular_hierarchies or name in self.singular_factors or \
                   name in self.singular_idnames or name in self.relative_names
        return all(self.is_singular_name(n) for n in pattern)

    def __getitem__(self, address):
        return self.query.__getitem__(address)

    def __getattr__(self, item):
        return self.query.__getattr__(item)

    def plot_relations(self, i=-1, show_hdus=True, fname='relations', format='pdf', include=None):
        graph = self.relation_graphs[i]
        if not show_hdus:
            graph = nx.subgraph_view(graph, lambda n: not issubclass(n, HDU))  # True to get rid of templated
        # G = nx.subgraph_view(graph, filter_edge=lambda a, b: graph.edges[a, b]['style'] !=  'dotted')
        G = graph
        if include is not None:
            include = [self.singular_hierarchies[i] for i in include]
            include_list = include.copy()
            include_list += [a for i in include for a in nx.ancestors(G, i)]
            include_list += [d for i in include for d in nx.descendants(G, i)]
            G = nx.subgraph_view(G, lambda n: n in include_list)
        plot_graph(G, fname, format)

    def _autosuggest(self, a, relative_to=None):
        a = self.singular_name(a)
        if relative_to is not None:
            relative_to = self.singular_name(relative_to)
        distance, distance_reverse = textdistance.jaro_winkler, True
        suggestions = []

        for h_singular_name, h in self.singular_hierarchies.items():
            newsuggestions = []
            if relative_to is not None:
                try:
                    self.paths_to_hierarchy(relative_to, h_singular_name, singular=True)
                except NetworkXNoPath:
                    hier = h.singular_name
                    factors = h.products_and_factors
                else:
                    hier = h.plural_name
                    factors = [self.plural_name(f) for f in h.products_and_factors]
                newsuggestions.append(hier)
                newsuggestions += factors
            else:
                newsuggestions += [h.singular_name, h.plural_name] + h.products_and_factors
                newsuggestions += [self.plural_name(f) for f in h.products_and_factors]
            try:
                newsuggestions.index(a)
            except ValueError:
                suggestions += newsuggestions
            else:
                return [a]
        inorder = sorted(list(set(suggestions)), key=lambda x: distance(a, x), reverse=distance_reverse)
        return inorder[:3]

    def autosuggest(self, a: str, relative_to: str = None, exception=None):
        suffix = ''
        try:
            l = self._autosuggest(a, relative_to)
            string = '\n'.join([f'{i}. {s}' for i, s in enumerate(l, start=1)])
        except ImportError:
            raise AttributeError(f"`{a}` not understood.{suffix}") from exception
        raise AttributeError(f"`{a}` not understood, did you mean one of:\n{string}{suffix}") from exception