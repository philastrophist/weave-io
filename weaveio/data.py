import time
from functools import reduce
from itertools import product
from operator import or_

import networkx
import numpy as np
import logging
from pathlib import Path
from typing import Union, List, Tuple, Type, Dict
import pandas as pd
import re

import networkx as nx
import py2neo
from tqdm import tqdm

from weaveio.address import Address
from weaveio.basequery.handler import Handler, defaultdict
from weaveio.basequery.tree import BranchHandler, TraversalPath
from weaveio.graph import Graph
from weaveio.writequery import Unwind
from weaveio.hierarchy import Multiple, Hierarchy, Indexed
from weaveio.file import File, HDU

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


def get_all_subclasses(cls):
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



class Data:
    filetypes = []

    def read_in_filetype(self, filetype: File):
        todo = set(filetype.produces + list(filetype.hdus.values()))
        todo.add(filetype)
        while len(todo):
            thing = todo.pop()
            if not thing.is_template:
                self.hierarchies.add(thing)
                for hier in thing.parents:
                    if not thing.is_template:
                        if isinstance(hier, Multiple):
                            todo.add(hier.node)
                        else:
                            todo.add(hier)

    def __init__(self, rootdir: Union[Path, str], host: str = 'host.docker.internal', port=11002, write=False,
                 password=None, user=None):
        self.branch_handler = BranchHandler()
        self.handler = Handler(self)
        self.host = host
        self.port = port
        self.write_allowed = write
        self._graph = None
        self.password = password
        self.user = user
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.address = Address()
        self.hierarchies = set()
        for f in self.filetypes:
            self.read_in_filetype(f)
        self.class_hierarchies = {h.__name__: h for h in self.hierarchies}
        self.singular_hierarchies = {h.singular_name: h for h in self.hierarchies}  # type: Dict[str, Type[Hierarchy]]
        self.plural_hierarchies = {h.plural_name: h for h in self.hierarchies if h.plural_name != 'graphables'}
        self.factor_hierarchies = defaultdict(list)
        for h in self.hierarchies:
            for f in getattr(h, 'factors', []):
                self.factor_hierarchies[f.lower()].append(h)
            if h.idname is not None:
                self.factor_hierarchies[h.idname].append(h)
        self.factor_hierarchies = dict(self.factor_hierarchies)  # make sure we always get keyerrors when necessary!
        self.factors = set(self.factor_hierarchies.keys())
        self.plural_factors =  {f.lower() + 's': f.lower() for f in self.factors}
        self.singular_factors = {f.lower() : f.lower() for f in self.factors}
        self.singular_idnames = {h.idname: h for h in self.hierarchies if h.idname is not None}
        self.plural_idnames = {k+'s': v for k,v in self.singular_idnames.items()}
        self.make_relation_graph()
        # self.make_path_finder()

    def make_path_finder(self):
        """
        Turns a directed graph with one with edges in both directions
        With an edge of (a)-[number=2, multiple=True]>(b), b has 2 of a, a has <unknown> of b
        """
        # reversed because you should travel against the arrows to read the number
        self.path_finder = self.relation_graph.reverse(copy=True)
        for a, b in list(self.path_finder.edges):
            attrs = self.path_finder.edges[(a, b)]
            if attrs.get('belongs_to', False):
                multiplicity = False
                number = 1
            else:
                multiplicity = True
                number = None
            self.path_finder.edges[(a, b)]['inverted'] = False
            self.path_finder.add_edge(b, a, multiplicity=multiplicity, number=number, inverted=True)
        for n in self.path_finder.edges:
            value = self.path_finder.edges[n]['number']
            self.path_finder.edges[n]['number'] = np.inf if value is None else value


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
            self._graph = Graph(host=self.host, port=self.port, write=self.write, **d)
        return self._graph

    def make_relation_graph(self):
        self.relation_graph = nx.DiGraph()
        d = list(self.singular_hierarchies.values())
        while len(d):
            h = d.pop()
            try:
                is_file = issubclass(h, File)
            except:
                is_file = False
            self.relation_graph.add_node(h.singular_name, is_file=is_file, is_hdu=issubclass(h, HDU),
                                         factors=h.factors+[h.idname], idname=h.idname)
            for parent in h.parents:
                multiplicity = isinstance(parent, Multiple)
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
                    parent = parent.node
                else:
                    number = 1
                    numberlabel = f'={number}'
                subclasses = [parent] + get_all_subclasses(parent)
                for subclass in subclasses:
                    if not subclass.is_template:
                        self.relation_graph.add_node(subclass.singular_name, is_file=is_file, is_hdu=issubclass(subclass, HDU),
                                                     factors=subclass.factors+[subclass.idname],
                                                     idname=subclass.idname)
                        self.relation_graph.add_edge(subclass.singular_name, h.singular_name, belongs_to=subclass in h.belongs_to,
                                                     multiplicity=multiplicity, number=number,
                                                     label=numberlabel)
                        d.append(subclass)
            for hier in h.produces:
                self.relation_graph.add_edge(h.singular_name, hier.singular_name, belongs_to=h in hier.belongs_to,
                                             multiplicity=False, number=1, index=False, name='file',
                                             label='=1')
                for product_name in hier.products:
                    if isinstance(product_name, Indexed):
                        index = True
                        product_name = product_name.name
                    else:
                         index = False
                    product = h.hdus[product_name]
                    self.relation_graph.add_edge(product.singular_name, hier.singular_name, belongs_to=product in hier.belongs_to,
                                                 multiplicity=False, number=1, index=index, name=product_name,
                                                 label='=1')

    def make_constraints_cypher(self):
        return [hierarchy.make_schema() for hierarchy in self.hierarchies]

    def apply_constraints(self):
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        for q in tqdm(self.make_constraints_cypher(), desc='applying constraints'):
            self.graph.neograph.run(q)

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

    def read_files(self, *paths: Union[Path, str], collision_manager='ignore', batch_size=None, halt_on_error=False) -> pd.DataFrame:
        """
        Read in the files given in `paths` to the database.
        `collision_manager` is the method with which the database deals with overwriting data.
        Values of `collision_manager` can be {'ignore', 'overwrite', 'track&flag'}.
        track&flag will have the same behaviour as ignore but places the overlapping data in its own node for later retrieval.
        :return
            statistics dataframe
        """
        batches = []
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
        bar = tqdm(batches)
        for filetype, fname, slc in bar:
            bar.set_description(f'{fname}[{slc.start}:{slc.stop}]')
            with self.write(collision_manager) as query:
                filetype.read(self.rootdir, fname, slc)
            cypher, params = query.render_query()
            start = time.time()
            try:
                results = self.graph.execute(cypher, **params)
            except py2neo.database.work.ClientError as e:
                logging.exception('ClientError:', exc_info=True)
                if halt_on_error:
                    raise e
                print(e)
            stats.append(results.stats())
            timestamps.append(results.evaluate())
            elapsed_times.append(time.time() - start)
        df = pd.DataFrame(stats)
        df['timestamp'] = timestamps
        df['elapsed_time'] = elapsed_times
        if len(batches):
            _, df['fname'], slcs = zip(*batches)
            df['batch_start'], df['batch_end'] = zip(*[(i.start, i.stop) for i in slcs])
        else:
            df['fname'], df['batch_start'], df['batch_end'] = [], [], []
        return df.set_index(['fname', 'batch_start', 'batch_end'])

    def read_directory(self, *filetype_names, collision_manager='ignore', skip_extant_files=True, halt_on_error=False):
        filelist = []
        extant_fnames = self.get_extant_files() if skip_extant_files else []
        print(f'Skipping {len(extant_fnames)} extant files (use skip_extant_files=False to go over them again)')
        if len(filetype_names) == 0:
            filetypes = self.filetypes
        else:
            filetypes = [f for f in self.filetypes if f.singular_name in filetype_names]
        for filetype in filetypes:
            filelist += [i for i in self.rootdir.rglob(filetype.match_pattern) if str(i.relative_to(self.rootdir)) not in extant_fnames]
        return self.read_files(*filelist, collision_manager=collision_manager, halt_on_error=halt_on_error)

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

    def node_implies_plurality_of(self, a: str, b: str) -> Tuple[bool, TraversalPath, int, Type[Hierarchy]]:
        """
        returns: multiplicity, number, path
        """
        graph = self.relation_graph
        try:
            path = nx.shortest_path(graph, a, b)
            edges = [(x, y) for x, y in zip(path[:-1], path[1:])]
            multiplicity = [True for e in edges]
            number = [None for e in edges]
            reversed = False
        except nx.NetworkXNoPath:
            path = nx.shortest_path(graph, b, a)[::-1]
            edges = [(y, x) for x, y in zip(path[:-1], path[1:])]
            multiplicity = [graph.edges[e]['multiplicity'] for e in edges]
            number = [graph.edges[e]['number'] for e in edges]
            reversed = True
        direction = ['<-' if reversed else '->' for e in edges]
        total_multiplicity = reduce(or_, multiplicity)
        total_number = np.product([np.inf if n is None else n for n in number])
        if total_number == np.inf:
            total_number = None
        total_path = []
        for i, node in enumerate(path[1:]):
            total_path.append(direction[i])
            hier = self.singular_hierarchies[node]
            total_path.append(hier.__name__)
        return total_multiplicity, total_number, TraversalPath(*total_path), hier

    def is_factor_name(self, name):
        try:
            name = self.singular_name(name)
            return self.is_singular_factor(name) or self.is_singular_idname(name)
        except KeyError:
            return False

    def is_singular_idname(self, value):
        return value.split('.')[-1] in self.singular_idnames

    def is_plural_idname(self, value):
        return value.split('.')[-1] in self.plural_idnames

    def is_plural_factor(self, value):
        return value.split('.')[-1] in self.plural_factors

    def is_singular_factor(self, value):
        return value.split('.')[-1] in self.singular_factors

    def plural_name(self, singular_name):
        split = singular_name.split('.')
        before, singular_name = '.'.join(split[:-1]), split[-1]
        if singular_name in self.singular_idnames:
            return singular_name + 's'
        else:
            try:
                return before + self.singular_factors[singular_name] + 's'
            except KeyError:
                return before + self.singular_hierarchies[singular_name].plural_name

    def singular_name(self, plural_name):
        split = plural_name.split('.')
        before, plural_name = '.'.join(split[:-1]), split[-1]
        if self.is_singular_name(plural_name):
            return plural_name
        if plural_name in self.plural_idnames:
            return plural_name[:-1]
        else:
            try:
                return before + self.plural_factors[plural_name]
            except KeyError:
                return before + self.plural_hierarchies[plural_name].singular_name

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        name = name.split('.')[-1]
        return name in self.plural_hierarchies or name in self.plural_factors or name in self.plural_idnames

    def is_singular_name(self, name):
        name = name.split('.')[-1]
        return name in self.singular_hierarchies or name in self.singular_factors or name in self.singular_idnames

    def __getitem__(self, address):
        return self.handler.begin_with_heterogeneous().__getitem__(address)

    def __getattr__(self, item):
        return self.handler.begin_with_heterogeneous().__getattr__(item)

    def plot_relations(self, show_hdus=True, fname='relations.pdf'):
        from networkx.drawing.nx_agraph import to_agraph
        if not show_hdus:
            G = nx.subgraph_view(self.relation_graph, lambda n: not self.relation_graph.nodes[n]['is_hdu'])
        else:
            G = self.relation_graph
        A = to_agraph(G)
        A.layout('dot')
        A.draw(fname)