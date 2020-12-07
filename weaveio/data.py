from itertools import product

import networkx
import numpy as np
import logging
from pathlib import Path
from typing import Union
import pandas as pd
import re

import networkx as nx
import py2neo
from tqdm import tqdm

from weaveio.address import Address
from weaveio.basequery.handler import Handler, defaultdict
from weaveio.graph import Graph
from weaveio.writequery import Unwind
from weaveio.hierarchy import Multiple
from weaveio.file import File

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


class Data:
    filetypes = []

    def __init__(self, rootdir: Union[Path, str], host: str = 'host.docker.internal', port=11002, write=False):
        self.handler = Handler(self)
        self.host = host
        self.port = port
        self.write = write
        self._graph = None
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.address = Address()
        self.hierarchies = set()
        todo = set(self.filetypes.copy())
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
        self.hierarchies |= set(self.filetypes)
        self.class_hierarchies = {h.__name__: h for h in self.hierarchies}
        self.singular_hierarchies = {h.singular_name: h for h in self.hierarchies}
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

    def is_unique_factor(self, name):
        return len(self.factor_hierarchies[name]) == 1

    @property
    def graph(self):
        if self._graph is None:
            self._graph = Graph(host=self.host, port=self.port, write=self.write)
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
            self.relation_graph.add_node(h.singular_name, is_file=is_file,
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
                        self.relation_graph.add_node(subclass.singular_name, is_file=is_file,
                                                     factors=subclass.factors+[subclass.idname],
                                                     idname=subclass.idname)
                        self.relation_graph.add_edge(subclass.singular_name, h.singular_name,
                                                     multiplicity=multiplicity, number=number,
                                                     label=numberlabel)
                        d.append(subclass)

    def make_constraints_cypher(self):
        return [hierarchy.make_schema() for hierarchy in self.hierarchies]

    def apply_constraints(self):
        for q in tqdm(self.make_constraints_cypher(), desc='applying constraints'):
            self.graph.neograph.run(q)

    def drop_all_constraints(self):
        self.graph.neograph.run('CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *')

    def get_extant_files(self):
        return self.graph.execute("MATCH (f:File) RETURN DISTINCT f.fname").to_series(dtype=str).values.tolist()

    def directory_to_neo4j(self, *filetype_names):
        extant_fnames = self.get_extant_files()
        for filetype in self.filetypes:
            self.filelists[filetype] = self.rootdir.rglob(filetype.match_pattern)
        for filetype, files in self.filelists.items():
            skipped = 0
            if filetype.__name__ not in filetype_names and len(filetype_names) != 0:
                continue
            for file in tqdm(files, desc=filetype.__name__):
                if file in extant_fnames:
                    skipped += 1
                    continue
                try:
                    with self.graph.write() as query:
                        filetype.read(self.rootdir, file.relative_to(self.rootdir))
                    cypher, parameters = query.render_query()
                    self.graph.neograph.run(cypher, parameters=parameters)
                except py2neo.database.work.ClientError as e:
                    logging.exception('ClientError:', exc_info=True)
                    raise e
            print(f'Skipped {skipped} files')

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

    def node_implies_plurality_of(self, start_node, implication_node):
        start_factor, implication_factor = None, None
        if start_node in self.singular_factors or start_node in self.singular_idnames:
            start_factor = start_node
            start_nodes = [n for n, data in self.relation_graph.nodes(data=True) if start_node in data['factors']]
        else:
            start_nodes = [start_node]
        if implication_node in self.singular_factors or implication_node in self.singular_idnames:
            implication_factor = implication_node
            implication_nodes = [n for n, data in self.relation_graph.nodes(data=True) if implication_node in data['factors']]
        else:
            implication_nodes = [implication_node]
        paths = []
        for start, implication in product(start_nodes, implication_nodes):
            if nx.has_path(self.relation_graph, start, implication):
                paths.append((nx.shortest_path(self.relation_graph, start, implication), 'below'))
            elif nx.has_path(self.relation_graph, implication, start):
                paths.append((nx.shortest_path(self.relation_graph, implication, start)[::-1], 'above'))
        paths.sort(key=lambda x: len(x[0]))
        if not len(paths):
            raise networkx.exception.NodeNotFound(f'{start_node} or {implication_node} not found')
        path, direction = paths[0]
        if len(path) == 1:
            return False, 'above', path, 1
        if direction == 'below':
            if self.relation_graph.nodes[path[-1]]['is_file']:
                multiplicity = False
            else:
                multiplicity = True
        else:
            multiplicity = any(self.relation_graph.edges[(n2, n1)]['multiplicity'] for n1, n2 in zip(path[:-1], path[1:]))
        numbers = [self.relation_graph.edges[(n2, n1)]['number'] for n1, n2 in zip(path[:-1], path[1:])]
        if any(n is None for n in numbers):
            number = None
        else:
            number = int(np.product(numbers))
        return multiplicity, direction, path, number

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

    def plot_relations(self, fname='relations.pdf'):
        from networkx.drawing.nx_agraph import to_agraph
        G = self.relation_graph
        A = to_agraph(G)
        A.layout('dot')
        A.draw(fname)
