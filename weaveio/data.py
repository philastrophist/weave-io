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
from weaveio.basequery.hierarchy import HeterogeneousHierarchyFrozenQuery
from weaveio.graph import Graph, Unwind
from weaveio.hierarchy import Multiple
from weaveio.file import Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget, File
from weaveio.queries import BasicQuery, HeterogeneousHierarchy

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


class Data:
    filetypes = []

    def __init__(self, rootdir: Union[Path, str], host: str = 'host.docker.internal', port=11002):
        self.handler = Handler(self)
        self.host = host
        self.port = port
        self._graph = None
        self.filelists = {}
        self.rootdir = Path(rootdir)
        self.address = Address()
        self.hierarchies = set()
        todo = set(self.filetypes.copy())
        while len(todo):
            thing = todo.pop()
            self.hierarchies.add(thing)
            for hier in thing.parents:
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
            self._graph = Graph(host=self.host, port=self.port)
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
                self.relation_graph.add_node(parent.singular_name, is_file=is_file,
                                             factors=parent.factors+[h.idname], idname=h.idname)
                self.relation_graph.add_edge(parent.singular_name, h.singular_name, multiplicity=multiplicity)
                d.append(parent)

    def make_constraints(self):
        for hierarchy in self.hierarchies:
            self.graph.create_unique_constraint(hierarchy.__name__, 'id')

    def drop_constraints(self):
        for hierarchy in self.hierarchies:
            self.graph.drop_unique_constraint(hierarchy.__name__, 'id')

    def directory_to_neo4j(self, *filetype_names):
        for filetype in self.filetypes:
            self.filelists[filetype] = list(filetype.match(self.rootdir))
        with self.graph:
            self.make_constraints()
            for filetype, files in self.filelists.items():
                if filetype.__name__ not in filetype_names and len(filetype_names) != 0:
                    continue
                for file in tqdm(files, desc=filetype.__name__):
                    tx = self.graph.begin()
                    f = filetype(file)
                    if not tx.finished():
                        try:
                            self.graph.commit()
                        except py2neo.database.work.ClientError as e:
                            process_neo4j_error(self, f, e.message)
                            logging.exception(self.graph.make_statement(), exc_info=True)
                            raise e
            logging.info('Cleaning up...')
            self.graph.execute_cleanup()

    def validate(self, *hierarchy_names):
        bads = []
        if len(hierarchy_names) == 0:
            hierarchies = self.hierarchies
        else:
            hierarchies = [h for h in self.hierarchies if h.__name__ in hierarchy_names]
        print(f'scanning {len(hierarchies)} hierarchies')
        for hierarchy in tqdm(hierarchies):
            for parent in hierarchy.parents:
                if isinstance(parent, Multiple):
                    lower, upper = parent.minnumber or 0, parent.maxnumber or np.inf
                    parent_name = parent.node.__name__
                else:
                    lower, upper = 1, 1
                    parent_name = parent.__name__
                child_name = hierarchy.__name__
                query = f"MATCH (parent: {parent_name}) MATCH (parent)-->(child: {child_name}) " \
                        f"RETURN parent.id, child.id, '{child_name}' as child_label, '{parent_name}' as parent_label"
                result = self.graph.neograph.run(query)
                df = result.to_data_frame()
                if len(df) == 0 and lower > 0:
                    bad = pd.DataFrame({'child_label': child_name, 'child.id': np.nan,
                                       'nrelationships': 0, 'parent_label': parent_name,
                                       'expected': f'[{lower}, {upper}]'}, index=[0])
                    bads.append(bad)
                elif len(df):
                    grouped = df.groupby(['child_label', 'child.id']).apply(len)
                    grouped.name = 'nrelationships'
                    bad = pd.DataFrame(grouped[(grouped < lower) | (grouped > upper)]).reset_index()
                    bad['parent_name'] = parent_name
                    bad['expected'] = f'[{lower}, {upper}]'
                    bads.append(bad)
        if len(bads):
            bads = pd.concat(bads)
            print(bads)
        print(f"There are {len(bads)} potential violations of expected relationship number")

    def traversal_path(self, start, end):
        multiplicity, direction, path = self.node_implies_plurality_of(start, end)
        a, b = path[:-1], path[1:]
        if direction == 'above':
            b, a = a, b
            plurals = [self.relation_graph.edges[(n1, n2)]['multiplicity'] for n1, n2 in zip(a, b)]
            names = [self.plural_name(other) if is_plural else other for other, is_plural in zip(path[1:], plurals)]
        else:
            names = [self.plural_name(p) for p in path[1:]]
        if start in self.singular_factors or start in self.singular_idnames:
            if self.is_singular_name(names[0]):
                names.insert(0, start)
            else:
                names.insert(0, self.plural_name(start))
        if end in self.singular_factors or end in self.singular_idnames:
            if self.is_singular_name(names[-1]):
                names.append(end)
            else:
                names.append(self.plural_name(end))
        return names

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
            return False, 'above', path
        if direction == 'below':
            if self.relation_graph.nodes[path[-1]]['is_file']:
                multiplicity = False
            else:
                multiplicity = True
        else:
            multiplicity = any(self.relation_graph.edges[(n2, n1)]['multiplicity'] for n1, n2 in zip(path[:-1], path[1:]))
        return multiplicity, direction, path

    def is_singular_idname(self, value):
        return value in self.singular_idnames

    def is_plural_idname(self, value):
        return value in self.plural_idnames

    def is_plural_factor(self, value):
        return value in self.plural_factors

    def is_singular_factor(self, value):
        return value in self.singular_factors

    def plural_name(self, singular_name):
        if singular_name in self.singular_idnames:
            return singular_name + 's'
        else:
            try:
                return self.singular_factors[singular_name] + 's'
            except KeyError:
                return self.singular_hierarchies[singular_name].plural_name

    def singular_name(self, plural_name):
        if plural_name in self.plural_idnames:
            return plural_name[:-1]
        else:
            try:
                return self.plural_factors[plural_name]
            except KeyError:
                return self.plural_hierarchies[plural_name].singular_name

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        return name in self.plural_hierarchies or name in self.plural_factors or name in self.plural_idnames

    def is_singular_name(self, name):
        return name in self.singular_hierarchies or name in self.singular_factors or name in self.singular_idnames

    def __getitem__(self, address):
        return self.handler.begin_with_heterogeneous().__getitem__(address)
        # return HeterogeneousHierarchy(self, BasicQuery()).__getitem__(address)

    def __getattr__(self, item):
        return self.handler.begin_with_heterogeneous().__getattr__(item)
        # return HeterogeneousHierarchy(self, BasicQuery()).__getattr__(item)


class OurData(Data):
    filetypes = [Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget]
