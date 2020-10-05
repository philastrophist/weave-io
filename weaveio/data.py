import numpy as np
import logging
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Union, List
import pandas as pd
import re

import networkx as nx
import py2neo
from tqdm import tqdm

from weaveio.address import Address
from weaveio.graph import Graph, Unwind
from weaveio.hierarchy import Multiple, Graphable
from weaveio.file import Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget, File
from weaveio.neo4j import parse_apoc_tree
from weaveio.neo4jqueries import number_of_relationships
from weaveio.product import get_product
from weaveio.utilities import quote

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
        self.graph = Graph(host=host, port=port)
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
        self.factors = {f.lower() for h in self.hierarchies for f in getattr(h, 'factors', [])}
        self.plural_factors =  {f.lower() + 's': f.lower() for f in self.factors}
        self.singular_factors = {f.lower() : f.lower() for f in self.factors}
        self.singular_idnames = {h.idname: h for h in self.hierarchies if h.idname is not None}
        self.plural_idnames = {k+'s': v for k,v in self.singular_idnames.items()}
        self.make_relation_graph()

    def make_relation_graph(self):
        self.relation_graph = nx.DiGraph()
        d = list(self.singular_hierarchies.values())
        while len(d):
            h = d.pop()
            try:
                is_file = issubclass(h, File)
            except:
                is_file = False
            self.relation_graph.add_node(h.singular_name, is_file=is_file)
            for parent in h.parents:
                multiplicity = isinstance(parent, Multiple)
                self.relation_graph.add_node(parent.singular_name, is_file=is_file)
                self.relation_graph.add_edge(parent.singular_name, h.singular_name, multiplicity=multiplicity)
                d.append(parent)
            try:
                for f in h.factors:
                    self.relation_graph.add_node(f.lower(), is_file=False)
                    self.relation_graph.add_edge(f.lower(), h.singular_name, multiplicity=False)
            except AttributeError:
                pass

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
        if nx.has_path(self.relation_graph, end, start):
            start, end = end, start
            reverse = True
        else:
            reverse = False
        edges = nx.shortest_path(self.relation_graph, start, end)
        plurals = [self.relation_graph.edges[(n1, n2)]['multiplicity'] for n1, n2 in zip(edges[:-1], edges[1:])]
        if reverse:
            edges = edges[::-1]
            plurals = plurals[::-1]
        return [self.plural_name(other) if is_plural else other for other, is_plural in zip(edges[1:], plurals)]

    def node_implies_plurality_of(self, start_node, implication_node):
        if nx.has_path(self.relation_graph, start_node, implication_node):
            if self.relation_graph.nodes[implication_node]['is_file']:
                return False, 'below'
            return True, 'below'
        else:
            edges = nx.shortest_path(self.relation_graph, implication_node, start_node)
            return any(self.relation_graph.edges[(n1, n2)]['multiplicity']
                       for n1, n2 in zip(edges[:-1], edges[1:])), 'above'

    def is_singular_idname(self, value):
        return value in self.singular_idnames

    def is_plural_idname(self, value):
        return value in self.plural_idnames

    def plural_name(self, singular_name):
        try:
            return self.singular_factors[singular_name] + 's'
        except KeyError:
            return self.singular_hierarchies[singular_name].plural_name

    def singular_name(self, plural_name):
        try:
            return self.singular_factors[plural_name[:-1]]
        except KeyError:
            return self.plural_hierarchies[plural_name].singular_name

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        return name in self.plural_hierarchies or name in self.plural_factors

    def is_singular_name(self, name):
        return name in self.singular_hierarchies or name in self.singular_factors

    def __getitem__(self, address):
        return HeterogeneousHierarchy(self, BasicQuery()).__getitem__(address)

    def __getattr__(self, item):
        return HeterogeneousHierarchy(self, BasicQuery()).__getattr__(item)


class BasicQuery:
    def __init__(self, blocks: List = None, current_varname: str = None,
                 current_label: str = None, counter: defaultdict = None):
        self.blocks = [] if blocks is None else blocks
        self.current_varname = current_varname
        self.current_label = current_label
        self.counter = defaultdict(int) if counter is None else counter

    def spawn(self, blocks, current_varname, current_label) -> 'BasicQuery':
        return BasicQuery(blocks, current_varname, current_label, self.counter)

    def make(self, branch=False, property=None) -> str:
        """
        Return Cypher query which will return json records with entries of HierarchyName and branch
        `HierarchyName` - The nodes to be realised
        branch - The complete branch for each `HierarchyName` node
        """
        if self.blocks:
            match = '\n\n'.join(['\n'.join(blocks) for blocks in self.blocks])
        else:
            raise ValueError(f"Cannot build a query with no MATCH statements")
        if property is not None:
            if branch:
                raise ValueError(f"May not return branches if returning a property value")
            returns = f"\nRETURN DISTINCT {self.current_varname}.{property}, {{}}, {{}}\nORDER BY {self.current_varname}.id"
        else:
            if branch:
                returns = '\n'.join([f"WITH DISTINCT {self.current_varname}",
                                     f"OPTIONAL MATCH p1=({self.current_varname})<-[*]-(n1)",
                                     f"OPTIONAL MATCH p2=({self.current_varname})-[*]->(n2)",
                                     f"WITH collect(p2) as p2s, collect(p1) as p1s, {self.current_varname}",
                                     f"CALL apoc.convert.toTree(p1s+p2s) yield value",
                                     f"RETURN {self.current_varname} as {self.current_label}, value as branch, indexer",
                                     f"ORDER BY {self.current_varname}.id"])
                returns = r'//Add Hierarchy Branch'+'\n' + returns
            else:
                returns = f"\nRETURN DISTINCT {self.current_varname}, {{}}, indexer \nORDER BY {self.current_varname}.id"
        indexers = f"\nOPTIONAL MATCH ({self.current_varname})<-[:INDEXES]-(indexer)"
        return f"{match}\n{indexers}\n{returns}"

    def index_by_address(self, address):
        if self.current_varname is None:
            name = '<first>'
        else:
            name = self.current_varname
        blocks = []
        for k, v in address.items():
            k = k.lower()
            count = self.counter[k]
            path_match = f"OPTIONAL MATCH ({k}{count} {{{k}: {quote(v)}}})-[:IS_REQUIRED_BY*]->({name})"
            possible_index_match = f"OPTIONAL MATCH ({name})<-[:INDEXES]-({k}{count+1} {{{k}: {quote(v)}}})"
            with_segment = f"WITH {k}{count}, {k}{count+1}, {name}"
            where = f"WHERE ({k}{count}={k}{count+1} OR {k}{count+1} IS NULL) OR ({name}.{k}={quote(v)})"
            block = [path_match, possible_index_match, with_segment, where]
            self.counter[k] += 2
            blocks.append(block)
        return self.spawn(self.blocks + blocks, self.current_varname, self.current_label)

    def index_by_hierarchy_name(self, hierarchy_name, direction):
        name = '{}{}'.format(hierarchy_name.lower(), self.counter[hierarchy_name])
        if self.current_varname is None:
            first_encountered = False
            blocks = deepcopy(self.blocks)
            for i, block in enumerate(blocks):
                for j, line in enumerate(block):
                    if '<first>' in line and not first_encountered:
                        blocks[i][j] = line.replace('<first>', f'{name}:{hierarchy_name}')
                        first_encountered = True
                    else:
                        blocks[i][j] = line.replace('<first>', f'{name}')
            current_varname = name
        else:
            if direction == 'above':
                arrows = '<-[*]-'
            elif direction == 'below':
                arrows = '-[*]->'
            else:
                raise ValueError(f"Direction must be above or below")
            blocks = self.blocks + [[f"MATCH ({self.current_varname}){arrows}({name}:{hierarchy_name})"]]
            current_varname = name
        self.counter[hierarchy_name] += 1
        return self.spawn(blocks, current_varname, hierarchy_name)

    def index_by_id(self, id_value):
        blocks = self.blocks + [[f"WITH {self.current_varname}",
                                 f"WHERE {self.current_varname}.id = {quote(id_value)}"]]
        return self.spawn(blocks, self.current_varname, self.current_label)


class Indexable:
    def __init__(self, data, query: BasicQuery):
        self.data = data
        self.query = query

    def index_by_single_hierarchy(self, hierarchy_name, direction):
        try:
            hierarchy = self.data.singular_hierarchies[hierarchy_name]
        except KeyError:
            return self.index_by_single_factor(hierarchy_name, direction)
        name = hierarchy.__name__
        query = self.query.index_by_hierarchy_name(name, direction)
        return SingleHierarchy(self.data, query, hierarchy)

    def index_by_plural_hierarchy(self, hierarchy_name, direction):
        try:
            hierarchy = self.data.plural_hierarchies[hierarchy_name]
        except KeyError:
            return self.index_by_plural_factor(hierarchy_name, direction)
        query = self.query.index_by_hierarchy_name(hierarchy.__name__, direction)
        return HomogeneousHierarchy(self.data, query, hierarchy)

    def index_by_single_factor(self, factor_name, direction):
        factor = self.data.singular_factors[factor_name]
        query = self.query.index_by_hierarchy_name(factor, direction)
        return SingleFactor(self.data, query, Factor)

    def index_by_plural_factor(self, factor_name, direction):
        factor = self.data.plural_factors[factor_name]
        query = self.query.index_by_hierarchy_name(factor, direction)
        return HomogeneousFactor(self.data, query, Factor)

    def index_by_address(self, address):
        query = self.query.index_by_address(address)
        return HeterogeneousHierarchy(self.data, query)

    def implied_plurality_direction_of_node(self, name):
        """
        Returns True if the current hierarchy object expects name to be plural by looking at the
        relation graph
        """
        raise NotImplementedError


class HeterogeneousHierarchy(Indexable):
    """
	.<other> - data[Address(vph='green')].vph
		- if <other> is in the address, returns other
		- else raise IndexError
	.<other>s - data.OBs
		- returns HomogeneousStore
	[Address()] - data[Address(vph='green')]
		- Returns HeterogeneousStore filtered by the combined address
	[key]
		- Not implemented
    """
    def __getitem__(self, address):
        if isinstance(address, Address):
            return self.index_by_address(address)
        else:
            raise NotImplementedError("Cannot index by an id over multiple heterogeneous hierarchies")

    def implied_plurality_direction_of_node(self, name):
        return True, 'below'

    def index_by_hierarchy_name(self, hierarchy_name):
        if not self.data.is_plural_name(hierarchy_name):
            raise NotImplementedError(f"Can only index plural hierarchies in a heterogeneous address")
        return self.index_by_plural_hierarchy(hierarchy_name, 'below')

    def __getattr__(self, item):
        return self.index_by_hierarchy_name(item)


class Executable(Indexable):
    return_branch = False

    def __init__(self, data, query, nodetype):
        assert issubclass(nodetype, Graphable)
        super().__init__(data, query)
        self.nodetype = nodetype

    def __call__(self):
        starts = time.perf_counter_ns(), time.process_time_ns()
        result = self.data.graph.neograph.run(self.query.make(self.return_branch)).to_table()  # type: py2neo.database.work.Table
        durations = (time.perf_counter_ns() - starts[0]) * 1e-9, (time.process_time_ns() - starts[1]) * 1e-9
        logging.info(f"Query completed in {durations[0]} secs ({durations[1]}) of which were process time")
        return self._process_result(result)


    def _process_result_row(self, row, nodetype):
        node, branch, indexer = row
        inputs = {}
        for f in nodetype.factors:
            inputs[f] = node[f]
        inputs[nodetype.idname] = node[nodetype.idname]
        try:
            base_query = self[node['id']]
        except TypeError:
            base_query = getattr(self, nodetype.plural_name)[node['id']]
        for p in nodetype.parents:
            if p.singular_name == nodetype.indexer:
                inputs[p.singular_name] = self._process_result_row([indexer, {}, {}], p)
            elif isinstance(p, Multiple):
                inputs[p.plural_name] = getattr(base_query, p.plural_name)
            else:
                inputs[p.singular_name] = getattr(base_query, p.singular_name)
        h = nodetype(**inputs)
        h.add_parent_query(base_query)
        return h

    def _process_result(self, result):
        results = []
        for row in result:
            h = self._process_result_row(row, self.nodetype)
            results.append(h)
        return results


class SingleFactor(Executable):
    return_branch = False

    def _process_result(self, result):
        return result[0][0]['value']


class HomogeneousFactor(Executable):
    return_branch = False

    def _process_result(self, result):
        return [r[0]['value'] for r in result]


class ExecutableHierarchy(Executable):
    pass


class SingleHierarchy(ExecutableHierarchy):
    def __init__(self, data, query, nodetype, idvalue=None):
        super().__init__(data, query, nodetype)
        self.idvalue = idvalue

    def index_by_address(self, address):
        raise NotImplementedError("Cannot index a single hierarchy by an address")

    def index_by_hierarchy_name(self, hierarchy_name):
        if self.data.is_plural_name(hierarchy_name):
            multiplicity, direction = self.implied_plurality_direction_of_node(hierarchy_name)
            return self.index_by_plural_hierarchy(hierarchy_name, direction)
        elif self.data.is_singular_name(hierarchy_name):
            multiplicity, direction = self.implied_plurality_direction_of_node(hierarchy_name)
            if multiplicity:
                plural = self.data.plural_name(hierarchy_name)
                raise KeyError(f"{self} has several possible {plural}. Please use `.{plural}` instead")
            return self.index_by_single_hierarchy(hierarchy_name, direction)
        else:
            raise KeyError(f"{hierarchy_name} is an unknown factor/hierarchy")

    def implied_plurality_direction_of_node(self, name):
        if self.data.is_plural_name(name):
            name = self.data.singular_name(name)
        return self.data.node_implies_plurality_of(self.nodetype.singular_name, name)

    def __getattr__(self, item):
        if item in getattr(self.nodetype, 'products', []):
            return Products(self, item)
        return self.index_by_hierarchy_name(item)

    def __call__(self):
        rs = super().__call__()
        assert len(rs) == 1
        return rs[0]


class HomogeneousHierarchy(ExecutableHierarchy):
    """
	.<other> - OBs.OBspec
		- returns Hierarchy/factor/id/file
	.<other>s
		- return HomogeneousStore (e.g. ob.l1.singles)
	[Address()]
		- return Hierarchy if address describes a unique object
		- return HomogeneousStore if address contains some missing factors
		- return HomogeneousStore  if not
		- raise IndexError if address is incompatible
	[key] - OBs[obid]
		- if indexable by key, return Hierarchy
    """
    def index_by_id(self, idvalue):
        query = self.query.index_by_id(idvalue)
        return SingleHierarchy(self.data, query, self.nodetype, idvalue)

    def implied_plurality_direction_of_node(self, name):
        return self.data.node_implies_plurality_of(self.nodetype.singular_name, name)

    def index_by_address(self, address):
        query = self.query.index_by_address(address)
        return HomogeneousHierarchy(self.data, query, self.nodetype)

    def __getitem__(self, item):
        if isinstance(item, Address):
            return self.index_by_address(item)
        else:
            return self.index_by_id(item)

    def __getattr__(self, item):
        if item in getattr(self.nodetype, 'products', []):
            return Products(self, item)
        if self.data.is_plural_name(item):
            name = self.data.singular_name(item)
        else:
            raise ValueError(f"{self} requires a plural {item}, try `.{self.data.plural_name(item)}`")
        _, direction = self.implied_plurality_direction_of_node(name)
        return self.index_by_plural_hierarchy(item, direction)



class Products:
    def __init__(self, filenode, product_name, index=None):
        self.filenode = filenode
        self.product_name = product_name
        if issubclass(filenode.nodetype, File):
            indexables = self.filenode.nodetype.product_indexables[product_name]
            if indexables is None and index is not None:
                raise ValueError(f"{filenode.nodetype.singular_name}.{product_name} is not to be indexed")
            if not isinstance(indexables, (list, tuple)):
                indexables = [indexables]
            if isinstance(index, (list, tuple, np.ndarray)):
                index = np.asarray(index)
                self.index = pd.DataFrame(index, columns=indexables)
            elif index is None:
                self.index = None
            else:
                self.index = pd.DataFrame([index], columns=indexables)
        elif index is not None:
            raise ValueError(f"{filenode.nodetype.singular_name}.{product_name} is not to be indexed")

    def __getitem__(self, item):
        if isinstance(item, Address):
            return Products(self.filenode.__getitem__(item), self.product_name, self.index)
        return Products(self.filenode, self.product_name, item)

    def __getattr__(self, item):
        if item == self.filenode.nodetype.singular_name:
            return self.filenode
        return self.filenode.__getattr__(item)

    def __call__(self):
        result = self.filenode.__call__()
        if not isinstance(result, (list, tuple)):
            result = [result]
        return get_product(result, self.product_name, self.index)


# class HomogeneousProduct(HomogeneousHierarchy):
#     """
#     Products are not present in the graph
#     query requests are mostly performed on the parent file object
#     """
#     def __init__(self, data, file_query: BasicQuery, file_type: Type[File],
#                  product_type: Type[Product]):
#         super().__init__(data, file_query, file_type)
#         self.file_type = file_type
#         self.product_type = product_type
#
#     def index_by_id(self, idvalue):
#         raise NotImplementedError(f"Products have no id to index with")
#
#     def index_by_address(self, address):
#         homogeneous_hierarchy = super().index_by_address(address)
#         return HomogeneousProduct(self.data, homogeneous_hierarchy.query, self.file_type, self.product_type)
#
#     def __getattr__(self, item):
#         try:
#             return super().__getattr__(item)  # attempt to search hierarchy above the file
#         except KeyError as e:
#             raise KeyError(f"Cannot find {item} in the hierarchy above {self}, you must execute the"
#                            f" query to access product attributes: `query()`") from e
#
#     def __call__(self):
#         files = super().__call__()
#         if files:
#             return files[0].from_files(files)
#         return []
#
#
# class SingleProduct(SingleHierarchy):
#     def __init__(self, data, file_query: BasicQuery, file_type: Type[File], file_idvalue: Any,
#                  product_name: str):
#         super().__init__(data, file_query, file_type, file_idvalue)
#         self.product_name = product_name
#
#     def __getattr__(self, item):
#         try:
#             return super().__getattr__(item)  # attempt to search hierarchy above the file
#         except KeyError as e:
#             raise KeyError(f"Cannot find {item} in the hierarchy above {self}, you must execute the"
#                            f" query to access product attributes: `query()`") from e
#
#     def __call__(self):
#         file = super().__call__()
#         return file.products[self.product_name]()


class OurData(Data):
    filetypes = [Raw, L1Single, L1Stack, L1SuperStack, L1SuperTarget, L2Single, L2Stack, L2SuperTarget]
