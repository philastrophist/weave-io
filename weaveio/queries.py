import logging
import time
from collections import defaultdict
from copy import deepcopy
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from weaveio.address import Address
from weaveio.file import File
from weaveio.hierarchy import Graphable, Multiple
from weaveio.product import get_product
from weaveio.utilities import quote


class BasicQuery:
    def __init__(self, blocks: List = None, current_varname: str = None,
                 current_label: str = None, counter: defaultdict = None, termination_factor: str = None):
        self.blocks = [] if blocks is None else blocks
        self.current_varname = current_varname
        self.current_label = current_label
        self.counter = defaultdict(int) if counter is None else counter
        self.termination_factor = termination_factor

    def spawn(self, blocks, current_varname, current_label) -> 'BasicQuery':
        if self.termination_factor is not None:
            raise IndexError(f"Cannot continue query if it has terminated at a factor {self.termination_factor}")
        return BasicQuery(blocks, current_varname, current_label, self.counter)

    def terminate(self, blocks, current_varname, current_label, factor_name) -> 'BasicQuery':
        return BasicQuery(blocks, current_varname, current_label, self.counter, factor_name)

    def make(self, branch=False, orders_under=None) -> str:
        """
        Return Cypher query which will return json records with entries of HierarchyName and branch
        `HierarchyName` - The nodes to be realised
        branch - The complete branch for each `HierarchyName` node
        """
        if self.blocks:
            match = '\n\n'.join(['\n'.join(blocks) for blocks in self.blocks])
        else:
            raise ValueError(f"Cannot build a query with no MATCH statements")
        if self.termination_factor is not None:
            if branch:
                raise ValueError(f"May not return branches if returning a property value")
            returns = f"\nRETURN {self.current_varname}.id, " \
                      f"{self.current_varname}.{self.termination_factor}, " \
                      f"{{}} as branch, {{}} as indexer"
        else:
            if branch:
                returns = '\n'.join([f"OPTIONAL MATCH p1=({self.current_varname})<-[*]-(n1)",
                                     f"OPTIONAL MATCH p2=({self.current_varname})-[*]->(n2)",
                                     f"WITH collect(p2) as p2s, collect(p1) as p1s, {self.current_varname}",
                                     f"CALL apoc.convert.toTree(p1s+p2s) yield value",
                                     f"RETURN {self.current_varname} as {self.current_label}, value as branch, indexer"])
                returns = r'//Add Hierarchy Branch'+'\n' + returns
            else:
                returns = f"\nRETURN {self.current_varname}, {{}} as branch, indexer"
        indexers = f"\nOPTIONAL MATCH ({self.current_varname})<-[:INDEXES]-(indexer)"
        if orders_under is not None:
            filt_string = ', '.join([f"{k}: under.{k}" for k in orders_under])
            indexers += '\n' + '\n'.join([
                f"UNWIND $unders as under",
                f"MATCH p=({self.current_varname})<-[:IS_REQUIRED_BY*]-({{{filt_string}}})",
                "WITH *, relationships(p) as rels"])
            returns += ", [r IN rels | r.order] as orders"
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
            if len(blocks) == 0:
                blocks += [[f"MATCH ({name} :{hierarchy_name})"]]
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

    def select_factor_name(self, factor_name):
        return self.terminate(self.blocks, self.current_varname, self.current_label, factor_name)

    def select_factor_of(self, factor_name, hierarchy_name, direction):
        if hierarchy_name != self.current_label:
            hierarchy = self.index_by_hierarchy_name(hierarchy_name, direction)
            return hierarchy.select_factor_name(factor_name)
        else:
            return self.select_factor_name(factor_name)


class Indexable:
    def __init__(self, data, query: BasicQuery):
        self.data = data
        self.query = query

    def index_by_factor(self, factor_name):
        raise NotImplementedError(f"Please specify the parent hierarchy for {factor_name}")

    def index_by_single_hierarchy(self, hierarchy_name, direction):
        try:
            hierarchy = self.data.singular_hierarchies[hierarchy_name]
        except KeyError:
            return self.index_by_factor(hierarchy_name)
        name = hierarchy.__name__
        query = self.query.index_by_hierarchy_name(name, direction)
        return SingleHierarchy(self.data, query, hierarchy)

    def index_by_plural_hierarchy(self, hierarchy_name, direction):
        try:
            hierarchy = self.data.plural_hierarchies[hierarchy_name]
        except KeyError:
            return self.index_by_factor(hierarchy_name)
        query = self.query.index_by_hierarchy_name(hierarchy.__name__, direction)
        return HomogeneousHierarchy(self.data, query, hierarchy)

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
            plural_name = self.data.plural_name(hierarchy_name)
            raise NotImplementedError(f"Can only get a single hierarchy when you have specified it. "
                                      f"Instead try `.{plural_name}`")
        return self.index_by_plural_hierarchy(hierarchy_name, 'below')

    def __getattr__(self, item):
        # if self.data.is_singular_idname(item) or self.data.is_plural_idname(item) or \
        #             self.data.is_singular_factor(item) or self.is_plural_factor(item):
        #     return self.index_by_factor(item)
        return self.index_by_hierarchy_name(item)


class Executable(Indexable):
    return_branch = False

    def __init__(self, data, query, nodetype):
        assert issubclass(nodetype, Graphable)
        super().__init__(data, query)
        self.nodetype = nodetype

    def __call__(self, payload=None):
        starts = time.perf_counter_ns(), time.process_time_ns()
        query_string = self.query.make(self.return_branch)
        result = self.data.graph.neograph.run(query_string, parameters=payload).to_table()  # type: py2neo.database.work.Table
        durations = (time.perf_counter_ns() - starts[0]) * 1e-9, (time.process_time_ns() - starts[1]) * 1e-9
        logging.info(f"Query completed in {durations[0]} secs ({durations[1]}) of which were process time")
        return self._process_result(result)

    def index_by_factor(self, factor_name):
        if self.data.is_plural_name(factor_name):
            name = self.data.singular_name(factor_name)
            plural_requested = True
        else:
            name = factor_name
            plural_requested = False
        plural, direction, path = self.data.node_implies_plurality_of(self.nodetype.singular_name, name)
        if plural and self.data.is_singular_name(factor_name):
            plural_name = self.data.plural_name(factor_name)
            raise KeyError(f"{self} has several possible {plural_name}. Please use `.{plural_name}` instead")
        nodetype = self.data.singular_hierarchies[path[-1]]
        query = self.query.select_factor_of(name, nodetype.__name__, direction)
        if plural or plural_requested:
            return HomogeneousFactor(self.data, query, nodetype, name)
        return SingleFactor(self.data, query, nodetype, name)

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
        if len(result) == 1 and result[0] is None:
            return []
        results = []
        for row in result:
            h = self._process_result_row(row, self.nodetype)
            results.append(h)
        return results


class ExecutableFactor(Executable):
    return_branch = False

    def __init__(self, data, query, nodetype, factor_name):
        super().__init__(data, query, nodetype)
        self.factor_name = factor_name


class SingleFactor(ExecutableFactor):
    def _process_result(self, result):
        return result[0][1]


class HomogeneousFactor(ExecutableFactor):
    def _process_result(self, result):
        if len(result) == 1 and result[0] is None:
            return []
        return [r[1] for r in result]


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
        if not self.data.is_singular_name(name):
            raise ValueError(f"{name} is not recognised as a factor/hierarchy name")
        return self.data.node_implies_plurality_of(self.nodetype.singular_name, name)[:-1]

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
        _, direction, _ = self.implied_plurality_direction_of_node(name)
        return self.index_by_plural_hierarchy(item, direction)


class Products:
    def __init__(self, filenode, product_name, index=None):
        self.filenode = filenode
        self.product_name = product_name
        assert issubclass(filenode.nodetype, File)
        self.default_key = self.filenode.nodetype.product_indexables[product_name]
        graph = self.filenode.data.relation_graph  # type: nx.DiGraph
        if self.default_key is None:
            self.available_keys = None
        else:
            self.available_keys = [f for h in nx.ancestors(graph, filenode.nodetype.singular_name)
                                   for f in graph.nodes[h]['factors']]
        if self.default_key is None and index is not None:
            raise ValueError(f"{filenode.nodetype.singular_name}.{product_name} cannot be indexed")
        if index is not None:
            if isinstance(index, (list, tuple, np.ndarray)):
                index = {self.default_key: index}
            elif isinstance(index, dict):
                index = {k: v if isinstance(v, (list, tuple, np.ndarray)) else v for k, v in index.items()}
            else:
                index = {self.default_key: index}
        self.index = index

    def __getitem__(self, item):
        if isinstance(item, Address):
            return Products(self.filenode.__getitem__(item), self.product_name, self.index)
        return Products(self.filenode, self.product_name, item)

    def __getattr__(self, item):
        if item == self.filenode.nodetype.singular_name:
            return self.filenode
        return self.filenode.__getattr__(item)

    def _process_result(self, result):
        l = np.asarray(result['orders'].tolist())
        which = np.sum(l != 0, axis=0)
        if sum(which > 0) > 1:
            raise NotImplementedError(f"Cannot get data from a multi-index search, please report this")
        try:
            indices = l[:, np.where(which>0)[0][0]]
        except IndexError:
            indices = l[:, 0]  # they are all 0
        product_type = self.filenode.nodetype.products[self.product_name]
        data = []
        for fname, index in zip(result.iloc[:, 1], indices):
            datum = getattr(self.filenode.nodetype(fname), f'read_{self.product_name}')().data
            if self.index is not None:
                datum = datum[index]
            data.append(product_type(datum, None))
        if len(data) == 1:
            return data[0].data
        return product_type.concatenate_data(*data)


    def __call__(self):
        under_type = self.filenode.nodetype.product_indexables[self.product_name]
        starts = time.perf_counter_ns(), time.process_time_ns()
        if self.index is None:
            query_string = self.filenode.fnames.query.make(False)
            result = self.filenode.data.graph.neograph.run(query_string).to_data_frame()
            result['orders'] = 0
            result['orders'] = result['orders'].apply(lambda x: [x])  # just to get a list
        else:
            orders_under = list(self.index.keys())
            query_string = self.filenode.fnames.query.make(False, orders_under)
            index = []
            for i in range(max(len(vs) for vs in self.index.values())):
                d = {}
                for k, v in self.index.items():
                    try:
                        d[k] = v[i]
                    except IndexError:
                        d[k] = v
                index.append(d)
            result = self.filenode.data.graph.neograph.run(query_string, parameters={'unders': index}).to_data_frame()
        durations = (time.perf_counter_ns() - starts[0]) * 1e-9, (time.process_time_ns() - starts[1]) * 1e-9
        logging.info(f"Query completed in {durations[0]} secs ({durations[1]}) of which were process time")
        return self._process_result(result)
