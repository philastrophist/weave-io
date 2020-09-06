from collections import defaultdict
from functools import reduce
from pathlib import Path
from json import dumps, loads
from typing import Union, Any

import networkx as nx


lightblue = '#69A3C3'
lightgreen = '#71C2BF'
red = '#D08D90'
orange = '#DFC6A1'
purple = '#a45fed'
pink = '#d50af5'

hierarchy_graph_attrs = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
factor_graph_attrs = {'type': 'factor', 'style': 'filled', 'fillcolor': orange, 'shape': 'box', 'edgecolor': orange}
id_graph_attrs = {'type': 'id', 'style': 'filled', 'fillcolor': purple, 'shape': 'box', 'edgecolor': purple}
file_graph_attrs = {'type': 'l1file', 'style': 'filled', 'fillcolor': lightblue, 'shape': 'box', 'edgecolor': lightblue}


class Data:
    hierarchies = []
    filetypes = []

    def __init__(self, directory: Path, overwrite_index=True):
        self.graph = nx.DiGraph()
        self.make_graph()
        self.directory = directory
        self.index_path = directory / Path('.indices')
        self.files = {}
        if not overwrite_index and self.index_path.exists():
            self.recover_indices()
        else:
            self.mapping = {factor_name: defaultdict(list) for hierarchy in self.hierarchies for factor_name in hierarchy}
            self.mapping.update({id_name: defaultdict(list) for hierarchy in self.hierarchies for id_name in hierarchy})
            self.mapping['filetype'] = {f.name: [] for f in self.filetypes}
            self.mapping['filename'] = {}
        self.update_indices()
        self.persist_indices()

    def make_graph(self):
        self.graph = nx.DiGraph()
        for hierarchy in self.hierarchies:
            self.graph.add_node(hierarchy.name, **hierarchy_graph_attrs)
            for factor in hierarchy.factors:
                if factor not in self.graph.nodes:
                    self.graph.add_node(factor, **factor_graph_attrs)
                    self.graph.add_edge(factor, hierarchy,
                                        color=self.graph.nodes[hierarchy]['edgecolor'],
                                        minnumber=1, maxnumber=1,
                                        headlabel=f'1-1')
            for id in hierarchy.ids:
                if id not in self.graph.nodes:
                    self.graph.add_node(id, **id_graph_attrs)
                    self.graph.add_edge(id, hierarchy,
                                        color=self.graph.nodes[hierarchy]['edgecolor'],
                                        minnumber=1, maxnumber=1,
                                        headlabel=f'1-1')
            for parent_hierarchy, (minnumber, maxnumber) in hierarchy.parent_hierarchies:
                self.graph.add_node(parent_hierarchy.name, **hierarchy_graph_attrs)
                self.graph.add_edge(parent_hierarchy, hierarchy,
                                    color=self.graph.nodes[hierarchy]['edgecolor'],
                                    minnumber=minnumber, maxnumber=maxnumber,
                                    headlabel=f'{minnumber}-{maxnumber}')
        for filetype in self.filetypes:
            self.graph.add_node(filetype.name, **file_graph_attrs)
            for hierarchy, (minnumber, maxnumber) in filetype.parent_hierarchies:
                if hierarchy not in self.graph.nodes:
                    self.graph.add_node(hierarchy, **hierarchy_graph_attrs)
                    self.graph.add_edge(hierarchy, filetype,
                                        color=self.graph.nodes[filetype]['edgecolor'],
                                        minnumber=minnumber, maxnumber=maxnumber,
                                        headlabel=f'{minnumber}-{maxnumber}')

    def persist_indices(self):
        with open(self.index_path, 'w') as file:
            string = dumps(self.mapping, indent=4)
            file.write(string)

    def recover_indices(self):
        with open(self.index_path, 'r') as file:
            string = file.read()
            self.mapping = loads(string)

    def update_indices(self):
        for filetype in self.filetypes:
            for filename in self.directory.glob(filetype.pattern):
                if filename not in self.mapping['filename']:
                    self.mapping['filetype'][filetype].append(filename)
                    file = filetype.read(filename)
                    for name, value in file.factors_and_ids.items():
                        self.mapping[name][value] = filename
                    self.mapping['filename'][filename] = file.factors_and_ids

    def resolve_filenames(self, factors: dict = None, ids: dict = None, filetype: str = None):
        filenames = []
        if factors is not None:
            filenames += [set(self.mapping[name][value]) for name, value in factors.items()]
        if ids is not None:
            filenames += [set(self.mapping[name][value]) for name, value in ids.items()]
        if filetype is not None:
            filenames.append(self.mapping['filetype'][filetype])
        return list(reduce(lambda a, b: a.intersection(b), filenames))

    def is_plural(self, target_node, given):
        """
        Traverses the factor graph to get whether the requested factors/ids imply a plural or singular
        product/filename/etc
        If all necessary factors are known then it is singular
        An ID implies that all factors are known
        Read min-max number going backwards along an edge
        """

    def instantiate_hierarchy(self, hierarchy_type, filename):
        return hierarchy_type(**self.mapping['filename'][filename])

    def read_products(self, filetype, product_name, filenames):
        unified = filetype.product_types[product_name].unified_product
        products = []
        for filename in filenames:
            file = self.files.get(filename, filetype(filename))
            self.files[filename] = file
            product = file.read(product_name)
            products.append(product)
        return unified(products)  # then it can be indexed by things like targets

    def __getitem__(self, address):
        """
        Accepts an address to filter the files
        """

    def __getattr__(self, item):
        """
        Accepts a factor/hierarchy/filetype/
        """
        self.is_plural(item, self.address)


class Indexable:
    def __init__(self, data: Data, address: Address, ids: Dict[str, str]):
        self.data = data
        self.address = address
        self.ids = ids


class HierarchyGroup(Indexable):
    pass


class Hierarchy(Indexable):
    pass


class Address:
    pass


class File:
    pass


class HeterogeneousHierarchyGroup(HierarchyGroup):
    def filter_by_address(self, address: Address) -> 'HeterogeneousHierarchyGroup':
        return HeterogeneousHierarchyGroup(self.data, self.address & address, self.ids)

    def _get_singular_hierarchy(self, hierarchy_name: str) -> Hierarchy:
        """
        For queries like: data[address].ob
        """
        fnames = self.data.resolve_filenames(self.address.factors, self.ids)
        Hierarchy = self.data.hierarchies[hierarchy_name]
        idname = Hierarchy.idname
        ids = {self.data.mapping[fname][idname] for fname in fnames}
        assert len(ids) == 1, "_get_singular has resulted in a plural..."
        return Hierarchy(ids.pop())

    def _get_singular_factor(self, factor_name: str):
        """
        For queries like: data[address].vph
        """
        try:
            return self.address[factor_name]
        except KeyError:
            fnames = self.data.resolve_filenames(self.address.factors, self.ids)
            factors = {self.data.mapping[fname][factor_name] for fname in fnames}
            assert len(factors) == 1, "_get_singular has resulted in a plural..."
            return factors.pop()

    def _get_singular_id(self, id_name: str):
        """
        For queries like: data[address].obid
        """
        try:
            return self.ids[id_name]
        except KeyError:
            fnames = self.data.resolve_filenames(self.address.factors, self.ids)
            ids = {self.data.mapping[fname][id_name] for fname in fnames}
            assert len(ids) == 1, "_get_singular has resulted in a plural..."
            return ids.pop()

    def _get_singular_file(self, file_type: str) -> File:
        """
        For queries like: data[address].single
        """
        filenames = self.data.resolve_filenames(self.address.factors, self.ids, file_type)
        assert len(filenames) == 1
        return self.data.filetypes[file_type](filenames[0])

    def _get_plural_hierarchy(self, hierarchy_name: str) -> 'HomogeneousHierarchyGroup':
        """
        For queries like: data[address].obs
        """
        Hierarchy = self.data.hierarchies[hierarchy_name]
        fnames = self.data.resolve_filenames(self.address.factors, self.ids)
        idname = Hierarchy.idname
        ids = {self.data.mapping[fname] for fname in fnames}
        assert len(ids) == 1, "_get_singular has resulted in a plural..."
        return Hierarchy(ids.pop())

    def _get_singular_factor(self, factor_name: str):
        """
        For queries like: data[address].vph
        """
        return self.address[factor_name]

    def _get_singular_id(self, id_name: str):
        """
        For queries like: data[address].obid
        """
        return self.ids[id_name]

    def _get_singular_file(self, file_type: str) -> File:
        filenames = self.data.resolve_filenames(self.address.factors, self.ids, file_type)
        assert len(filenames) == 1
        return self.data.filetypes[file_type](filenames[0])

    def get_singular(self, item):
        if item in self.data.hierarchies:
            self._get_singular_hierarchy(item)
        elif item in self.data.factors:
            return self._get_singular_factor(item)
        elif item in self.data.ids:
            return self._get_singular_id(item)
        elif item in self.data.filetypes:
            return self._get_singular_file(item)
        else:
            raise KeyError(f"{item} is not a valid Hierarchy/Factor/ID/FileType")



    def get_plural(self, item) -> 'HomogeneousHierarchyGroup':
        if item in self.data.hierarchies:
            self._get_plural_hierarchies(item)
        elif item in self.data.factors:
            return self._get_plural_factors(item)
        elif item in self.data.ids:
            return self._get_plural_ids(item)
        elif item in self.data.filetypes:
            return self._get_plural_files(item)
        else:
            raise KeyError(f"{item} is not a valid Hierarchy/Factor/ID/FileType")

    def __getitem__(self, item: Union[Address, str]) -> 'HeterogeneousHierarchyGroup':
        pass

    def __getattr__(self, item) -> Union[Hierarchy, Any, 'HomogeneousHierarchyGroup']:
        pass


class HomogeneousHierarchyGroup(HierarchyGroup):
    def filter_by_id(self, hierarchy_type, id) -> Hierarchy:
        pass

    def filter_by_address(self, address: Address) -> 'HomogeneousHierarchyGroup':
        return HomogeneousHierarchyGroup(self.data, self.address & address)

    def get_singular(self, item) -> Union[Hierarchy, Any]:
        pass

    def get_plural(self, item) -> 'HomogeneousHierarchyGroup':
        pass

    def __getitem__(self, item: Union[Address, str]) -> Union['HomogeneousHierarchyGroup', Data, Hierarchy]:
        pass

    def __getattr__(self, item):
        pass