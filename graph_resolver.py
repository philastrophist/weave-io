
class HeterogeneousHierarchyGroup(HierarchyGroup):
    def filter_by_address(self, address: Address) -> 'HeterogeneousHierarchyGroup':
        return HeterogeneousHierarchyGroup(self.data, self.address & address, self.ids)

    def _get_singular_hierarchy(self, hierarchy_name: str) -> Hierarchy:
        """
        For queries like: data[address].ob
        """



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