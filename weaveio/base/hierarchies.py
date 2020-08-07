from .address import Address


class BaseHierarchy:
    def __init__(self, base, address):
        self.views = {}  # views of {'product_name': Product}
        self.base = base  # overall structure
        self.address = address  # the address to this hierarchy mixture

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.base}/{self.address})>"

    def exists(self):
        raise NotImplementedError

    def _get_products(self, product_name):
        """
        1. Get files from this address
        2. Filter files by those that can produce the named product
        3. Instantiate the empty File objects
        4. Store instantiated files in base DataStore (file instances are singletons)
        5. Read the Product into each File
            - If this is a Table, we read into the mega-store and provide only a view in this class here
            - If not, then we keep the original here and others can take views
        6. Group the Products into a View of the underlying data (AggregatedProducts)
        7. Store the view here
        8. Return the view
        """
        product_type = self.base.product_types[product_name]
        file_types = [file_type for file_type in self.base.file_types if product_name in file_type.product_types]
        files = []
        for file_type in file_types:
            files += file_type.from_address(self.address)  # will return empty list if address is not compatible
        products = []
        for file in files:
            product = getattr(file, product_name)  # implicit read
            products.append(product)
        return product_type.make_view(products)

    def _get_files(self, filetype_name):
        raise NotImplementedError


class IndexableHierarchy(BaseHierarchy):
    def _index_by_address(self, address):
        new_address = self.address.combine(address)
        # Does this address resolve into a list of the same hierarchy type
        hierarchy_matches = []
        for hierarchy_type in self.base.hierarchy_types:
            if new_address.matches_multiple(hierarchy_type):
                hierarchy_matches.append(hierarchy_type)
        if len(hierarchy_matches) == 1:
            return HomogeneousHierarchy(self.base, new_address, hierarchy_matches[0])
        # Does this address resolve into a single uniquely identified hierarchy instance?
        hierarchy_matches = []
        for hierarchy_type in self.base.hierarchy_types:
            if new_address.matches_unique(hierarchy_type):
                hierarchy_matches.append(hierarchy_type)
        if len(hierarchy_matches) == 1:
            return hierarchy_matches[0].from_address(new_address)
        # otherwise, return new HeterogeneousHierarchy
        return HeterogeneousHierarchy(self.base, new_address)


class HeterogeneousHierarchy(IndexableHierarchy):
    def __getitem__(self, address):
        # still in lazy mode because no actual data has been requested
        if not isinstance(address, Address):
            raise TypeError("A HeterogeneousHierarchy can only be index by an address")
        return self._index_by_address(address)

    def _subhierarchy_list(self, hierarchy_type_name):
        """
        Need to look down the tree of this hierarchy (required_by) for `name`
        Take the address and combine it
        """
        return HomogeneousHierarchy(self.base, self.address, self.base.hierarchy_types[hierarchy_type_name])

    def __getattr__(self, name):
        if name in self.base.product_types:
            return self._get_products(name)
        elif name in self.base.file_types:
            return self._get_files(name)
        elif name in self.base.hierarchy_types:
            return self._subhierarchy_list(name)
        elif name in self.base.derived_types:
            return self._get_derived(name)
        else:
            raise AttributeError(f"{name} is not a valid product, file, hierarchy, or derived type")


class HomogeneousHierarchy(IndexableHierarchy):
    def __init__(self, base, address, hierarchy_type):
        super().__init__(base, address)
        self.hierarchy_type = hierarchy_type
        self.hierarchies = []

    def __getitem__(self, key):
        if isinstance(key, str):
           self.hierarchy_type(self.base, self.address.refine(**{self.hierarchy_type.key: key}))
        elif isinstance(key, Address):
            return HomogeneousHierarchy(self.base, self.address.combine(key), self.hierarchy_type)
        else:
            raise TypeError(f"{key} must be a str or Address")

    def keys(self):
        return KeysView(self.base, self.address, self.hierarchy_type)

    def values(self):
        return ValuesView(self.base, self.address, self.hierarchy_type)

    def items(self):
        return ItemsView(self.base, self.address, self.hierarchy_type)

    def __iter__(self):
        return self.values()


class Hierarchy(BaseHierarchy):
    requires = []
    produces = []
    identifiers = []

    def __init__(self, base, address):
        super().__init__(base, address)
        self._requires = {i.__class__.__name__.lower(): i for i in self.requires}
        self._required_by = {}
        for r in self._requires.values():
            r._required_by[self.__class__.__name__.lower()] = self
        self._produces = {i.__class__.__name__.lower(): i for i in self.produces}

    def _subhierarchy_list(self, hierarchy_type_name):
        """
        Need to look down the tree of this hierarchy (required_by) for `name`
        Take the address and combine it
        """
        return HomogeneousHierarchy(self.base, self.address, self._required_by[hierarchy_type_name])

    def _parent_hierarchy(self, hierarchy_type_name):
        """
        Need to take the address labels of this hierarchy and instantiate the parent.
        The parent doesnt have to be the immediate parent
        """
        return self.required_by[hierarchy_type_name](self.address)

    def __getattr__(self, name):
        """
        If name is describing a hierarchy list e.g. OBs etc, then return HomogeneousHierarchy
        >>> OB.exposures  # direct
        <List of Exposures>
        >>> OB.runs  # `Run` is in the `required_by` of `Exposure` so this is allowed
        If name is describing an input, then return that Hierarchy
        >>> OB.targetset  # maybe an alias of `targets`
        If that name is describing a product/file/attr return that
        >>> run.single
        SingleFile
        >>> run.runid

        If it is plural, then list the hierarchies below
        If it is singular, then get the corresponding Hierarchy above
        """
        if name.endswith('s') and name[:-1] in self.required_by_names:
            return self._subhierarchy_list(name[:-1])
        elif name in self.requires_names:
            return self._parent_hierarchy(name)
        elif name.endswith('s') and name[:-1] in self.produces_names:
            return self._get_products(name[:-1])
        else:
            raise AttributeError(f"Unknown attribute {name} for {self}")
