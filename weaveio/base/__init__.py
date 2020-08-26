from collections import defaultdict

from .address import Address
from .hierarchies import Hierarchy, HeterogeneousHierarchy, HomogeneousHierarchy
from .files import File
from .products import Product, Derived


class DataStore(HeterogeneousHierarchy):
	file_types = []
	address_type = None

	def __init__(self, basename):
		super(DataStore, self).__init__(self, self.address_type())
		self.basename = basename
		self.product_types = defaultdict(dict)
		self.hierarchies = defaultdict(set)
		for f in self.file_types:
			for name, Prod in f.product_types.items():
				self.product_types[name][f] = Prod
		self.files = set()

