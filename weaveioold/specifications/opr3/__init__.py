from .address import Address
from .files import Single, Stack, SuperStack, SuperTarget
from .products import Product, Derived, AggregatedProduct


class OpR3DataStore(HeterogeneousHierarchy):
	file_types = [Single, Stack, SuperStack, SuperTarget]
	address_type = Address
