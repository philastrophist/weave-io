
class File:
	product_types = {}

	def __init__(self, fname):
		self.fname = fname

	def __getattr__(self, name):
		if name not in self.product_types:
			raise AttributeError(f"Unknown attribute {name} in {self}")
		try:
			return self.products[name]
		except KeyError:
			self.products[name] = getattr(self, f'read_{name}')()
			return self.products[name]


	@classmethod
	def from_address(cls, address):
		"""
		Returns a list of File objects that match the given address
		"""
		raise NotImplementedError