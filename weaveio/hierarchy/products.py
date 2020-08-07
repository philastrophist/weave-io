
class Product:
	@classmethod
	def make_view(cls, products):
		if any(not isinstance(p, Product) for p in products):
			raise TypeError(f"Can only make Views for products if they are all of type {cls}")


class Derived(Product):
	pass

