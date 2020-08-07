from ...base.products import Product, Derived


class Spectra(Product):
	name = 'spectra'


class HomogeneousSpectra(Spectra):
	"""
	Spectra all on the same wavelength array
	"""


class GroupedSpectra(Spectra):
	"""
	A group of homogeneousSpectra i.e. for blue and red arms of the spectrograph
	This is a view into n HomogeneousSpectra objects
	"""


class HeterogeneousSpectra(Spectra):
	"""
	An assortment of spectra without a specific order. i.e. all spectra at once
	"""


class Table(Product):
	pass


class Header(Product):
	pass


class FibTable(Table):
	pass


