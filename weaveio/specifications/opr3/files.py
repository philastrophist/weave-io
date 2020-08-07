from ...hierarchy.files import File
from .products import Run, 


class L1File(File):
	level = 'L1'
	product_types = {'spectra': HomogeneousSpectra}


class Single(L1File):
	pass


class Stack(L1File):
	pass


class SuperStack(L1File):
	pass


class SuperTarget(L1File):
	pass
