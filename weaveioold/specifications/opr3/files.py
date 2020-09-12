from astropy.io import fits

from ...base.files import File
from .hierarchies import Run, OB, OBSpec, Target
from .products import HomogeneousSpectra, Header


class L1File(File):
	level = 'L1'
	product_types = {'spectra': HomogeneousSpectra, 'header': Header}

	def hdulist(self):
		return fits.open(self.fname)

	def read_header(self):
		return {'header': self.hdulist()[0]}

	def read_obid(self):
		return {'obid': self.header['OBID']}

	def read_camera(self):
		return {'camera': self.header['CAMERA']}

	def read_vph(self):
		return {'vph': self.header['VPH']}

	def read_mode(self):
		return {'mode': self.header['OBSMODE']}


class Single(L1File):
	attributes = ['obid', 'runid', 'camera', 'vph', 'mode']
	parent_hierarchies = [Run]

	def read_runid(self):
		return {'runid': self.header['RUN']}


class Stack(L1File):
	attributes = ['obid', 'runids', 'cameras', 'vphs', 'mode']
	parent_hierarchies = [OB]


class SuperStack(L1File):
	parent_hierarchies = [OBSpec]


class SuperTarget(L1File):
	parent_hierarchies = [Target]
