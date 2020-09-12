from ...base.hierarchies import Hierarchy
from .files import Single, Stack, SuperStack, SuperTarget


class Target(Hierarchy):
	requires = []
	identifiers = ['targetname']


class TargetSet(Hierarchy):
	requires = [Target]


class ProgTemp(Hierarchy):
	requires = []


class ObsTemp(Hierarchy):
	requires = []


class OBSpec(Hierarchy):
	requires = [ObsTemp, ProgTemp, TargetSet]


class OB(Hierarchy):
	requires = [OBSpec]
	identifiers = ['obid']


class Exposure(Hierarchy):
	requires = [OB]


class Run(Hierarchy):
	requires = [Exposure]
	identifiers = ['runid']
