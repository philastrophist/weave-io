from ...hierarchy.hierarchies import Hierarchy
from .files import Single, Stack, SuperStack, SuperTarget


class Target(Hierarchy):
	requires = []
	required_by = [TargetSet]
	produces = [SuperTarget]


class TargetSet(Hierarchy):
	requires = [Target]
	required_by = []
	hierarchies = [OBSpec]


class ProgTemp(Hierarchy):
	requires = []
	required_by = [OBSpec]


class ObsTemp(Hierarchy):
	requires = []
	required_by = [OBSpec]


class OBSpec(Hierarchy):
	requires = [ObsTemp, ProgTemp, TargetSet]
	required_by = [OB]
	produces = [SuperStack]


class OB(Hierarchy):
	requires = [OBSpec]
	required_by = [Exposure]
	produces = [Stack]


class Exposure(Hierarchy):
	requires = [OB]
	required_by = [Run]


class Run(Hierarchy):
	requires = [Exposure]
	required_by = []
	produces = [Single]

