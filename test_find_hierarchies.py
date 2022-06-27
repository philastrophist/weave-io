import networkx as nx
import pytest

from find_hierarchies_test import HierarchyGraph
from weaveio.opr3.hierarchy import *
from weaveio.opr3.l1files import *
from weaveio.opr3.l2files import *
from weaveio.opr3.l1 import *
from weaveio.opr3.l2 import *

@pytest.fixture()
def graph():
    g = HierarchyGraph()
    g.initialise()
    return g


def make_path_test(graph, a, b):
    paths = graph.find_paths(a, b)
    for path in paths:
        yield '-'.join(n.__name__ for n in path)


class _TestTemplate:
    a, b = None, None
    expected = None

    def setup_class(self):
        if self.expected is not None:
            self.expected = set(self.expected)
            self.expected_backwards = {'-'.join(path.split('-')[::-1]) for path in self.expected}
        else:
            self.expected_backwards = None

    def test_path(self, graph):
        if self.expected is None:
            with pytest.raises(nx.NetworkXNoPath):
                make_path_test(graph, self.a, self.b)
        else:
            assert set(make_path_test(graph, self.a, self.b)) == self.expected

    def test_path_backwards(self, graph):
        if self.expected_backwards is None:
            with pytest.raises(nx.NetworkXNoPath):
                make_path_test(graph, self.b, self.a)
        else:
            assert set(make_path_test(graph, self.b, self.a)) == self.expected_backwards


class TestGandalfSurvey(_TestTemplate):
    a, b = Gandalf, Survey
    expected = {'Gandalf-L1Spectrum-FibreTarget-SurveyTarget-SurveyCatalogue-Subprogramme-Survey'}

class TestRVSpecfitSurvey(_TestTemplate):
    a, b = RVSpecfit, Survey
    expected = {'RVSpecfit-L1Spectrum-FibreTarget-SurveyTarget-SurveyCatalogue-Subprogramme-Survey'}

class TestRedrockSurvey(_TestTemplate):
    a, b = Redrock, Survey
    expected = {'Redrock-L1Spectrum-FibreTarget-SurveyTarget-SurveyCatalogue-Subprogramme-Survey'}

class TestFitSurvey(_TestTemplate):
    a, b = Fit, Survey
    expected = {'Fit-L1Spectrum-FibreTarget-SurveyTarget-SurveyCatalogue-Subprogramme-Survey'}

class TestFitIngestedSpectrum(_TestTemplate):
    a, b = Fit, IngestedSpectrum
    expected = {'Fit-ModelSpectrum-IngestedSpectrum'}

class TestFitUncombinedIngested(_TestTemplate):
    a, b = Fit, UncombinedIngestedSpectrum
    expected = {'Fit-ModelSpectrum-UncombinedIngestedSpectrum'}
    # TODO: this is actually ok for this database, but it feels wrong

class TestOBRun(_TestTemplate):
    a, b = OB, Run
    expected = {'OB-Exposure-Run'}

class TestL1SpectrumSurvey(_TestTemplate):
    a, b = L1Spectrum, Survey
    expected = {'L1Spectrum-FibreTarget-SurveyTarget-SurveyCatalogue-Subprogramme-Survey'}


class TestOBFit(_TestTemplate):
    a, b = OB, Fit
    expected = {
        'OB-Exposure-Run-RawSpectrum-L1SingleSpectrum-L1SupertargetSpectrum-Fit',
        'OB-Exposure-Run-RawSpectrum-L1SingleSpectrum-L1SuperstackSpectrum-Fit',
        'OB-L1StackSpectrum-Fit',
        'OB-Exposure-Run-RawSpectrum-L1SingleSpectrum-Fit',
    }

class TestOBArmConfig(_TestTemplate):
    a, b = ArmConfig, OB
    expected = {'ArmConfig-InstrumentConfiguration-Progtemp-OBSpec-OB'}


class TestArmconfigL1Single(_TestTemplate):
    a, b = ArmConfig, L1SingleSpectrum
    expected = {'ArmConfig-L1SingleSpectrum'}

class TestArmConfigL1Stack(_TestTemplate):
    a, b = ArmConfig, L1StackSpectrum
    expected = {'ArmConfig-L1StackSpectrum'}

class TestArmConfigL1Superstack(_TestTemplate):
    a, b = ArmConfig, L1SuperstackSpectrum
    expected = {'ArmConfig-L1SuperstackSpectrum'}

class TestArmConfigL1SuperTarget(_TestTemplate):
    a, b = ArmConfig, L1SupertargetSpectrum
    expected = {'ArmConfig-L1SupertargetSpectrum'}

class TestOBNoSS(_TestTemplate):
    a, b = OB, NoSS
    expected = {
        'OB-Exposure-Run-RawSpectrum-L1SingleSpectrum-NoSS',
        'OB-L1StackSpectrum-NoSS',
        'OB-Exposure-Run-RawSpectrum-L1SingleSpectrum-L1SuperstackSpectrum-NoSS',
        'OB-Exposure-Run-RawSpectrum-L1SingleSpectrum-L1SupertargetSpectrum-NoSS'
    }

class TestL1SpectrumTemplate(_TestTemplate):
    a, b = L1Spectrum, Template
    expected = {'L1Spectrum-Template'}

class TestRunWeaveTarget(_TestTemplate):
    a, b = Run, WeaveTarget
    expected = {'Run-Exposure-OB-OBSpec-FibreTarget-SurveyTarget-WeaveTarget'}