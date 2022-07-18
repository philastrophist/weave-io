import pytest
from weaveio import *
from weaveio.opr3 import Data

@pytest.fixture
def data():
    return Data()

def test_example1a(data):
    runid = 1003453
    nsky = sum(data.runs[runid].targuses == 'S')()
    assert nsky == 100

def test_example1b(data):
    nsky = sum(data.runs.targuses == 'S', wrt=data.runs)()  # sum the number of skytargets with respect to their runs
    assert set(nsky) == {100, 160, 198, 200, 299, 360}