import pytest

from weaveio.opr3 import Data

@pytest.fixture
def data():
    return Data()
