import pytest

from weaveio.opr3 import Data

@pytest.fixture(scope='module')
def data():
    return Data(dbname='opr3btestordering')
