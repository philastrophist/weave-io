import pytest

from weaveio.opr3 import Data

@pytest.fixture
def data():
    # return Data()
    return Data(dbname='weaveio', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
