import pytest

from weaveio.opr3 import Data

@pytest.fixture(scope='module')
def data():
    return Data(dbname='test', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
