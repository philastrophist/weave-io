import pytest

from weaveio.basequery.query import Path
from weaveio.basequery.tests.example_structures.one2one import MyData


@pytest.fixture(scope='module')
def workdir(tmpdir_factory):
        return tmpdir_factory.mktemp("data")

@pytest.fixture(scope='module')
def database(workdir):
    for i in range(5):
        fname = Path(str(workdir.join(f'{i}.fits')))
        with open(str(fname), 'w') as file:
            file.write('')
    data = MyData(workdir, port=7687)
    assert data.graph.neograph.name == 'testweaveiodonotuse', "I will not run tests on this database as a safety measure"
    data.graph.neograph.run('MATCH (n) DETACH DELETE n')
    data.directory_to_neo4j()
    data.validate()
    return data