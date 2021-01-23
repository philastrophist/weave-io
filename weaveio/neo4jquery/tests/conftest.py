from datetime import datetime

import py2neo
import pytest
from py2neo.wiring import WireError

from weaveio.neo4jquery.procedures.make_procedure import get_all_procedures


@pytest.fixture(scope='module')
def procedure_tag():
    return str(hash(datetime.now())).replace('-', '')[:5]


@pytest.fixture(scope='function')
def database(procedure_tag) -> py2neo.Graph:
    try:
        graph = py2neo.Graph(port=7687, host='host.docker.internal')
        assert graph.name == 'testweaveiodonotuse', "I will not run tests on this database as a safety measure"
        graph.run('MATCH (n) DETACH DELETE n')
        for text in get_all_procedures('write', procedure_tag):
            graph.run(text)
        return graph
    except (AssertionError, WireError):
        pytest.xfail("unsupported configuration of testing database")

