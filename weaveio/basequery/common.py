from copy import deepcopy

from weaveio.basequery.parse_tree import parse, write_tree
from weaveio.basequery.query import FullQuery
from weaveio.basequery.tree import BranchHandler
from weaveio.neo4j import parse_apoc_tree
from weaveio.writequery import CypherQuery


class NotYetImplementedError(NotImplementedError):
    pass


class UnexpectedResult(Exception):
    pass


class FrozenQuery:
    executable = True

    def __init__(self, handler, branch: BranchHandler, parent: 'FrozenQuery' = None):
        self.handler = handler
        self.branch = branch
        self.parent = parent

    @property
    def data(self):
        return self.handler.data

    def _traverse_frozenquery_stages(self):
        query = self
        yield query
        while query.parent is not None:
            query = query.parent
            yield query

    def _prepare_query(self):
        return deepcopy(self.branch)

    def _execute_query(self):
        if not self.executable:
            raise TypeError(f"{self.__class__} may not be executed as queries in their own right")
        branch = self._prepare_query()
        subqueries = parse(branch)
        with CypherQuery() as query:
            for s in subqueries:
                write_tree(s)
        cypher, params = query.render_query()
        return self.data.graph.execute(cypher, **params)

    def _post_process(self, result):
        raise NotImplementedError

    def __call__(self):
        result = self._execute_query()
        return self._post_process(result)
