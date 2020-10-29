from weaveio.basequery.query import FullQuery
from weaveio.neo4j import parse_apoc_tree


class NotYetImplementedError(NotImplementedError):
    pass


class FrozenQuery:
    executable = True

    def __init__(self, handler, query: FullQuery, parent: 'FrozenQuery' = None):
        self.handler = handler
        self.query = query
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
        return self.query.copy()

    def _execute_query(self):
        if not self.executable:
            raise TypeError(f"{self.__class__} may not be executed as queries in their own right")
        query = self._prepare_query()
        cypher = query.to_neo4j()
        return self.data.graph.execute(cypher)

    def _post_process(self, result):
        raise NotImplementedError

    def __call__(self):
        result = self._execute_query()
        return self._post_process(result)
