from weaveio.basequery.query import FullQuery


class NotYetImplementedError(NotImplementedError):
    pass


class FrozenQuery:
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
