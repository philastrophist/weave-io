from time import sleep
from warnings import warn

import py2neo
from py2neo import Graph as NeoGraph

from weaveio.context import ContextMeta
from weaveio.writequery import CypherQuery


class Graph(metaclass=ContextMeta):
    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get("graph") is not None:
            instance._parent = kwargs.get("graph")
        else:
            instance._parent = cls.get_context(error_if_none=False)
        return instance

    def __init__(self, profile=None, name=None, **settings):
        self.neograph = NeoGraph(profile, name, **settings)

    def create_unique_constraint(self, label, key):
        try:
            self.neograph.schema.create_uniqueness_constraint(label, key)
        except py2neo.database.work.ClientError:
            pass

    def drop_unique_constraint(self, label, key):
        try:
            self.neograph.schema.drop_uniqueness_constraint(label, key)
        except py2neo.database.work.DatabaseError:
            pass

    def write(self):
        return CypherQuery()

    def _execute(self, cypher, parameters, backoff=1, limit=10):
        try:
            return self.neograph.run(cypher, parameters=parameters)
        except RuntimeError as e:
            if backoff >= limit:
                raise e
            warn(f'Connection possibly busy, waiting {backoff} seconds to retry')
            sleep(backoff)
            backoff *= 2
            return self._execute(cypher, parameters, backoff, limit)

    def execute(self, cypher, **payload):
        import pandas as pd
        d = {}
        for k, v in payload.items():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                v = pd.DataFrame(v).reset_index().to_dict('records')
            d[k] = v
        return self._execute(cypher, d)

Graph._context_class = Graph


