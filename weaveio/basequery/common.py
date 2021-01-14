from copy import deepcopy
from typing import Tuple, Dict, Any

import py2neo

from weaveio.basequery.parse_tree import parse, write_tree, branch2query
from weaveio.basequery.tree import Branch
from weaveio.writequery import CypherQuery


class AmbiguousPathError(Exception):
    pass


class NotYetImplementedError(NotImplementedError):
    pass


class UnexpectedResult(Exception):
    pass


class FrozenQuery:
    executable = True

    def __init__(self, handler, branch: Branch, parent: 'FrozenQuery' = None):
        self.handler = handler
        self.branch = branch
        self.parent = parent
        self.data = self.handler.data

    def _traverse_frozenquery_stages(self):
        query = self
        yield query
        while query.parent is not None:
            query = query.parent
            yield query

    def _prepare_branch(self) -> Branch:
        """Override to allow custom edits to the branch before execution"""
        return deepcopy(self.branch)

    def _prepare_query(self) -> CypherQuery:
        """Override to allow custom edits to the CypherQuery object after the branch is finalised"""
        branch = self._prepare_branch()
        query = branch2query(branch)
        return query

    def _prepare_cypher(self) -> Tuple[str, Dict[str, Any]]:
        """Override to allow custom edits to the actual cypher query text"""
        query = self._prepare_query()
        cypher, params = query.render_query()
        return cypher, params

    def _execute_query(self):
        """Override to allow custom edits as to how the cypher text is run"""
        if not self.executable:
            raise TypeError(f"{self.__class__} may not be executed as queries in their own right")
        cypher, params = self._prepare_cypher()
        return self.data.graph.execute(cypher, **params)

    def _post_process(self, result: py2neo.Cursor):
        """Override to turn a py2neo neo4j result object into something that the user wants"""
        raise NotImplementedError

    def __call__(self):
        """Prepare and execute the query contained by this frozen object"""
        result = self._execute_query()
        return self._post_process(result)
