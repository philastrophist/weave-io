

class QueryTree:
    """
    A query tree is a DAG
    All nodes have access to ones which came before them.
    All nodes have a distinct row number
        This allows query writing to be obvious
    All nodes are labelled by hashes which are only dependent on its predecessors and itself
        THis allows the application of one label to any graph
    """

    def add_source(self):
        """
        Add a new node with MATCH instead of OPTIONAL MATCH
        """


    def merge_branch(self):
        """
        Merges another query tree into this one.
        They can only be merged at the end point of this one.
        They must both have the same lineage before this point.
        The result is a a query tree with loose ends
        This is basically merging these trees on prior_node:
            OPTIONAL MATCH (prior_node)-->(new_node1)
            OPTIONAL MATCH (prior_node)-->(new_node2)
        """


    def add_hierarchy(self):
        """
        Add a new match
        This is basically:
            OPTIONAL MATCH ...
        """

    def unwind_data(self, data, name):
        """
        This is
        OPTIONAL MATCH (e)-->(r2: Run)
        UNWIND ['1002813', '1002813', '1002813'] as runid
        WITH r, e, r2, runid
        WHERE r2.runid = runid
        RETURN e
        """


    def transform(self):
        """
        Takes a node(s) of the same depth and a neo4j function
        Creates a new node with the same depth as the deepest input
        This is basically:
            WITH *, function(a, b, c) as name
        """

    def aggregate(self):
        """
        Returns a new node with reduced depth
        This is basically:
            WITH priornode1, priornode2, collect(node)
        """

if __name__ == '__main__':
    # exposures[all(exposures.runs.config.camera == 'red' | exposure.runs[red].runid != exposures.runs[blue].runid) | exposure.runs.mjd == 'mjd'].runs

    q1 = QueryTree('Exposure')
    q2 = QueryTree('')
