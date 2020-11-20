

class QueryTree:
    """
    A query tree is a DAG
    All nodes have access to ones which came before them.
    All nodes have a distinct row number
        This allows query writing to be obvious
    All nodes are labelled by hashes which are only dependent on its predecessors and itself
        THis allows the application of one label to any graph

    Each node which is added will have a depth associated with it.
    This can be None (meaning unknown) or integer > 0
    A depth of a tree leaf can then be calculated by multiplying all predecessor depths (None propagates to None)

    For filtering, you may attach any number of independent branches to a tree.

    """


    def add_source(self):
        """
        Add a new node with MATCH instead of OPTIONAL MATCH
        """


    def attach_branch(self):
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
            > OPTIONAL MATCH ...
        """

    def unwind_data(self, data, name):
        """
        Add data as a vector
        This is
            OPTIONAL MATCH (e)-->(r2: Run)
            ...
            > UNWIND ['1002813', '1002813', '1002813'] as runid
            ...
            WITH r, e, r2, runid
            WHERE r2.runid = runid
            RETURN e
        """

    def filter(self, condition):
        """
        Destroy references to rows which dont' match the condition
        THis is basically:
            WHERE condition
        """


    def transform(self):
        """
        Takes a node(s) of the same depth and a neo4j function
        Creates a new node with the same depth as the deepest input
        This is basically:
            WITH *, function(a, b, c) as name
        """

    def aggregate(self, node):
        """
        Returns a new node with the same depth as the trunk node
        This is joining a branch back to the trunk
        This is basically:
            WITH priornode1, priornode2, collect(node)
        """


if __name__ == '__main__':
    # exposures[all(exposures.runs.config.camera == 'red' | exposure.runs[red].runid != exposures.runs[blue].runid) | exposure.runs.mjd == 'mjd'].runs
    # exposure.runs[red].runid ==> exposures.runs[exposure.runs.config.camera == 'red'].runid

    # TODO: make these methods external functions, make this functional

    exposures = QueryTree('Exposure', depth=None)
    runs = exposures.add_hierarchy('Run', depth=None)
    config = runs.add_hierarchy('ArmConfig', depth=1)

    reds = config.transform(ScalarComparison('camera', '=', 'red'))
    blues = config.transform(ScalarComparison('camera', '=', 'blue'))

    red_runs = runs.attach_branch(reds).aggregate(reds).filter(reds)
    blue_runs = runs.attach_branch(blues).aggregate(blues).filter(blues)

    red_runs.attach_branch(blue_runs)  # attach to each other (at exposure)
    runids_unequal = red_runs.zip_up(Comparison(red_runs.runid, '<>', red_runs.runid))

    expmjd = exposures.transform(ScalarComparison('expmjd', '=', '57659.145637'))

    runids_unequal.attach_branch(reds).aggregate('any')


    total = exposures.attach_branch(runids_unequal).attach_branch(expmjd).attach_branch(reds)\
        .aggregate(runids_unequal, expmjd, reds).




    red_runs

