from typing import Dict, Set

from .actions import TraversalPath
from .hierarchy import *
from .tree import BranchHandler


class Handler:
    def __init__(self, data: 'Data'):
        self.data = data
        self.branch_handler = data.branch_handler  # type: BranchHandler

    def begin_with_heterogeneous(self):
        return HeterogeneousHierarchyFrozenQuery(self, self.branch_handler.entry)

    def hierarchy_from_neo4j_identity(self, htype, identity):
        begin = self.branch_handler.begin(htype.__name__)
        new = begin.add_data(identity)
        identifier_var = new.current_variables[0]
        branch = new.filter('id({h}) = {identifier}', h=begin.current_hierarchy, identifier=identifier_var)
        return DefiniteHierarchyFrozenQuery(self, branch, htype, branch.current_hierarchy, [], None)

    def paths2factor(self, factor_name: str,  plural: bool,
                     start: Type[Hierarchy] = None) -> Tuple[Dict[Type[Hierarchy], Set[TraversalPath]], Type[Hierarchy], bool]:
        """
        returns a dictionary of hierarchy: [path,...] and a shared hierarchy
        """
        factor_name = self.data.singular_name(factor_name)
        pathsetdict, base = self.data.find_factor_paths(start, factor_name, plural)
        is_product = factor_name in base.products.keys()
        return pathsetdict, base, is_product


    def paths2hierarchy(self, hierarchy_name, plural,
                        start: Type[Hierarchy] = None) -> Tuple[List[TraversalPath], List[Type[Hierarchy]], Type[Hierarchy], Type[Hierarchy]]:
        """
        Returns:
            list of possible paths
            list of hierarchies those paths end with
            the shared start hierarchy
            the shared end hierarchy
        """
        if start is None:
            end = self.data.singular_hierarchies[self.data.singular_name(hierarchy_name)]
            return [], [end], None, end
        return self.data.find_hierarchy_paths(start, self.data.singular_hierarchies[self.data.singular_name(hierarchy_name)], plural)
