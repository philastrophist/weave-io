from weaveio.basequery.common import FrozenQuery


class FactorFrozenQuery(FrozenQuery):
    pass


class SingleFactorFrozenQuery(FactorFrozenQuery):
    """A single factor of a single hierarchy instance"""


class ColumnFactorFrozenQuery(FactorFrozenQuery):
    """A list of the same factor values for different hierarchy instances"""


class RowFactorFrozenQuery(FactorFrozenQuery):
    """A list of different factors for one hierarchy"""

    def __init__(self, handler, query, return_keys, parent: 'FrozenQuery' = None):
        super().__init__(handler, query, parent)
        self.return_keys = return_keys

    def __getattr__(self, item):
        if self.handler.is_plural_factor(item):
            raise KeyError(f"{self} can only be indexed by singular names")
        if not self.handler.is_singular_factor(item):
            raise KeyError(f"Plural factors {item} is not a known factor")
        return self.handler._get_plural_factor(self.parent, item)


class TableFactorFrozenQuery(RowFactorFrozenQuery):
    """
    A matrix of different factors against different hierarchy instances
    This is only possible if the hierarchies each have only one of the factors
    """
