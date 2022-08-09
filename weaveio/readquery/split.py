from typing import Union

from . import BaseQuery, AttributeQuery, get_neo4j_id
from .objects import ObjectQuery


class SplitQuery:
    def __init__(self, parent: BaseQuery, groupby: AttributeQuery):
        self.parent = parent
        self.groupby = groupby
        self._ids = None
        self._group_query = None

    @property
    def groupby_ids(self):
        if self._ids is None:
            self._ids = self.groupby(limit=None, distinct=True)
        return self._ids

    @property
    def group_query(self):
        if self._group_query is None:
            group = self.parent._G.get_variable_name('group')
            self._group_query = self.parent[self.groupby == group]
        return self._group_query

    def __iter__(self):
        pass





    # run groupby and return distinct
    # add filter


def split(query: BaseQuery, groupby: Union[AttributeQuery, str, ObjectQuery] = None):
    """
    Splits the given query into unique parts based on the given groupby attribute.
    Each part is treated like a new query and is executed independently.
    This is the weaveio equivalent of an SQL GROUP BY statement.
    Except that you must perform a split *before* the main query.
    This is unlike the SQL GROUP BY statement which is performed *after* the main query.

    Any action you perform on the query after the split is performed on each part automatically.
    Queries that result in duplicates i.e. `data.runs.l1single_spectra.runs` will not be changed, duplications will persist.

    :param query: The query to split.
    :param groupby: The name of the groupby attribute.
    :return: A SplitQuery object containing many queries that were split from the given query.

    Example:
    >>> obs = data.obs
    >>> groups = split(obs, 'id')
    >>> result = groups.l1single_spectra[['flux', 'wvl']]
    >>> result
    <SplitQuery(TableQuery(L1SingleSpectrum-L1SingleSpectrum))>
    for name, group in groups:
    >>>     print(name)
    3901
    >>>     print(group)
    <TableQuery(L1SingleSpectrum-L1SingleSpectrum)>
    >>> print(group())
       flux [14401]     wvl [14401]
    ----------------- ----------------
            0.0 .. 0.0 4730.0 .. 5450.0
            0.0 .. 0.0 4730.0 .. 5450.0
       1.203763 .. 0.0 4730.0 .. 5450.0
     0.62065125 .. 0.0 4730.0 .. 5450.0
      347.95587 .. 0.0 4730.0 .. 5450.0
            0.0 .. 0.0 4730.0 .. 5450.0
            0.0 .. 0.0 4730.0 .. 5450.0
            0.0 .. 0.0 4730.0 .. 5450.0
            0.0 .. 0.0 4730.0 .. 5450.0
            0.0 .. 0.0 4730.0 .. 5450.0
    """
    if groupby is None and isinstance(query, ObjectQuery):
        groupby = query._get_default_attr()
    elif isinstance(groupby, str):
        groupby = query[groupby]
    elif isinstance(groupby, ObjectQuery):
        groupby = groupby._get_default_attr()
    elif not isinstance(groupby, AttributeQuery):
        raise TypeError('groupby must be an AttributeQuery, str, or ObjectQuery')
    group_id = query._G.add_groupby(groupby)
    group_eq = groupby._perform_arithmetic(f'{{0}} = {group_id}', '=', group_id, returns_dtype='boolean')
    grouped_query = query[group_eq]
    return grouped_query