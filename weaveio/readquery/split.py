from typing import Union

from . import BaseQuery, AttributeQuery
from .objects import ObjectQuery


def split(query: BaseQuery, splitby: Union[AttributeQuery, str, ObjectQuery] = None):
    """
    Splits the given query into unique parts based on the given groupby attribute.
    Each part is treated like a new query and is executed independently.
    This is the weaveio equivalent of an SQL GROUP BY statement.
    Except that you must perform a split *before* the main query.
    This is unlike the SQL GROUP BY statement which is performed *after* the main query.

    Any action you perform on the query after the split is performed on each part automatically.
    Queries that result in duplicates i.e. `data.runs.l1single_spectra.runs` will not be changed, duplications will persist.

    :param query: The query to split.
    :param splitby: The name of the groupby attribute.
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
    if splitby is None and isinstance(query, ObjectQuery):
        try:
            splitby = query._get_default_attr()
        except SyntaxError:
            splitby = query._get_neo4j_id()
    elif isinstance(splitby, str):
        splitby = query[splitby]
    elif isinstance(splitby, ObjectQuery):
        splitby = splitby._get_default_attr()
    elif not isinstance(splitby, AttributeQuery):
        raise TypeError('groupby must be an AttributeQuery, str, or ObjectQuery')
    group_id = query._G.add_groupby(splitby)
    group_eq = splitby._perform_arithmetic(f'{{0}} = {group_id}', '=', group_id, returns_dtype='boolean', parameters=[group_id])
    grouped_query = query._filter_by_mask(group_eq, single=True, split_node=True)
    return grouped_query