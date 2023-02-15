from astropy.table import Table
from typing import Tuple, Union

from weaveio.readquery.objects import ObjectQuery, AttributeQuery, TableVariableQuery


def join(table: Table, index_column: str,
         object_query: ObjectQuery, join_query: Union[AttributeQuery, str] = None,
         join_on='table') -> Tuple[TableVariableQuery, ObjectQuery]:
    """
    Add each row of an astropy table to a query object.
    The columns of each row will be accessible to all subsequent queries that fork from `object_query`.
    This would traditionally be called a "left join" since only all rows in the table will be returned
    even if there is no match to a weaveio object (null).
    Furthermore, weaveio objects that are not matched will not be returned.

    :param table: Astropy Table containing all data that you want to be accessible to the query
    :param index_column: Column in the table that will be used to join the table to the query object
    :param object_query: Query object that will be joined to the table
    :param join_query: Query object that will be used to join the table to the query object.
                       This can be any attribute query as long as it is singular wrt to `object_query`
                       Leave as None if you want to join the table to the query object using the index_column name as the attribute.
    :param join_on: Type of join to use. "table" is per row in table, "object" is per object in query.
    :return: A tuple of the table variable query and the query object

    Examples:
        Join a table where each row corresponds to a weave_target cname
        >>> table, weave_targets = join(table, 'cname', data.weave_targets.cname)
        `weave_targets` is the subset of `data.weave_targets` that is matched to the table
        `table` is the entire table

        Join a table where each row corresponds to a weave_target cname but is matched to a specific spectrum
        >>> spectra = data.runs[1234].l1single_spectra  # get all spectra belonging to this run
        >>> table, query = join(table, 'cname', spectra, spectra.cname)
        `query` is the subset of `data.runs[1234].l1single_spectra` that is matched to the table
        `table` is the entire table

        Join a table where each row corresponds to an ob and matched based on the first mjd of that OB
        >>> table, query = join(table, 'mjd', data.obs, min(data.obs.exposures.mjd, wrt=data.obs))
        `query` is the subset of `data.obs` that is matched to the table
        `table` is the entire table
    """
    if join_query is None:
        join_query = index_column
    if isinstance(join_query, str):
        join_query = object_query[join_query]
    if not isinstance(join_query, AttributeQuery):
        raise TypeError(f"join_query must be an AttributeQuery or str, not {type(join_query)}")
    G = object_query._G
    param = G.add_parameter(table)
    # filter the input table to only include rows that match the join_query
    applied, applied_var = G.add_apply_to_list(object_query._node, param, '{0}', f'{{0}}.`{index_column}` = {{1}}',
                                               join_query._node, put_null_in_empty=True, parameters=[param])
    applied = G.fold_to_cardinal(applied)
    applied_var = G.G.nodes[applied]['variables'][0]
    row = G.add_unwind_parameter(object_query._node, applied_var, applied)

    ref = G.add_previous_reference(row, object_query._node)
    if join_on == 'table':
        not_null = G.add_scalar_operation(row, '{0} is not null', 'not-null')[0]
        obj_node = G.add_filter(ref, not_null, False)
    elif join_on == 'object':
        obj_node = ref
    else:
        raise ValueError(f"join_on must be either the input table `table` or the input object `object_query`, not {join_on}")
    single = False
    return TableVariableQuery._spawn(object_query, row, '_row', single=True, table=table),\
           ObjectQuery._spawn(object_query, obj_node, object_query._obj, single=single)