# Identifying, filtering and aggregating

## Finding objects by their unique identifiers
If an object defines an idname in its class definition, you can find it in a query by using square brackets:

```python
data.obs[2345]
```
This will return a query pointed at a single ob from which you can continue the query:

```python
data.obs[2345].l1single_spectra...
```

You can also access more than one object by listing them in the square brackets:

```python
data.obs[2345, 3443, ...]
```

!!! failure
    You can only filter by id if the object is uniquely defined (i.e. specifies an `idname` in its definition).
    ```python
    >>> data.l1single_spectra[0]
    ValueError: L1SingleSpectrum is not globally uniquely defined
    ```

!!! warning
    Cardinality still applies when filtering by id!
    ```python
    >>> all_spectra = data.l1single_spectra
    >>> an_ob = all_spectra.obs[1234]
    >>> an_ob.id()
    [1234, 1234, 1234, 1234, 1234, 1234, ...]
    >>> data.obs[1234].id()
    1234
    >>> all_spectra.ob.id()
    8452  # doesnt have to be the same here 
    ```

Once you've traversed to an attribute, you can filter and aggregate.

## Filtering
You can reduce the cardinality of a query structure by filtering either by id as above, or by a boolean mask constructed as part of the query.

You can create the mask by comparing either multiple queries or a query with a python value:

#### Using a python value: Filter spectra to those with SNR greater than 10 
```python
data.l1single_spectra[data.l1single_spectra.snr > 10]
```

#### Using queries: Filter spectra to those with SNR greater the average SNR of all spectra in the database
```python
data.l1single_spectra[data.l1single_spectra.snr > mean(data.l1single_spectra.snr)]
```

!!! failure
    You cannot filter an object by a boolean mask which does not have said object in its history.
    All of these will fail: 
    ```python
    >>> data.l1single_spectra[np.array([1,2,3]) > 0]
    TypeError: Cannot filter a query by a non-query
    >>> data.l1single_spectra[data.obs.mjd < today()]
    CardinalityError: Cannot filter <ObjectQuery L1singleSpectrum> relative to <ObjectQuery OB> since it is not a parent.
    ```

## Aggregation
A query is a chain of objects and attributes (maybe a combination of more than one chain if you've compared or performed arithmetic).
You can aggregate the end of this chain structure relative to some other point in the chain.
This is similar to groupby-apply in SQL except that everything is already implicitly grouped in `weaveio`.
`weaveio` defines the aggregation function: `sum, max, min, mean, std, count, any, all, exists`.

For example:

Count the number of objects in the database:

```python
>>> count(data.obs)()
30
```

Count the number of objects **with respect to another**:

```python
>>> result = count(data.obs.l1single_spectra, wrt=data.obs)()
>>> result
[902, 945, 854, ...]
>>> len(result)  # this is the length of the python list not the count of the query
30
```

All aggregation functions work in the same way: define a long query chain and use `wrt=` to select the point in the chain you wish to aggregate to.
The resulting structure will have the same cardinality as the query given in the `wrt` argument.
If `wrt` is not given, `weaveio` will assume you mean `wrt=database` and will always return a singular result.


!!! failure
    You may only aggregate a query relative to a point in its history.
    ```python
    >>> count(data.obs.l1single_spectra, wrt=data.runs)
    CardinalityError: Cannot aggregate <ObjectQuery OB-L1singleSpectrum> relative to <ObjectQuery Run> since it is not a parent.
    ```

!!! failure
    Moreover, you can only currently do aggregate to a point that was explicitly traversed before.
    The chain `data.obs.l1single_spectra` implicitly goes through `runs` because each `L1SingleSpectrum` originates from one run, 
    which originates from one OB.
    However, the following fails because `run` was not mentioned explicitly in the query before:
    ```python
    >>> count(data.obs.l1single_spectra, wrt=data.obs.runs)
    CardinalityError: Cannot aggregate <ObjectQuery OB-L1singleSpectrum> relative to <ObjectQuery OB-Run> since it is not a parent.
    ```
