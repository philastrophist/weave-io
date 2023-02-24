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

## 
For example:
```python
database.cats.owner.age > database.cats.age
```
This query is a boolean structure the same shape as `database.cats` (and also `database.cats.owner` since a cat only has one owner).
