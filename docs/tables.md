# Designing result tables

## Using strings as columns
Tables are build on top of objects in weaveio. 
To design a table in your query, you use the double square brackets:
```python
table = spectra[['nspec', 'snr']]
```
In this example, `spectra` is the **_inciting object_** and `'nspec'` and `'snr'` are the attribute names with which to make columns.

When the above query is executed, an astropy table is returned.
```python
>>> table()
nspec      snr
 int      float
-----     -----
  0       2.582
  1       0.124
```
Note that only the attributes that are requested are included in the table output.
Any other attributes including unique ids will not be returned unless specified.
And just like before, you may not return objects themselves only their attributes:
!!! failure
    ```python
    >>> spectra[['nspec', 'snr', 'survey_target']]
    SyntaxError: SurveyTarget cannot be returned/identified since it doesn't define any unique idname. 
                 If you want to return all singular data for SurveyTarget use ...['*']
    ```

## Using other queries as columns
Any attribute name in the square brackets is assumed to relate to the proceeding object.
Indeed, you can be explicit: 
```py
spectra[[spectra.nspec, spectra.snr]]
```
This is an example of using a query (`spectra.nspec` and `spectra.snr`) as a column.

You can specify any query as a column **as long as it has the _inciting object_ as a parent**.
This means that a query inside the square bracket **must** reference the inciting object at a point in its history.

To illustrate this point, consider the following example where we get a list of `spectra` for each `exposure` in the database.
```python
specs = data.exposures.l1single_spectra
significant = specs.snr > 5
```

!!! success
    ```python
    >>> specs[['nspec', 'snr']]()
    nspec        snr        
    ----- ------------------
      247 111.34782409667969
      177 47.183815002441406
    ```
!!! success 
    ```python
    >>> specs[['nspec', specs.snr]]()
    nspec        snr        
    ----- ------------------
      247 111.34782409667969
      177 47.183815002441406
    ```
!!! success 
    ```python
    >>> specs[['nspec', specs.snr>5]]()
    nspec  >  
    ----- ----
      247 True
      177 True
    ```
!!! success 
    ```python
    >>> specs[['nspec', significant]]()
    nspec  >  
    ----- ----
      247 True
      177 True
    ```
!!! success  
    ```python
    >>> specs[['nspec', specs.exposures.mjd]]()
    nspec     mjd     
    ----- ------------
      247 57639.865255
      177 57639.865255
    ```
!!! warning
    ```python
    >>> specs[['nspec', data.exposures.mjd]]()
    nspec     mjd[523]
    ----- ------------
      247 [57639.865255, ...]
      248 [57639.865255, ...]
    ```
    This query actually returns the same list of all exposure mjd values in the database for each spectrum - a huge duplication of effort.


## Renaming columns

Sometimes the column name that is assigned to a column is less than helpful:
```python
>>> specs[['nspec', specs.snr > 3, specs.snr > 4]]()
nspec  >0   >1 
----- ---- ----
  247 True True
  177 True True
```

If we want to override the given name we can use a dictionary:
```python
>>> specs[['nspec', {'snr > 3': specs.snr > 3, 'snr > 4': specs.snr > 4}]]()
nspec snr > 3 snr > 4
----- ------- -------
  247    True    True
  177    True    True

```

