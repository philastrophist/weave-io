# Quickstart

## Object orientation


## WEAVE-IO hierarchical structure


## Plurality
WEAVE-IO is plural-aware, meaning that it requires that you explicitly use the plural name of an object when
there is more than one instance being referenced in the schema.
This makes the query more natural to read and its easier catch problems. 
WEAVE-IO will reject any query that is not phrased correctly. 

For instance,

* `exposure.runs` is correct because an `exposure` can have more than 1 `run` (and usually does). 
It is still correct even if the database only contains 1 `run` because an `exposure` *is able* to have more than one `run`.
```python
>>> exposure.runs()
[Run(runid=1002814), Run(runid=1002815)]
```

* `exposure.run` is not correct because an `exposure` can contain more than one `run`.
```python
>>> exposure.run()
raise AmbiguousPathError
```

* `exposures.runs` is correct.
```python
>>> exposures.runs()
[Run(runid=1002812), Run(runid=1002813), Run(runid=1002814), Run(runid=1002815), ...]
```

* `run.exposure` is correct because a `run` can only ever have one `exposure`.
```python
>>> run.exposure()
Exposure(expmjd=57659.145637)
```

* `run.exposures` is correct because you can return a list.
```python
>>> run.exposures()
[Exposure(expmjd=57659.145637)]
```

* `runs.exposures` is correct and will return a list of exposures.
```python
>>> runs.exposures()
[Exposure(expmjd=57659.145637), Exposure(expmjd=57658.128364), ...]
```

Executing a singular will return one object, executing a plural will return a list.


## 