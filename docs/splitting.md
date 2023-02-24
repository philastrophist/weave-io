# Splitting queries

`weaveio` is more than just a query language, it is a system also to perform actions on its own queries.

The `split` function creates a separate *sub*-query for each row that would be returned.

For example, `split(data.obs)` will create around 30 sub-queries (one for each OB in the database).
You can continue the query as you normally would but this time you are acting on each sub-query.
It's like a for loop over all OBs.

You probably want to iterate over a split query rather than call it directly because that defeats the point of a split query.
Iteration works slightly differently as it yields and index (in this case the OB id) as well as the sub-query.

```python
for index, subquery in split(data.obs):
```

A full example is given here where you want to get all the L1SingleSpectra per ob, in batches per ob:
```python
from weaveio import *
data = Data()

obs = split(data.obs)  # mark the fact that you want have one table per OB thereby "splitting" the query in to multiple queries
singles = obs.l1single_spectra
query =  singles[['flux', 'ivar']]

for index, ob_query in query:
    print(f"stacks and singles for OB #{index}:")
    print(ob_query())
```