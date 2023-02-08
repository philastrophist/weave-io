# Stacking, aligning, and joining

## Stacking tables together
You can stack related tables together, which is much like an SQL merge operation.

For example, let's construct two tables of `run.id` and number of spectra for every red run and every blue run.

```python
exps = data.exposures
reds = exps.runs[exps.runs.colour == 'red']
blues = exps.runs[exps.runs.colour == 'blue']
red_table = reds[['id', count(reds.l1single_spectra, wrt=reds)]]
blue_table = blues[['id', count(blues.l1single_spectra, wrt=blues)]]
```

We can execute each of these tables to see their results:
```python
>>> red_table()
   id   count
------- -----
1002213   960
1002215   960
     ...
>>> blue_table()
   id   count
------- -----
1002214   960
1002216   960
     ...
```

Now if we wanted a table where each row refers to an exposure and contains 4 columns: red id, red count, blue id, and blue count, 
we would *stack* the `red_table` and `blue_table` like so:

```python
t = exps[[red_table, blue_table]]
```

!!! failure
    this next bit doesn't work

```python
>>> t()
id0      count0   id1     count1
---      ------   ---     ------
1002213   960     1002214       960
1002215   960     1002216       960
                ...
```
Or you could give more helpful names by specifying a prefix in a dictionary:

```python
t = exps[[{'red_': red_table, 'blue_': blue_table}]]
```

```python
>>> t()
red_id   red_count   blue_id     blue_count
------   ---------   -------     ----------
1002213   960     1002214       960
1002215   960     1002216       960
                ...
```

## Aligning queries
In the previous example, there was a bit of duplication of effort when designing the tables for the red and blue arms.
You can avoid this by using `align`:

```python
red_and_blue = align(reds, blues) 
```
After using `align` you can then continue to construct your query just as you would have done except now you don't have to do it for the red and blue arms individually:
```python
aligned_t = red_and_blue[['id', count(red_and_blue.l1single_spectra, wrt=red_and_blue)]]
```
If we execute this query we get the same result as above:
```python
>>> aligned_t()
>>> t()
id0      count0   id1     count1
---      ------   ---     ------
1002213   960     1002214       960
1002215   960     1002216       960
                ...
```

You can also get the helpful names back by specifying the prefix in the `align` function:
```python
red_and_blue = align(red=reds, blue=blues)
```

After aligning two or more query objects, you can perform any operation on the aligned query that you would perform on a normal query.
Behind the scenes, your operation is performed for each query in the aligned query. 

!!! failure
    You may only `align` queries that share a common ancestor. In the above case, that common ancestor was `exps (=data.exposures)`.
    If you try to `align` on queries which are not related in this way, the query will fail:
    ```python
    >>> align(data.runs, data.exposures)
    ValueError: All queries must be from the same parent Data object
    ```

## Joining
Sometimes you will want to query the database using some data that you have locally available. 
For instance, you may have a list of redshifts fit by your own method, and you want to find the spectra for which WEAVE and your method disagree.

Joining allows you to upload this personal table to the database temporarily and only accessible to you.
**The added data is only available to the uploader whilst the query is being executed and is never committed to the database.**

Joining works by finding an object in the database that you can match against.
You can then traverse and query as normal with each row in your joined table being treated as singular with respect to that object.

For example, we can join a table where each row corresponds to a weave_target cname
```python
table, weave_targets = join(table, 'cname', data.weave_targets.cname)
```
`weave_targets` is the subset of `data.weave_targets` that is matched to the table
`table` is the entire table


We can implement the above redshift example as detailed below:
```python
my_redshifts = Table.read('my/fancy/redshifts.fit')  # has columns: weave_cname, l2_filename, my_redshift   
table, l2 = join(my_redshifts, ['weave_cname', 'l2_filename'], data.l2s[['cname', 'fname']], 'table')

```