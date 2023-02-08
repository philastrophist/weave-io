# Writing to the database
Uploading data requires write permission which is not granted by default to avoid obvious problems.
The following can only be done if you have write permission.

The procedure for uploading a single node is:

    from weaveio.opr3 import Data, Survey, Subprogramme
    data = Data(username, password)
    with data:
        survey = Survey(surveyname='WL-deep')
        subprogramme = Subprogramme(targprog='WL1', surveys=[survey])
        
So all it entails is instantiating each of the objects as you would intuitively expect.
When you leave the `with data` indented context, you push the changes as a CYPHER query to the database.

To read an entire file into the database, you call its `.read` method. 

```diff
with data:
+  l1singlefile = L1SingleFile.read("path-to-file") 
-  l1singlefile = L1SingleFile("path-to-file") 
``` 
If you instantiate the file directly, you will fail because the L1SingleFile instance requires an input spectra as well.

## Uniqueness
There are three ways a node can be unique:

1. **Global**: Some nodes are globally unique to the database and are instantiated with an ID. E.g. `Run(runid=121344)` or `RawFile(fname='L1/r121344.fits')`. This is enforced by the a Neo4j uniqueness constraint.
2. **Relative**: Most other nodes are unique only relative to their inputs. For example a `OB` is unique only to its parent `FibreTargets` and so doesn't have an id. This is not enforced by Neo4j but will be enforced by the WEAVE-IO write process, which uses a tailored Neo4j procedure.
3. **Not unique**: Any other node which is not one of the above will not be unique. There are no non-unique nodes in WEAVE-IO.

Uniqueness allows us to keep adding files and data without worrying about duplicating things and it automatically ensures that the data are associated.

# How the writing process works

When a file is read, all the hierarchies that are contained within it are merged into the graph from top to bottom (not as you may expected, from the bottom to top). 
So when an `L1StackFile` is written to the db using `L1StackFile.read(fname)`, what is actually happening is something like the following (for one spectrum in that file):

```python
# L1StackFile.read(fname)  -  spectra stacked from the same OB but different exposures
with data:
    fibinfo = CypherData('fibinfo', fibinfo)
    checksums = CypherData('checksums', checksums)
    runids = CypherData('runids', runids)
    with unwind(runid) as runid:
        run = Run(runid=runid, exposure=None)  # match runs based only on runid
        raw = RawSpectrum(run=run, casu=casu, simulator=simulator, system=system)  #  merge spectrum
    raws = collect(raw)
    with unwind(checksums, fibinfo) as checksum, fibinfo:
        fibretarget = Fibretarget(...)  # make the hierarchy as in the raw file (this will just be a match if they are all the same)
        with unwind(runid) as runid:
            run = Run(runid=runid, exposure=None)  # match runs based only on runid
            raw = RawSpectrum(run=run, casu=casu, simulator=simulator, system=system)  #  merge spectrum 
            single = L1SingleSpectrum(raw=raw, casu=casu, fibretarget=fibretarget, checksum=None)  #  match the spectrum from the raw
        singles, runs = collect(single, run)
        stack = L1StackSpectrum(checksum=checksum, l1singlespectra=singles)  # still under fibretarget context
    stacks = collect(stack)
    L1StackFile(fname=fname, l1stackspectra=stacks)
    
        
```