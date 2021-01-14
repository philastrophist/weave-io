# Writing to the database
Uploading data requires write permission which is not granted by default to avoid obvious problems.
The following can only be done if you have write permission.

![relationships](relations.png)

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
2. **Relative**: Most other nodes are unique only relative to their inputs. For example a `FibreSet` is unique only to its input `FibreTarget`s and so doesn't have an id. This is not enforced by Neo4j but will be enforced by the WEAVE-IO write process, which uses a tailored Neo4j procedure.
3. **Not unique**: Any other node which is not one of the above will not be unique. There are no non-unique nodes in WEAVE-IO.

Uniqueness allows us to keep adding files and data without worrying about duplicating things and it automatically ensures that the data are associated.

## Versioning
There are three types of versioning data that WEAVE-IO deals with in some way.

1. **Schema changes**: If WEAVE output files change so dramatically that the weaveio schema no longer maps to it, then we would need to rebuild the database. 
1. **Mismatched ids**: A hierarchy purports to have an ID but its data are different in some to the one which already exists (mismatched). 
Due to the uniqueness constraint set in the database, these writes will be rejected. 
If this happens, then something has gone wrong with the ids WEAVE assigns to runs, obs, and files. Therefore, this should never happen.
1. **Reprocessing**: If a hierarchy is produced more than once with a different process (i.e. a raw spectrum is reprocessed using a different set of CASU parameters), then there would be duplication. 
However, WEAVEIO requires that you specify which version of CASU, APS etc was used and also requires the checksum for each spectrum/file. 
This will stop overwriting but, by itself, would complicate the graph and stop queries like `data.runs[12345].raw` from being valid (if there is now more than one version of raw). 
So WEAVEIO creates a `version` number as a node property.
 WEAVE-IO will then choose the most recent version when traversing the graph. 
 Only when you specify CASU or APS version explicitly, will WEAVE-IO be forced to traverse a less up-to-date version path.

### Version paths
Only some relationships are versioned. For example, a RawSpectrum is only supposed to have one L1SingleSpectrum.
But if CASU/someone else is run more than once, then there will be more than one. 
The L1SingleSpectrum is added to the graph as normal (if it is different, but then Cypher query will check for other versions and increment this L1SingleSpectrum<--RawSpectrum path version appropriately.
The actual information about the version of the data will be stored in nodes, either itself, or parent nodes. 
This is what the `CASU` and `system` nodes are doing in the graph above.

For each binary data product produced by CASU, versioning is done with  

1. A checksum (stored in the node)
1. Any author information (stored as parent nodes since they can author multiple products)




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