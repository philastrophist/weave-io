# WEAVE-LOFAR/WEAVE-IO

## Introduction
WEAVE-IO is a query builder syntax that allows you to construct complex requests for data beyond simple joins without the need for any SQL knowledge. 
It was developed with the intent to flatten the learning curve required to access spectra and spectral data-products in the WEAVE Operational Repository (OR) output files.
It is low-level and will not attempt to process any data coming from WEAVE or LOFAR since that remains in the remit of the OR or the individual scientist.

A typical question for the WEAVE data structure would be:

**"Fetch does the sky model look like in WEAVE-LOFAR stacked red-arm spectra that supposedly have best fit redshifts of z>8?"**

The workflow for answering this question would be something like:

0. Download all WL `stack`ed L1 and L2 data products (`stacked_{runid}.fit` and `stacked_{runid}.APS.fit`) 
1. Read the `CLASS_TABLE` hdu in all L2 `stacked_{runid}.APS.fit` files
1. Filter to `fibreid` and `runid` of those fibres which have fit redshifts > 8
1. Find L1 `stack` spectra from those runids: `stacked_{runid}.fit` 
1. Filter to the filenames which have `CAMERA=WEAVERED` in their headers (reading the headers)
1. Filter to the filenames which have `fibreid`s identified above (reading `FIBTABLE` hdu)
1. Index fibreids to get the position of the spectra of interest in the hdu
1. Read the `DATA_RED` hdu and index to the spectra of interest
1. Read the `DATA_RED_NOSS` hdu and index to the spectra of interest
1. `SKY = DATA_RED_NOSS - DATA_RED`

Everyone will end up writing their own IO scripts that will necessarily become more complicated and resource-intensive to run, the more
complex they are required to be...

WEAVE-IO unifies query writing and data retrieval. The equivalent WEAVE-IO of our example question above is:
```python
from weaveio import Data, Address
data = Data(version='opr3')  # instantiate the database connection
query = data.stacked.sky_spectra[Address(camera='red', survey='WL') & data.stacked.l2.z > 0.8]  # build the query
array = query()  # execute the query and fetch the relevant arrays from the fits files
```
WEAVE-IO only fetches data, it does not perform analysis, propagation, or reduction of any kind. 
The main consequence of this restriction is that you will have to join the two arms of the spectrograph yourself.
However, as should be obvious from the above code snippet, it is much easier to find the data you are looking for.

You must run your queries using the database found on the Hertfordshire High Performance Cluster (UHHPC), where the WEAVE data
are downloaded. 

## Object orientation and DB structure
The WEAVE-IO database is hierarchical and object-oriented. 
What this means is that each object that has some meaning in the world of WEAVE is represented in the database. 
You can access child objects that belong to parent objects (e.g. a `run` is a child of an `exposure`) by using 
Python's dot-attribute syntax. 

* An `exposure` contains `runs`: `exposure.runs`
* An observing block `OB` contains  


For example, each Observing Block (`ob`)
```python
>>> len(data.obs)
43  # there have 43 obs referenced by the database
>>> ob = data.obs[1125]  # using obid = 1125
```
is composed of 1 or more `exposures` 
```python
>>> len(ob.exposures)
3  # this ob had 3 separate exposures 
>>> len(ob.exposures.arms)
2  # this ob uses both arms
>>> exposure = ob.exposures[57659.145637]  # exposure modified julian date = 57659.145637
```
and each exposure is composed of 1 or 2 `runs` associated with each arm of the spectrograph.
```python
>>> run = exposures.runs['red']  # arm = 'red'
# or
>>> run = exposures.runs["1002814"]  # runid = '1002814'
```
It is the `runs` which have associated spectra and these are contained in the WEAVE file `{raw/single}_{runid}.fit`.

Each spectrum in the `single` file of a given `run` will be associated with a `fibre` which can be assigned to a `target`, sky, or not allocated.  
If we want to fetch all the spectra observed by a given `run`, we only need to say 
```python
run.spectra()
```

The [quickstart](/quickstart) guide will outline the basic concepts a bit further. 


WEAVE data products come in two forms:

1. **L1: Spectra products**
    * Raw
    * Single
    * Stack
    * Superstack