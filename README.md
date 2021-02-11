![PyPI](https://img.shields.io/pypi/v/weaveio)

# Tutorial

## Basics

## WEAVE objects

* An OB holds all the information that purtains to making an observation: the targets, the conditions, the instrument configuration. You can locate specific OBs with their obid `data.obs[obid]`

* An Exposure is one integration of both arms of the spectrograph. You can locate Exposures like so:  `data.exposures[mjd]`

* A Run is a weave term for an exposure taken in one arm of the spectrograph (so there are always 2 Runs per Exposure). You can locate runs using their runid `data.runs[runid]` or their colour relative to the exposure `data.exposures[mjd].runs['red']`

* A `spectrum` in weaveio refers to a single spectrum of a single target (not the block of `spectr*a*`)

* An L1 product refers to the spectra recorded in an L1File. All L1 data is separated by camera colour (red and blue). 
    * A single spectrum is the processed spectrum from the raw data
    * A stacked spectrum is the spectrum resulting from stacking two or more single spectra in a single ob
    * A superstacked spectrum results from stacking *between* OBs but with the same instrument configuration
    * A supertarget spectrum results from stacking every single spectrum of a single WeaveTarget cname.

* There are three types of `Target`
    1. `WeaveTarget` is the unified target based on ra, dec. They have a unique CNAME
    2. `SurveyTarget` is a target specified by a survey in a surveycatalogue (they reference a single WeaveTarget). These are unique to a catalogue.
    3. `FibreTarget` is a result of assigning a spectrograph `Fibre` to a `SurveyTarget`. These are unique to an OBSpec.
    

### What is an attribute? What is an object? What is a product?

* weave.io is an object orientated query language using a neo4j database hosted at lofar.herts.ac.uk. 
* weave.io stores 4 (+1) types of object:
    1. File - A reference to a physical fits file on the herts system (it also stores references to individual fits.HDUs and their headers as separate objects, but the user doesn't need to know of their existence to use weave.io)
    2. Object - A object that references a concept in the WEAVE universe that has attributes (an OB object has an obid attribute, an exposure object has an attribute expmjd
    3. Attribute - A piece of data that belongs to some object
    4. Product - A special type of attribute which references binary data not stored in the database itself (e.g. spectrum flux)

## If confused, ignore...

### object/attribute 
weave.io uses Python syntax to traverse a hierarchy of objects and their attributes. It is important to note that one object can be an attribute of another object (e.g. a run is an attribute of an OB). 

You can stitch together these objects to form a hierarchy of objects:

run <-- exposure <-- ob <--obspec

Every OB is a parent of multiple Exposures which in turn are parents exactly 2 runs each (one red, one blue).

Because of this chain of parentage/relation, every object has access to all attributes where there is a chain, as if they were its own attributes.

![image.png](attachment:image.png)

### Traversal syntax 

1. You can request any directly owned attribute of an object 
    * An OB has an obid: `ob.obid`
    * An obstemp has a maxseeing `obstemp.maxseeing` 
    
1. You can request any attribute of objects that are further away in the hierarchy as if it were its own. This is useful because a priori you wouldn't be expected to know where any particular piece of data is stored, just that it exists.
    * `run.maxseeing` is identical to `run.exposure.ob.obspec.obstemp.maxseeing`

1. Traversal works in any direction
    * You can go down a hierarchy: `exposure.run.rawspectrum`
    * You can go up as well: `rawspectrum.run.exposure`

1. Traversal can be implicit like with the indirectly accessed attributes above
    * You can skip out stages: `run.obspec` is identical to `run.ob.obspec`
    
1. Traversal requires you to be aware of plurality/multiplicity (neo4j calls this cardinality):
    * A run only ever has a single ob, so you query it using a singular name: `run.ob`
    * But an ob will typically have multiple runs, so you must use plural names: `ob.runs`
    * weave.io is aware of plurality of the whole hierarchy, so it will shout at you if you are obviously wrong: `ob.run` will fail before you even execute the query.

1. Traversal name plurality is relative
    * A run has a single ob, which in turn has multiple runs: `run.ob.runs` will return all runs of the ob (including the one that was explicitly referenced at the start. 
    * `ob.runs.weavetarget.obs` can be read as "**For each** of the runs, get its weavetarget, and then **for each** weavetarget get all OBs which assign a fibre to that target.

1. Traversal using dot syntax always increases/maintains the total number of rows returned at the end
    * A consequence of traversal is the building up of rows. This is useful to get properly aligned labels/indexes for things.
    * `ob.runs.ob.runs.ob.runs` will return not simply return the runs of this ob, but rather a huge duplicated list because each time you use the '.' syntax, we are saying **"for each"**


### Identifiers

1. You can request a specific object if you know its id 
    * `one_ob = data.obs[obid]` 
    * `a_list_of_obs = data.obs[[obid1, obid2, obid3]]` 
    * Plurality still applies here: `data.weavetargets.obs[obid]` will return one ob **for each** weavetarget
        * `data.obs[obid].weavetargets` returns all weavetargets for this particular ob
        * `data.weavetargets.obs[obid]` returns the ob with obid for each weavetarget (sometimes will be None)

1. Relative identifiers
    * `exposure.runs['red']` is functionally the same as `exposure.runs[runid]` (where runid is the red one)

### Exploration

You can use the `explain(obj)` function to see what information is available for a given object:

    >>> explain(data.obs)
    ===================ob===================
     An OB is an "observing block" which is essentially a realisation of
    an OBSpec. Many OBs can share the same xml OBSpec which describes how
    to do the observations.
    
    A ob has a unique id called 'obid'
    one ob is linked to:
        - many exposures
        - many l1stackfiles
        - many l1stackspectra
        - many l2stackfiles
        - many l2stacks
        - 1 obspec
    a ob directly owns these attributes:
        - obid
        - obstartmjd
    ========================================
Likewise, use the `explain(attr)` function to see what information is available for an attribute:
    
    >>> explain(data.runs.runid)
    runid is the unique id name of a run

    >>> explain('snr', data=data)
    snrs are owned by multiple different objects (['l1singlespectrum', 'l1stackspectrum', 'l1supertargetspectrum', 'l1superstackspectrum']). 
    They could be entirely different things.
    You will need to specify the parent object for snr when querying.
    snr is an attribute belonging to l1singlespectrum
    snr is an attribute belonging to l1stackspectrum
    snr is an attribute belonging to l1supertargetspectrum
    snr is an attribute belonging to l1superstackspectrum
    ========================================
    
Also, you can use the `explain(obj1, obj2)`function to see how two objects are related:
    
    >>> explain(data.obs, data.runs')
    ===================ob===================
     An OB is an "observing block" which is essentially a realisation of
    an OBSpec. Many OBs can share the same xml OBSpec which describes how
    to do the observations.
    
    A ob has a unique id called 'obid'
    one ob is linked to:
        - many exposures
        - many l1stackfiles
        - many l1stackspectra
        - many l2stackfiles
        - many l2stacks
        - 1 obspec
    a ob directly owns these attributes:
        - obid
        - obstartmjd
    ==================run===================
     A run is one observation of a set of targets for a given
    configuration in a specific arm (red or blue). A run belongs to an
    exposure, which always consists of one or two runs (per arm).
    
    A run has a unique id called 'runid'
    one run is linked to:
        - 1 armconfig
        - 1 exposure
        - 1 observation
    a run directly owns these attributes:
        - runid
    ========================================
    - A ob has many runs
    - A run has only one ob
    ========================================

When you make a mistake in plurality or a typo, weave-io will offer suggestions:
    
    >>> data.singlespectra
    AttributeError: `single_spectra` not understood, did you mean one of:
    1. l1singlespectra
    2. l1singlespectrum
    3. galaxy_spectra. 
    You can learn more about an object or attribute by using `explain(obj/attribute, ...)`

### Running a query
A query finds the locations of all the L1/L2/Raw products that you want. 
It is analagous to an SQL query except that it is written in Python.

* A query is constructed using python like so:

    ```
    from weaveio import Data
    data = Data(username, password)
    
    runs = data.obs[obid].runs
    reds = runs[runs.colour == 'red']
    spectra = reds.l1singlespectra
    ```
    `runs`, `reds`, `spectra` are all queries
* Each part of this query can be run independently, using the parentheses:
    * `runs.runids` is still a query 
    * `runs.runids()` returns a actual list of numbers
    * `reds()` will return a list of Run objects (containing all attributes of a run)
    * `spectra()` will return a spectra object (read from the fits file on herts.ac.uk) which contains the flux, wvls etc

# Examples of use:

# 1. I want to return the number of sky spectra in a given run (runid=1002850)


```python
from weaveio import *
data = Data()
runid = 1002813
nsky = sum(data.runs[runid].targuses == 'S')
print("number of sky targets = {}".format(nsky()))
```

## 1b. I want to see how many sky targets each run has
```python
from weaveio import *
data = Data()
nsky = sum(data.runs.targuses == 'S', wrt=data.runs)  # sum the number of skytargets with respect to their runs
print(nsky())
```


# 2. I want to plot all single sky spectra from last night in the red arm

Currently, it is only possible to filter once in a query so you have to do separate queries for each condition you want and then feed it back in. See below

```python
from weaveio import *
yesterday = 57634

data = Data()
runs = data.runs
is_red = runs.camera == 'red'
is_yesterday = floor(runs.expmjd) == yesterday

# we do 2 separate filters instead of 1 so to not read too much data into memory at once
runs = runs[is_red & is_yesterday]  # filter the runs first
singlespectra = runs.l1singlespectra
is_sky_target = singlespectra.targuse == 'S'  # then filter the spectra per filtered run
chosen = singlespectra[is_sky_target]

table = chosen['wvls', 'flux'](limit=10)

import matplotlib.pyplot as plt
# uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# %matplotlib 
plt.plot(table['wvls'].data.T, table['flux'].data.T)  # the .T means that matplotlib uses the rows as separate lines on the plot
```

# 3. I want to plot the H-alpha flux vs. L2 redshift distribution from all WL or W-QSO targets that were observed  from all OBs observed in the past month. Use the stacked data


```python
from weaveio import * 
data = Data()

obs = data.obs[data.obs.obstartmjd >= 57787]  # pick an OB that started after this date
fibretargets = obs.fibretargets[any(obs.fibretargets.surveys == '/WL|WQSO/')]  # / indicate regex is starting and ending

l2rows = fibretargets.l2stack
table = l2rows['lineflux_ha_6562', 'z']()

import matplotlib.pyplot as plt
# uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# %matplotlib 
plt.scatter(table['lineflux_ha_6562'], table['z'])
```

# 4. I want to identify the WL spectrum with the brightest continuum at 5000AA and plot the spectrum from both red and blue arms, together with the error (variance) spectrum. 


```python
data = Data()
stackedspectra = data.l1stackedspectra  # lots of different stacked spectra from many different OBs
wl_stackedspectra = stackedspectra[any(stackedspectra.surveys == 'WL')]

reds = wl_stackedspectra[wl_stackedspectra.camera == 'red']
blues = wl_stackedspectra[wl_stackedspectra.camera == 'blue']

continuum = []
for red, blue in reds(), blues():  # this loop is offline
    continuum.append(my_special_module.median_flux(red, blue, 4950, 5050))  # do some fancy function you have written
index = np.argmax(continuum)

red = reds[index]()
blue = blues[index]()

import matplotlib.pyplot as plt
# uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# %matplotlib 
plt.plot(red.wvls, red.flux)
plt.plot(blue.wvls, blue.flux)

plt.plot(red.wvls,  1 / red.ivar, label='variance')
plt.plot(blue.wvls, 1 /  blue.ivar, label='variance')
```

# 5. Get the brightest g-band target in an OB and plot some spectra 
a. I would like to identify the brightest (g band) WL spectrum observed in an OB with `obid=1234` (using the g band magnitude in the stacked spectrum). Plot the stack, in both red and blue arms (on same plot)

b. Plot the individual spectra that went into making the stack. 

d. I would like to search for any other OB that contains the same astronomical object. Overplot the single spectra that were observed in those OBs


```python
import matplotlib.pyplot as plt
# uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# % matplotlib 
data = Data()

####### < Part A
ob = data.obs[1234]  # get the ob 

# all L2 data that used stackedspectra.
l2stack = ob.l2stack

# return rows in the L2 dataset that correspond to a lofar target
lofar_l2 = l2stack[any(l2stack.surveys == 'WL')]  # each target can belong to more than one survey
l2row = lofar_l2[lofar_l2.mag_gs == max(lofar_l2.mag_gs)]  # get the one row that corresponds to the brightest lofar target

# now we jump from L2 rows to the stack spectra
brightest_red = l2row.l1stackspectra[l2row.l1stackspectra.camera == 'red']
brightest_blue = l2row.l1stackspectra[l2row.l1stackspectra.camera == 'blue']

# Now plot the actual data
fig, (redax, blueax) = plt.subplots()
redax.plot(brightest_red.wvls(), brightest_red.flux(), 'r-', label='brightest')
blueax.plot(brightest_blue.wvls(), brightest_blue.flux(), 'b-', label='brightest')
####### Part A />


####### < Part B
# now locate the indivdual single spectra that were stacked
red_spectra = brightest_red.l1singlespectra
blue_spectra = brightest_blue.l1singlespectra

# matplotlib allows you to plot multiple lines with 2d arrays
redax.plot(red_spectra.wvls(), red_spectra.flux(), 'r-', alpha=0.4, label='single for brightest')
blueax.plot(blue_spectra.wvls(), blue_spectra.flux(), 'r-', alpha=0.4, label='single for brightest')
####### Part B />


####### < Part C
# Now get all other stacked spectra that were observed for this target, no matter the OB
brightest_target = l2row.weavetarget
other_reds = brightest_target.l1stackspectra[brightest_target.l1stackspectra.camera == 'red']
other_blues = brightest_target.l1stackspectra[brightest_target.l1stackspectra.camera == 'blue']

# overplot the other observations
for wvl, flux in other_reds.wvls(), other_reds.flux():
    redax.plot(wvl, flux, 'k:', alpha=0.2, label='from all obs')
####### Part C />
```

# Demo

## 6. I would like to identify all of the sky spectra in an OB used to create the sky model spectrum for that OB. I would then like to compare that list to the list of spectra denoted to be used as sky in the input OB to determine how many extra or fewer sky spectra were used in the combination. 


```python

```

## Rohit's question

I'm interested in getting the spectra for the AGN sample (specifically the LERGs) from the SV input, looking for sources between a redshift range, say 0.5 < z < 1 (or 1.4, whichever is the upper limit), and analysing their continuum. 

I'm also interested in using emission line properties to perform source classifications

1. select redshift 0.5 - 1
1. get LERGs:
    * $\log ([O III]/Hβ) - 1/3 \log ([N II]/Hα) + \log ([S II]/Hα) + \log ([O I]/Hα)$
1. get stacked spectra

* Need `FLUX_EL_W` and `ERR_FLUX_EL_W` columns from L2 data, where EL is the emission line and W its wavelength


```python
def excitation_index(oiii, hb, nii, sii, oi, ha):
    return log(oiii/hb) - (log(nii/ha) / 3) + log(sii/ha) + log(oi/ha)

data = Data()
l2stack = data.l2stack
redshift = l2stack.z
EI = excitation_index(l2stack.flux_oiii_5007, l2stack.flux_hbeta, l2stack.flux_nii_6583, l2stack.flux_oi_6300, 
                      l2stack.flux_sii_6716, l2stack.flux_halpha)

in_redshift_range = (l2stack.zbest > 0.5) & (l2stack.zbest < 1.)
is_lerg = EI < 0.95  

spectra = l2stack[in_redshift_range & is_lerg].l1stackspectra
reds = spectra[spectra.camera == 'red']
blues = spectra[spectra.camera == 'blue']


import matplotlib.pyplot as plt
# uncomment the next line if you are using ipython so that you can see the plots interactively (don't forget to do ssh -XY lofar)
# % matplotlib 
plt.plot(reds.wvls(), reds.flux())
plt.plot(blues.wvls(), blues.flux())
```

# Anniek's fluxes and other catalogues
We can add any catalogue or file we like to the database, all that is required is a precise knowledge of what data was used to create the catalogue/file (so we can associate it correctly). 

## Find all reprocessed Ha flux done by Anniek for a given OB


```python
data = Data()
data.ob.anniek.flux_halpha
```

## Join your own catalogue based on cname


```python
data = Data()
catalogue = read_my_nice_catalogue('...')   # has a cname column
data.join(catalogue, 'cname', weavetarget.cname, name='my_catalogue')  # only exists for this session
```

## Join your own file that reprocessed data from a specific file (like Anniek's)


```python
data = Data()
line_fluxes = read_line_fluxes('...')  # from the weave single spectra associated with run 100423 
data.join(line_fluxes, 'index', data.runs[100423].singlespectra, name='line_fluxes')
```
