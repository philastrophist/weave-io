# WEAVE-IO

This code is intended to be one level above reading fits files and tables. 
There should be minimal science at work here. Essentially, this is just a way to find and index the data.
All of this design is up for discussion.

A quick sketch of intended workflow (python 3):

Reading all stacked spectra from an OB
```python
from weaveio import Address, Data
data = Data('my_local_directory')
ob = data[Address(obid=1245)]
spectra_block = ob.stacks.spectra
spectra_block[0].plot()
```
Reading a single exposure spectrum in the red arm
```python
from weaveio import Address, Data
data = Data('my_local_directory')
ob = data[Address(obid=1245)]
spectrum = ob.runs[0].spectrum.red
spectrum.plot()
```

Finding expected magnitudes
```python
from weaveio import Address, Data
data = Data('my_local_directory')
ob = data[Address(obid=1245)]
sdssg_mags = ob.stacks.fibinfo['MAG_G']  # all g-band magnitudes for all stacked data
``` 

## Rules for development
Since this package is intended for wide use within WL, please use the [git flow](https://jeffkreeftmeijer.com/git-flow/) methods.
This keeps the stable production level code away from new unstable features, and it means people can rely on that stability.
The WEAVE data structure is likely to change in small ways and so git flow is well suited to handle this.

There are 3 main branches:

* Release - Static versions of the code base that will never change.
* Master - production level code: the most up to date version of the code base - should be well tested
* Develop - pre-production level code: where all the different contributions are unified and tested before being signed off.

Also, Feature branches: Where your own innovations happen

Example:
0. Install git flow: https://github.com/nvie/gitflow/wiki/Installation
1. Fork [WEAVE-LOFAR/weave-io](https://github.com/WEAVE-LOFAR/weave-io) to your own account
2. Clone your version: `git clone USER/weave-io.git && git flow init -d`
3. Make a new feature/change/fix:
    * New branch `git flow feature start my_fancy_feature`
    * Do the work
    * Commit changes: `git add file1 file2 && git commit -m "description"`
    * Commit more changes: `git add file1 file2 && git commit -m "description 2"`
    * Push to your github account: `git flow feature publish my_fancy_feature`
    * Make a pull request to [WEAVE-LOFAR/weave-io](https://github.com/WEAVE-LOFAR/QAG/compare/): 
4. Dicussion
5. Feature is merged into `develop` (to combine it with all the other features)
6. Further discussion
7. Merge in `master` for common use. Master should always be the most up to date and tested branch. 
Users should be able to install and use immediately.
![](https://blog.axosoft.com/wp-content/uploads/2018/03/FlowChart-701x1024.png)