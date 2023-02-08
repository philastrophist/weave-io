# weaveio
This is documentation for the use, maintenance, and development of `weaveio`, the data product database tool in use by the WEAVE-LOFAR survey.

## Philosophy
The purpose of `weaveio` is to facilitate easy querying of complex data spread over multiple file types. 

Specifically, it was designed to allow the user write complex queries over the L1 and L2 data structures output by the WEAVE CPS and APS. 
The data consists of weather measurements, spectra (both 1D and 2D), previous survey input target observations, redshifts, emission lines, velocity dispersion etc.

All of this information is highly hierarchical. I.e. a spectrum requires a target, which has an {ra,dec} coordinate. 
The spectrum also has numerous emission line fits and redshifts which have been inferred from the spectrum.
You could write this as `(target {ra, dec})-->(spectrum)-->(fit {redshift})-->(emission_line)`. 
This kind of structure lends itself to be written in an object orientated way and queried as such.

Essentially, all of `weaveio` is designed to turn this:

```python linenums="1"
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import os
target_name = 'xxxxx'
exposure_mjd = 57639
wvl = []
flux = []
for file in os.listdir('opr3'):
    if 'single' in file:
        hdus = fits.open(file)
        if floor(hdus[0].header['MJD']) == exposure_mjd:
            t = Table(hdus['FIBTABLE'].data)
            filt = t['TARGNAME'] == target_name
            if sum(filt):
                header = hdulist[1].header
                increment, zeropoint = header['cd1_1'], header['crval1'] 
                size = header['naxis1']
                wvl.append(np.arange(0, size) * increment) + zeropoint)
                flux.append(hdulist[1].data[filt])
plt.plot(wvl, flux)
plt.title('All L1 single spectra of {target_name} on day {exposure_mjd}')
```

Into this:

```python linenums="1"
import matplotlib.pyplot as plt
from weaveio import *
db = Data('opr3')
spectra = db.survey_targets[target_name].l1single_spectra
result = spectra[floor(spectra.exposure.mjd) == exposure_mjd][['wvl', 'flux']]()
plt.plot(result.wvl, result.flux)
plt.title('All L1 single spectra of {target_name} on day {exposure_mjd}')         
```