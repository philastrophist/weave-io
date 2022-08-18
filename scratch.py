import logging

import inspect

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()

red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
blue = red.adjunct
aligned = align(red=red, blue=blue)
aligned[['run.id', 'snr']](limit=10)

