import logging

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()

red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
blue = red.adjunct#.l1stack_spectra

aligned = align(red=red, blue=blue)
q = aligned.snr * 2
print(q(limit=10))


aligned = align(red=red, blue=blue)
avg = mean(aligned.l1stack_spectra.snr, wrt=aligned)
q = avg / aligned.snr

print(q(limit=10))