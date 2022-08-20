import logging

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()

red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
blue = red.adjunct
# cols = ['run.id', 'snr']
# cols = (red[cols], blue[cols])
# q = red[cols]
# red._G.export('parser2')
# r = q(limit=10)
# print(r)

aligned = align(red=red, blue=blue)
print((aligned.snr * 2)(limit=10))
# print(aligned[['run.id', 'snr']](limit=10))

