import logging

logging.basicConfig(level=logging.INFO)
from weaveio import *
data = Data()

red = data.l1single_spectra[data.l1single_spectra.camera == 'red']
blue = red.adjunct

aligned = align(red=red, blue=blue)
avg = mean(aligned.l1stack_spectra.snr, wrt=aligned)
snr =  aligned.snr

# q = aligned[[avg / snr]]
q = aligned[{'divide': avg / snr}]
# lines, ordering, params = data.query._debug_output(graph_export_fname='parser')
# print(ordering)
# print(lines)
# print(q(limit=10))

for i, row in enumerate(q):
    print(row)
    if i > 10:
        break