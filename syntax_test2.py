from weaveio import *

data = Data()
l2s = data.l2stacks
ingested = l2s.ingested_spectra
q = ingested.camera
# q = count(l2s.ingested_spectra, wrt=l2s)

print('\n'.join(q._precompile()._to_cypher()[0]))
# print(q(limit=1))