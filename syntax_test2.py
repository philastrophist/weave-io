from weaveio import *

data = Data()
# ingested = data.ingested_spectra
# q = ingested.camera
q = data.ingested_spectra.camera

print('\n'.join(q._precompile()._to_cypher()[0]))
# print(q(limit=1))