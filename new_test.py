from weaveio.opr3 import Data

data = Data()
# paths, singulars = data.paths_to_hierarchy('Run',  'L1SingleSpectrum', True)
# for p, s in zip(paths, singulars):
#     print(p, s)
q = data.runs.l1_spectra.wvl
print('\n'.join(q._precompile()._to_cypher()[0]))