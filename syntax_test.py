from weaveio import *
import logging
logging.basicConfig(level=logging.INFO)

data = Data(dbname='lowleveltest2')
cnt = count(data.l1single_spectra.l2single, wrt=data.l1single_spectra)
l1 = data.l1single_spectra[cnt > 0]#count(data.l1single_spectra[cnt > 0].l2single, wrt=data.l1single_spectra)
q = l1[[l1.l2single.aps.version, cnt, l1.l2single.templates['qso'].redshift_array]]
# q = data.obs.l1single_spectra[['snr']]
print('\n'.join(q._precompile()._to_cypher()[0]))
print(q(limit=4))