import logging
import sys
from time import sleep

from astropy.io import fits

from weaveio.opr3.l2files import L2File

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='lowleveltest2')
with data.write:
    # data.restore_state(1653303490523)  # restore back to L1 only
    fs = [data.rootdir / f for f in ["L2/20160908/single_1002216__single_1002215_APS.fits", "L2/20160908/single_1002218__single_1002217_APS.fits", "L2/20160908/single_1002214__single_1002213_APS.fits", "L2/20170226/stack_1003438__stack_1003437_APS.fits", "L2/20170226/stack_1003414__stack_1003413_APS.fits", "L2/20170226/stack_1003450__stack_1003449_APS.fits", "L2/20170226/stack_1003426__stack_1003425_APS.fits", "L2/20160908/stack_1002226__stack_1002225_APS.fits", "L2/20160908/stack_1002214__stack_1002213_APS.fits", "L2/20160908/stack_1002238__stack_1002237_APS.fits", "L2/20160908/stack_1002250__stack_1002249_APS.fits", "L2/20170204/stack_1002958__stack_1002957_APS.fits", "L2/20170204/stack_1002982__stack_1002981_APS.fits", "L2/20170204/stack_1002994__stack_1002993_APS.fits", "L2/20170204/stack_1002946__stack_1002945_APS.fits", "L2/20170204/stack_1002970__stack_1002969_APS.fits"]]
    print(len(fs))
    data.write_files(*fs[1:], debug_time=True, dryrun=False, batch_size=50, test_one=False, parts=['GAND'], halt_on_error=False)
