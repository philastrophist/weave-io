import logging
import sys
from time import sleep

logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='lowleveltest2')
graph = data.graph.neograph
assert graph.run("""match (e:Exposure)
optional match (e)--(r:Run)
with e, count(r) as cnt
return all(x in collect(cnt) where x < 3)""").evaluate()

