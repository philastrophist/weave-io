import logging
logging.basicConfig(level=logging.INFO)
from weaveio.opr3 import Data

data = Data(dbname='weaveio', host='127.0.0.1', user='neo4j', password='password', rootdir='/data/weave')
# print(data.graph.neograph.evaluate('match (n) return count(*)'))
# runids = list(map(str, data.runs.id()))
# singles = [f for f in data.find_files('l1stack_file') if any(r in f.name for r in runids)]
#
with data.write:
    fs = sorted(data.find_files('raw_file', skip_extant_files=False), key=lambda f: f.name)[:1]
    # while True:
    report = data.write_files(*fs, timeout=5*60, debug=True,
                              debug_time=True, dryrun=False, halt_on_error=True)
    # report = data.write_directory('raw_file', 'l1single_file', 'l1stack_file',
    #                               timeout=5*60, debug=True, debug_time=True,
    #                               dryrun=False, halt_on_error=False)
    # print(report)
# #     data.write_files(*singles, timeout=5*60)
    #     debug=True,
    #     # dryrun=True
    # )


#
# bad_files = data.find_files('l1stack_file', skip_extant_files=True)
# all_files = data.find_files('l1stack_file', skip_extant_files=False)
# good_files = list(set(all_files) - set(bad_files))
#
# for f in bad_files:
#     print(f)
# with data.write:
#     report = data.write_files('/beegfs/car/weave/weaveio/L1/20160909/stack_1002286.fit', batch_size=1,
#                               dryrun=True, test_one=True,
#                               debug_time=True, debug=True, timeout=5*60, halt_on_error=False)
#     print(report)
