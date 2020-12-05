import py2neo
from time import clock
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

graph = py2neo.Graph(host='host.docker.internal')
deletion = graph.run('call apoc.periodic.iterate("MATCH (n) return n", "DETACH DELETE n", {batchSize:1000}) yield failedBatches, failedOperations').to_ndarray()
assert np.all(deletion == 0)

times = []
xs = range(1, 50)
for i in tqdm(xs):
    start = clock()
    query = """
    with timestamp() as time0, $a as uniquetorun
    MERGE (arm:ArmConfig {id: uniquetorun})
    MERGE (obspec: OBSpec {xml: uniquetorun})
    MERGE (ob: OB {obid: uniquetorun})
    MERGE (obspec)-[:req]->(ob)
    MERGE (obspec)<-[:req]-(fibreset: FibreSet)
    MERGE (arm)-[:req]->(obspec)
    
    MERGE (exposure:Exposure {expmjd: uniquetorun, otherthing: uniquetorun})
    MERGE (ob)-[:req]->(exposure)
    MERGE (run:Run {runid: uniquetorun, otherthing: uniquetorun})
    MERGE (exposure)-[:req]->(run)
    MERGE (run)<-[:req]-(arm)
    MERGE (run)-[:req]->(raw:Raw {checksum: uniquetorun})

    WITH * UNWIND range($a*$length, ($a*$length)+$length) as uniquei
        with *,  uniquei/$a as uniqueinloop
        MERGE (survey: Survey {survey: uniquetorun})
        MERGE (subprogramme: SubProgramme {targprog: uniquetorun})
            MERGE (survey)-[:req]->(subprogramme)
        MERGE (surveycatalogue:SurveyCatalogue {catname: uniqueinloop})
        MERGE (surveycatalogue)<-[:req]-(subprogramme)

        MERGE (w:WeaveTarget {cname: uniquei})
        MERGE (s:SurveyTarget {targid: uniquei, ra: uniquei, dec: uniquei, otherthing: uniquei})<-[:req]-(surveycatalogue)
        MERGE (w)-[:req]->(s)
        MERGE (f: Fibre {id: uniqueinloop})
        MERGE (s)-[:req]->(ft:FibreTarget {fibrera: uniquei, fibredec: uniquei, status: uniquei, otherthing: uniqueinloop})<-[:req]-(f)
        MERGE (ft)-[:req]->(fibreset)
        MERGE (ft)-[:req]->(spec:L1SingleSpectrum {checksum: uniquei, otherthing: uniquei})<-[:req]-(raw)
        
    
    """
    graph.run(query, parameters={'a': i, 'length': 9000})
    times.append(clock() - start)
print(query)
times = np.array(times)
print(times)
plt.plot(xs, times)
plt.gcf().set_facecolor('white')
plt.savefig('times.png')