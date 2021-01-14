# The Graph Database

Weave-io runs on top of a Neo4j database hosted at lofar.herts.ac.uk.
Neo4j is a graph database which means that there is no schema and no SQL interface. 
Instead Neo4j stores data as `nodes` and `relationships` (each with their own corresponding json-like data store).

This is beneficial to us since it allows statements like "OB-1234 has 2 exposures" to translate directly into code.
For that example, to retrieve the exposure for that OB, we use the following CYPHER query: 

    MATCH (ob: OB {id: 1234})
    OPTIONAL MATCH (ob)-[:is_required_by]->(e: Exposure)
    RETURN e

However, since we want WEAVE-IO to be intuitive and CYPHER requires some learning, we abstract a different WEAVE-IO syntax on top of the cypher backend.
The above example is written as:

    from weaveio import Data
    data = Data(username, password)
    data.obs[1234].exposures()
