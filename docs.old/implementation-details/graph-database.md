# The Graph Database

Weave-io runs on top of a Neo4j database hosted at lofar.herts.ac.uk.
Neo4j is a graph database which means that there is no internal schema and no SQL interface. 
Instead Neo4j stores data as `nodes` and `relationships` (each with their own corresponding json-like data store).

This is beneficial to us since it allows statements like "OB-1234 has 2 exposures" to translate directly into code.
If we wanted to find the exposures belonging to an OB with `id=1234` with `weaveio` we would write this as:

    database.obs[1234].exposures

Here we've made use of the graph database structure. Each dot, `.`, can be interpreted as retrieving the child objects:
1. `database.obs` - Get all the OBs in the database
2. `database.obs[1234]` - Select only the OB with `id=1234`
3. `database.obs[1234].exposures` - Get all exposures belonging to this one OB

In cypher, neo4j's native query language, this would be written as the following:

    MATCH (ob: OB {id: 1234})
    OPTIONAL MATCH (ob)-[:is_required_by]->(e: Exposure)
    RETURN e

But the user does not need to see or use cypher.
