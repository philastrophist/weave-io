# The Schema

Neo4j is schemaless, which is good because it is flexible and we can adapt the data structure as needed, on the fly.
However, WEAVE files are not schemaless and they have a specific structure.
The current structure of WEAVE data is summarised in the following graph:

![relationships](relations.png)
The arrows can be read as "is required by" and the numbers correspond to how many instances are required.

There are a few points to make:

1. Files are at the bottom of the structure and have no successor nodes since everything springs from them.
2. Each row in a spectrum or table that corresponds to another object (for example, one spectrum in an L1 single file corresponding to one target) has its own node.
3. Binary data products that should not be put into the graph are not represented, instead, they are named in the `products` attribute of a given node. For example, `flux`, `ivar`, `noss_flux`, and `noss_ivar` of each fibre spectrum in an L1 single file are represented by the following Cypher syntax: `(:L1SingleSpectrum {products: ['flux', 'ivar', 'noss_flux', 'noss_ivar'])`. 

