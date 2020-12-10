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

# Neo4j Data Model
## Node: 
`(:Hierarchy:Type1:Type2:... {id: idvalue, factorname: factorvalue, ..., products: [<string>, ...], concatenation_constants: [<string>, ...], version: <int>)`

* `Hierarchy:Type1:Type2`: Python types as neo4j labels representing object oriented class structure. Everyone has `Hierarchy` as its base.
* `{factorname: factorvalue}, ...`: The actual data of this node represented in a dictionary
* `products: [<string>]`: Binary data which resides in a file and cannot easily be represented in neo4j. There must a file which is 1 step away from this node.
When this data is requested, the filename, concatenation_constants, and index, will be returned.
* `concatenation_constants`: The names of factors whose values must be the same, in order to concatenate two or more of this type of product.
* `version`: The priority of this node when considering duplicate results.
  

## Relationship: 
`(parent)-[:IS_REQUIRED_BY {order: <int>, optional[name: <string>]}]->(child)`

* `(parent)`: a preceeding node which defines the parent node (along with others not seen in this example)
* `IS_REQUIRED_BY`: The relationship type (which is the same for all weaveio relationships)
* `order`: The order in which the parent is given to the child when instantiated. In `Thing(a=[a1,a2], b=b1, c=[c1, c2])` a1 has order 0, a2 has 1, b1 has 0, c1 has 0, c2 has 1. 
* `name`: The name of this order if there is one.
* `(child)`: The node being defined.
