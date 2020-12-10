// param: specs => LIST OF LIST OF ANY  //[[node:node, reltype:string, properties:map]]
// param: labels => LIST OF STRING
// param: properties => MAP
// param: exclusive => BOOL => true
// param: allownull => BOOL => true
// returns: child => NODE
// description: Performs a cypher match to match a node(s) that has only the exact relationships given in spec. Labels must be identical, properties are matched as normal in cypher.
////////////////////////////////////////////////////////////

UNWIND $specs as spec
WITH spec
WHERE spec[0] is not null
WITH [spec[0], spec[1], spec[2]] as newspec
WITH collect(newspec) as specs  // filter out null entries

UNWIND specs as spec
OPTIONAL MATCH p=(a)-[r]->(d)
WHERE a = spec[0] AND type(r) = spec[1]
AND all(k in keys($properties) WHERE d[k] = $properties[k] OR $properties[k] is NULL)  // specifed node properties
AND labels(d) = $labels
AND all(k in keys(spec[2]) WHERE r[k] = spec[2][k] OR spec[2][k] is NULL) // specified relationship properties


WITH specs, spec, collect(DISTINCT d) as childlist  // [child1, child2] per spec relation
WITH specs, collect(spec[0]) as parents, collect(childlist) as childlists  // [[child1, child2], ...] per entire specification

// now do a intersection reduce to get the parents which are shared between all requiured spec paths
WITH specs, parents, reduce(shared = childlists[0], childlist IN childlists | apoc.coll.intersection(shared, childlist)) as children
WITH specs, parents, children+[null] as children
UNWIND children as child
CALL apoc.do.when($exclusive = true, 'OPTIONAL MATCH (_child) WHERE NOT EXISTS{MATCH (_child)<-[]-(other)
WHERE NOT (other in $parents)} AND _child = $child
RETURN _child as child', 'RETURN $child as child', {child:child, parents:parents}) YIELD value

CALL apoc.do.when($allownull, 'RETURN $value.child as child', 'WITH value.child as child WHERE child is not NULL RETURN child', {value:value}) yield value as value2
RETURN value2.child as child
