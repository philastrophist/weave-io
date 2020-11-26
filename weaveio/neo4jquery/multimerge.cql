// param: specs => LIST OF LIST OF ANY  //[[node:node, reltype:string, properties:map]]
// param: labels => LIST OF STRING
// param: properties => MAP
// param: createprops => MAP
// param: createpropsrel => MAP
// returns: child => NODE
// description: Performs a cypher merge to create or match a node(s) that has only the exact relationships given in spec
////////////////////////////////////////////////////////////

WITH $specs as specs
UNWIND specs as spec
OPTIONAL MATCH p=(a)-[r]->(d)
WHERE a = spec[0] AND type(r) = spec[1]
AND all(l in $labels WHERE l in labels(d))
AND all(k in keys(spec[2]) WHERE r[k] = spec[2][k] OR spec[2][k] is NULL) // specified relationship properties
AND all(k in keys($properties) WHERE d[k] = $properties[k] OR $properties[k] is NULL)  // specifed node properties

WITH specs, spec, collect(DISTINCT d) as childlist  // [child1, child2] per spec relation
WITH specs, collect(spec[0]) as parents, collect(childlist) as childlists  // [[child1, child2], ...] per entire specification

// now do a intersection reduce to get the parents which are shared between all requiured spec paths
WITH specs, parents, reduce(shared = childlists[0], childlist IN childlists | apoc.coll.intersection(shared, childlist)) as children
WITH specs, parents, children+[null] as children
UNWIND children as child
OPTIONAL MATCH (_child)
WHERE NOT EXISTS{MATCH (_child)<-[]-(other) WHERE NOT (other in parents)} AND _child = child
WITH specs, collect(_child) as children

CALL apoc.do.when(size(children) > 0, "UNWIND $children as child RETURN child", "
        CALL apoc.create.node($labels, $props) YIELD node
		SET node += $createprops
		WITH node
		UNWIND $specs as spec
		CALL apoc.create.relationship(spec[0], spec[1], spec[2], node) yield rel
        SET rel += $createpropsrel
        RETURN DISTINCT node as child",
		{specs: specs, children:children, labels:$labels, props:$properties, createprops:$createprops, createpropsrel:$createpropsrel}
	) YIELD value
RETURN value.child as child