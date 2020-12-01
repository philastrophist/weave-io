// param: specs => LIST OF LIST OF ANY  //[[node:node, reltype:string, properties:map]]
// param: labels => LIST OF STRING
// param: properties => MAP
// param: createprops => MAP
// param: createpropsrel => MAP
// returns: child => NODE
// description: Performs a cypher merge to create or match a node(s) that has only the exact relationships given in spec. Labels must be identical, properties are matched as normal in cypher.
// requires: multimatch
////////////////////////////////////////////////////////////

CALL multimatch($specs, $labels, $properties) YIELD child
WITH collect(child) as children
CALL apoc.do.when(size(children) > 0, "UNWIND $children as child RETURN child", "
        CALL apoc.create.node($labels, $props) YIELD node
		SET node += $createprops
		WITH node
		UNWIND $specs as spec
		CALL apoc.create.relationship(spec[0], spec[1], spec[2], node) yield rel
        SET rel += $createpropsrel
        WITH node as child
        RETURN child",
		{specs: $specs, children:children, labels:$labels, props:$properties, createprops:$createprops, createpropsrel:$createpropsrel}
	) YIELD value
RETURN DISTINCT value.child as child