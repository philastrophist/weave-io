MERGE (a1: A {id: 1})
MERGE (a2: A {id: 2})
MERGE (a3: A {id: 3})

MERGE (b1: B {id: 1})
MERGE (b2: B {id: 2})
MERGE (b3: B {id: 3})

with [a1, a2, a3] as input_anodes, [b1, b2, b3] as input_bnodes, ['red', 'blue', 'green'] as input_bnames
CALL apoc.lock.nodes(input_anodes)
CALL apoc.lock.nodes(input_bnodes)

UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
WITH [ai, input_anodes[ai], 'arel', NULL] AS arow, input_bnodes, input_bnames
WITH collect(arow) as anodes, input_bnodes, input_bnames

UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
WITH [bi, input_bnodes[bi], 'brel', input_bnames[bi]] AS brow, anodes
WITH collect(brow) as bnodes, anodes
WITH *, bnodes+anodes as specification


WITH specification as specs
UNWIND specs as spec
OPTIONAL MATCH p=(a)-[r {order: spec[0]}]->(d: SomeLabel) WHERE a = spec[1] AND type(r) = spec[2] AND (r['name'] = spec[3] OR spec[3] is NULL)
WITH specs, spec, collect(DISTINCT d) as childlist  // [child1, child2] per spec relation
WITH specs, collect(spec[1]) as parents, collect(childlist) as childlists  // [[child1, child2], ...] per entire specification

// now do a intersection reduce to get the parents which are shared between all requiured spec paths
WITH specs, parents, reduce(shared = childlists[0], childlist IN childlists | apoc.coll.intersection(shared, childlist)) as children
UNWIND children as child
OPTIONAL MATCH (_child)
WHERE NOT EXISTS{MATCH (_child)<-[]-(other) WHERE NOT (other in parents)} AND _child = child
WITH specs, collect(_child) as children

CALL apoc.do.when(size(children) > 0, "UNWIND $children as child RETURN child",
		"
        CALL apoc.create.node($labels, $props) YIELD node
		SET node += $createprops
		WITH node
		UNWIND $specs as spec
		CALL apoc.do.case([spec[3] is NULL, 'RETURN {order: $spec[0]} as prop'], 'RETURN {order: $spec[0], name: $spec[3]} as prop', {spec:spec}) YIELD value
		CALL apoc.create.relationship(spec[1], spec[2], value.prop, node) yield rel
        SET rel += $createpropsrel
        RETURN DISTINCT node as child",
		{specs: specs, children:children, labels:['SomeLabel'], props:{}, createprops:{}, createpropsrel:{}}
	) YIELD value
WITH collect(value) as vs
MATCH (n) return n