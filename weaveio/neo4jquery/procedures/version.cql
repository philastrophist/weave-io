// param: specs => LIST OF LIST OF ANY  //[[node:node, reltype:string, properties:map]]
// param: child => NODE  // the node that is being versioned
// param: versionedlabels => LIST OF STRING  // the labels which define a version (not all parent nodes have to be unique)
// param: versionattribute => STRING
// returns: version => INT
// description: Given a parent specification and a child, find all other children of the specification that have versionedlabels. The version # of the passed child node is 1 + max(versions).
// requires: multimatch
////////////////////////////////////////////////////////////

Unwind $specs as spec
WITH CASE WHEN any(l in $versionedlabels WHERE l in labels(spec[0])) THEN spec ELSE null END AS row, $child as unversioned_child
WITH collect(row) as newspecs, unversioned_child
CALL multimatch(newspecs, labels(unversioned_child), {}, false) YIELD child  // false= other unmentioned parent nodes are allowed
WHERE (child <> unversioned_child) OR (child is null)
WITH unversioned_child, max(child[$versionattribute]) + 1 as latest_version
WITH *, CASE WHEN latest_version is null THEN 0 ELSE latest_version END as latest_version
CALL apoc.create.setProperty(unversioned_child, $versionattribute, latest_version) YIELD node
RETURN latest_version as version