// param: specs => LIST OF LIST OF ANY  //[[node:node, reltype:string, properties:map]]
// param: child => NODE  // the node that is being versioned
// param: versionedlabels => LIST OF STRING  // the labels which define a version (not all parent nodes have to be unique)
// param: versionattribute => STRING
// returns: version => INT
// description: Will version a node based on selected parent node labels and its own labels. If there is more than one instance of this label with those parent labels attached to it, the node will have a versionlabel property set to the number of instances. The relationship type is also versioned.
// requires: multimerge
////////////////////////////////////////////////////////////

CALL multimerge(specs, labels($child), {}, {}, {}) yield child
Unwind $specs as spec
WITH CASE WHEN any(l in $versionedlabels WHERE l in labels(spec[0])) THEN [spec[0], spec[1], {}] ELSE null END AS row, $child as unversioned_child
WITH collect(row) as newspecs, unversioned_child
WITH unversioned_child, max(child[$versionattribute]) as latest_version
WITH *, CASE WHEN latest_version is null THEN 0 ELSE latest_version END as latest_version
CALL apoc.create.setProperty(unversioned_child, $versionattribute, latest_version) YIELD node
RETURN latest_version as version