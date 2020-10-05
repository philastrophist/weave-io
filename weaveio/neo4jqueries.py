def split_node_names(label, main_property, delimiter, *other_properties_to_set):
    s = f"""
MATCH (old: {label})
WHERE old.{main_property} CONTAINS '{delimiter}'
UNWIND split(old.{main_property}, "{delimiter}") as replacement
CALL apoc.refactor.cloneNodesWithRelationships([old])
YIELD input, output"""
    for p in other_properties_to_set+(main_property, ):
        s += f"\nSET output.{p} = replacement"
    return s + "\ndetach delete old"

def number_of_relationships(child_label, parent_label, lower, upper):
    s = f"""
match (children: {child_label})
unwind children as child
match (child)<-[]-(parent:{parent_label})
WITH child, parent, count(parent) as cnt"""
    wheres = []
    if lower is not None:
        wheres.append(f"cnt < {lower}")
    if upper is not None:
        wheres.append(f"cnt > {upper}")
    if len(wheres):
        s += '\nWHERE ' + ' OR '.join(wheres)
    s += f'\nRETURN "{parent_label}" as parent, "{child_label}" as child, child.id as child_id, ' \
         f'parent.id as parent_id, cnt'
    if lower is not None:
        s += f', {lower} as expected_lower'
    if upper is not None:
        s += f', {upper} as expected_upper'
    return s