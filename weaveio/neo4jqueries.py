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