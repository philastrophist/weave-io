def mask_infs(x):
    return f"CASE WHEN {x} > apoc.math.maxLong() THEN null ELSE {x} END"
