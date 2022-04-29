def mask_infs(x):
    return f"CASE WHEN {x} > apoc.math.maxLong() THEN null ELSE {x} END"

def is_regex(other):
    """
    Regex is defined either by:
     1. starting and ending the string with '/'
        OR
     2. When the string contains * and the string doesn't start and end with '"'
    """
    return (other.startswith('/') and other.endswith('/')) or \
           ('*' in other and not (other.startswith('"') and other.endswith('"')))


