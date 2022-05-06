from typing import List


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


def remove_successive_duplicate_lines(cypher: List[str]):
    """
    Given a list of cypher statements, remove any lines that appear more than once in succession only
    """
    if not cypher:
        return []
    deduplicated = [cypher[0]]
    for line in cypher[1:]:
        if line != deduplicated[-1]:
            deduplicated.append(line)
    return deduplicated


def safe_name(name):
    if name is None:
        return name
    return '__dot__'.join(name.split('.'))
