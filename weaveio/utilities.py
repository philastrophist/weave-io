

class Varname:
    def __init__(self, name):
        self.name = name

def quote(x):
    """
    Return a quoted string if x is a string otherwise return x
    """
    if isinstance(x, str):
        return f"'{x}'"
    if isinstance(x, Varname):
        return x.name
    return x