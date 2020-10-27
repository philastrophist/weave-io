import xxhash


class Varname:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

def quote(x):
    """
    Return a quoted string if x is a string otherwise return x
    """
    if isinstance(x, str):
        return f"'{x}'"
    if isinstance(x, Varname):
        return x.name
    return x


def hash_pandas_dataframe(df):
    digester = xxhash.xxh64()
    for i in df.astype(str).values.ravel():
        digester.update(i)
    return digester.hexdigest()