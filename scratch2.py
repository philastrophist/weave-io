import types

class Other(object):
    pass


class Hierarchy:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __new__(cls, *args, **kwargs):
        print(cls, args, kwargs)
        return super().__new__(cls,  *args, **kwargs)


class OB(Hierarchy):
    idname = 'id'


ob = OB('hello', a=1)