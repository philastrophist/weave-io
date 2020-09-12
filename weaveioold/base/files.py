def make_fget(key):
    def fget(self):
        if not hasattr(self, f"_{key}"):
            result = getattr(self, f"read_{key}")()
            for k, v in result.items():
                setattr(self, f"_{k}", v)
        return getattr(self, f"_{key}")
    return fget


class FileMeta(type):
    def __new__(mcs, name, bases, dct, **kwargs):
        for a in dct['attributes']:
            dct[a] = property(make_fget(a))
        for pt in dct['product_types']:
            dct[pt.name] = property(make_fget(pt.name))
        return super().__new__(mcs, name, bases, dct)

    def __init__(cls, name, bases, dct, **kwargs):
        for a in cls.attributes:
            assert hasattr(cls, f'read_{a}'), f"{cls} needs all attributes to have a read_attribute method"
        super().__init__(name, bases, dct)


class File(metaclass=FileMeta):
    product_types = {}
    parent_hierarchies = []
    attributes = []


    @property
    def available_attributes(self):
        return {a: [getattr(self, f"_{a}")] for a in self.attributes}


    def __init__(self, base, fname, address=None):
        self.base = base
        self.fname = fname
        self.address = address

    @classmethod
    def from_address(cls, address):
        """
        Returns a list of File objects that match the given address
        """
        raise NotImplementedError


class FileList:
    pass