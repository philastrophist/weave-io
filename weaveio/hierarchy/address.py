
class Address:
    coordinates = []

    def __init__(self, **keys):
        for c in self.coordinates:
            assert hasattr(self, c), f"Not all coordinates are defined: {c} is missing"
            assert isinstance(getattr(self, c), list) or getattr(self, c) is None

    @property
    def knowns(self):
        return [c for c in self.coordinates if getattr(self, c) is not None]

    @property
    def unknowns(self):
        return [c for c in self.coordinates if getattr(self, c) is None]

    def check(self, variable, dtype, validator=None, failmsg='', choices=None):
        if choices is not None:
            if any(not isinstance(c, dtype) for c in choices):
                raise ValueError(f"{choices} must of type {dtype}")
        if isinstance(variable, (list, tuple)):
            rs = list(map(dtype, variable))
            if choices is not None:
                if any(r not in choices for r in rs):
                    raise ValueError(f"{variable} must be one of {choices}")
        elif variable is None:
            if choices is not None:
                rs = choices
            return None  # selects all
        else:
            r = dtype(variable)
            if choices is not None:
                if r not in choices:
                    raise ValueError(f"{variable} must be one of {choices}")
            rs = [r]
        if validator is not None:
            for r in rs:
                if not validator(r):
                    raise ValueError(f"{variable} is invalid by its {validator}: {failmsg}")
        return rs

    def match(self, product):
        matches = []
        for c in self.coordinates:
            required = getattr(self, c) # always a list or None
            if required is not None:
                matches.append(getattr(product, c) in required)
        return all(matches)

    def is_compatible(self, address):
        return not any(k in self.knowns for k in address.knowns)

    def combine(self, address):
        if not self.is_compatible(address):
            raise KeyError(f"The new address {address} is not compatible with this one")
        knowns = {getattr(self, c) for c in self.knowns}
        knowns.update({getattr(address, c) for c in address.knowns})
        return Address(**knowns)