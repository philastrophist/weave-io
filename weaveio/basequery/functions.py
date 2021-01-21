from typing import overload, List


def sum(x):
    pass


def max(x):
    pass


def min(x):
    pass


def all(x):
    pass


def any(x):
    pass


def count(x):
    pass


class OperableMixin:

    def __add__(self, x: float) -> float:
        return super().__add__(x)

    def __sub__(self, x: float) -> float:
        return super().__sub__(x)

    def __mul__(self, x: float) -> float:
        return super().__mul__(x)

    def __truediv__(self, x: float) -> float:
        return super().__truediv__(x)

    def __pow__(self, x: float, mod: None = ...) -> float:
        return super().__pow__(x, mod)

    def __radd__(self, x: float) -> float:
        return super().__radd__(x)

    def __rsub__(self, x: float) -> float:
        return super().__rsub__(x)

    def __rmul__(self, x: float) -> float:
        return super().__rmul__(x)

    def __rtruediv__(self, x: float) -> float:
        return super().__rtruediv__(x)

    def __rpow__(self, x: float, mod: None = ...) -> float:
        return super().__rpow__(x, mod)

    def __eq__(self, x: object) -> bool:
        return super().__eq__(x)

    def __ne__(self, x: object) -> bool:
        return super().__ne__(x)

    def __lt__(self, x: float) -> bool:
        return super().__lt__(x)

    def __le__(self, x: float) -> bool:
        return super().__le__(x)

    def __gt__(self, x: float) -> bool:
        return super().__gt__(x)

    def __ge__(self, x: float) -> bool:
        return super().__ge__(x)

    def __neg__(self) -> float:
        return super().__neg__()

    def __abs__(self) -> float:
        return super().__abs__()

    def __len__(self) -> int:
        return super().__len__()

    def __contains__(self, o: object) -> bool:
        return super().__contains__(o)