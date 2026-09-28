"""Exact, ordered arithmetic in the real quadratic field Q(sqrt(2))."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from math import sqrt
from numbers import Rational


@total_ordering
@dataclass(frozen=True, eq=False)
class Qsqrt2:
    """The exact number ``a + b*sqrt(2)`` for rational ``a`` and ``b``.

    Pass fractions or integer/string coefficients, not floating approximations.
    Comparisons use rational arithmetic even arbitrarily close to zero.
    """

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, "a", Fraction(self.a))
        object.__setattr__(self, "b", Fraction(self.b))

    @staticmethod
    def _coerce(value):
        if isinstance(value, Qsqrt2):
            return value
        if isinstance(value, Rational):
            return Qsqrt2(value)
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Qsqrt2(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __neg__(self):
        return Qsqrt2(-self.a, -self.b)

    def __sub__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Qsqrt2(self.a * other.a + 2 * self.b * other.b,
                      self.a * other.b + self.b * other.a)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        denominator = other.a * other.a - 2 * other.b * other.b
        if not denominator:
            raise ZeroDivisionError("division by zero in Q(sqrt(2))")
        return Qsqrt2((self.a * other.a - 2 * self.b * other.b) / denominator,
                      (self.b * other.a - self.a * other.b) / denominator)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else other / self

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return (1 / self) ** (-exponent)
        result, base = Qsqrt2(1), self
        while exponent:
            if exponent & 1:
                result *= base
            base *= base
            exponent >>= 1
        return result

    def sign(self):
        a, b = self.a, self.b
        if not b:
            return (a > 0) - (a < 0)
        if not a:
            return (b > 0) - (b < 0)
        if (a > 0) == (b > 0):
            return 1 if a > 0 else -1
        difference = a * a - 2 * b * b
        return ((difference > 0) - (difference < 0)) * (1 if a > 0 else -1)

    def __eq__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self.a == other.a and self.b == other.b

    def __lt__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else (self - other).sign() < 0

    def __hash__(self):
        return hash(self.a) if not self.b else hash((self.a, self.b))

    def __bool__(self):
        return bool(self.a or self.b)

    def __abs__(self):
        return -self if self.sign() < 0 else self

    def __float__(self):
        return float(self.a) + float(self.b) * sqrt(2)

    def __repr__(self):
        if not self.b:
            return str(self.a)
        if not self.a:
            return f"({self.b})*sqrt(2)"
        operator = "+" if self.b > 0 else "-"
        return f"{self.a} {operator} ({abs(self.b)})*sqrt(2)"


ZERO = Qsqrt2()
ONE = Qsqrt2(1)
SQRT2_INV = Qsqrt2(0, Fraction(1, 2))
