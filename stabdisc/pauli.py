"""Binary symplectic arithmetic for Hermitian Paulis and stabilizer codes.

An unsigned n-qubit Pauli occupies 2*n bits: ``x | (z << n)``. Bit q of
both masks refers to qubit q, the q-th character of a Pauli word. The signed
encoding adds bit 2*n, representing an overall factor of minus one.

For each qubit the local index ``x + 2*z`` selects I, X, Z, or Y. The
Hermitian convention is P(x,z) = i**popcount(x & z) X**x Z**z. These are
low-level operations: callers supply valid masks and valid stabilizer groups.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


# Local products in the order I, X, Z, Y. OUT identifies the output Pauli;
# PH gives its phase exponent, so P_left P_right = i**PH * P_OUT.
OUT = (
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
)
PH = (
    (0, 0, 0, 0),
    (0, 0, 3, 1),
    (0, 1, 0, 3),
    (0, 3, 1, 0),
)
LABEL_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Z": (0, 1),
    "Y": (1, 1),
}


def parity(x: int) -> int:
    """Return the sum of the set bits modulo two."""
    return x.bit_count() & 1


def enc(n: int, x: int, z: int, sign: int = 0) -> int:
    """Pack x/z masks and a sign bit into a signed Hermitian Pauli."""
    return x | (z << n) | ((sign & 1) << (2 * n))


def dec(n: int, p: int) -> Tuple[int, int, int]:
    """Unpack a valid signed Pauli into its x mask, z mask, and sign bit."""
    sign = p >> (2 * n)
    unsigned = p & ((1 << (2 * n)) - 1)
    x_mask = unsigned & ((1 << n) - 1)
    z_mask = unsigned >> n
    return x_mask, z_mask, sign


def commute_unsigned(n: int, u: int, v: int) -> bool:
    """Test whether the binary symplectic product of two labels vanishes."""
    mask = (1 << n) - 1
    left_x, left_z = u & mask, u >> n
    right_x, right_z = v & mask, v >> n
    return parity((left_x & right_z) ^ (left_z & right_x)) == 0


def multiply(n: int, p: int, q: int) -> int:
    """Multiply commuting signed Paulis, including their resulting sign.

    Anticommuting Hermitian Paulis have an anti-Hermitian product, which this
    signed representation cannot encode; such products raise ``ValueError``.
    """
    left_x, left_z, left_sign = dec(n, p)
    right_x, right_z, right_sign = dec(n, q)
    output_x = output_z = phase = 0
    for qubit in range(n):
        left = ((left_x >> qubit) & 1) | (((left_z >> qubit) & 1) << 1)
        right = ((right_x >> qubit) & 1) | (((right_z >> qubit) & 1) << 1)
        output = OUT[left][right]
        phase = (phase + PH[left][right]) & 3
        output_x |= (output & 1) << qubit
        output_z |= ((output >> 1) & 1) << qubit
    if phase not in (0, 2):
        raise ValueError("Attempted to multiply anticommuting Paulis")
    sign = left_sign ^ right_sign ^ (phase >> 1)
    return enc(n, output_x, output_z, sign)


def pauli_string_to_unsigned(label: str) -> int:
    """Encode an I/X/Y/Z word, ignoring case; its first character is qubit 0."""
    n = len(label)
    x_mask = z_mask = 0
    for qubit, character in enumerate(label):
        x_bit, z_bit = LABEL_TO_XZ[character.upper()]
        x_mask |= x_bit << qubit
        z_mask |= z_bit << qubit
    return x_mask | (z_mask << n)


def unsigned_to_pauli_string(n: int, u: int) -> str:
    """Decode an unsigned label to a word with qubit 0 on the left."""
    mask = (1 << n) - 1
    x_mask, z_mask = u & mask, u >> n
    characters = []
    for qubit in range(n):
        local = ((x_mask >> qubit) & 1) | (((z_mask >> qubit) & 1) << 1)
        characters.append(("I", "X", "Z", "Y")[local])
    return "".join(characters)


def extend_code(
    code: Sequence[int], q_unsigned: int, sign: int, n: int
) -> Tuple[int, ...]:
    """Adjoin an independent commuting signed generator to a full subgroup.

    The caller ensures independence and commutation. The resulting full
    subgroup is sorted so subsequent construction is independent of ordering.
    """
    signed_generator = q_unsigned | ((sign & 1) << (2 * n))
    result = list(code) + [multiply(n, element, signed_generator) for element in code]
    result.sort()
    return tuple(result)


def canonical_key(code: Sequence[int], n: int) -> Tuple[int, ...]:
    """Return a deterministic signed-generator key for a full subgroup.

    Binary elimination produces a unique reduced basis for the unsigned
    stabilizer span, scanning the highest bit first. Each resulting row's
    sign is recovered from the supplied full subgroup. Recovering signs this
    way includes the Pauli multiplication phases rather than XORing signs.
    """
    unsigned_count = 1 << (2 * n)
    unsigned_mask = unsigned_count - 1
    signs = [-1] * unsigned_count
    rows = []
    for element in code:
        unsigned = element & unsigned_mask
        signs[unsigned] = (element >> (2 * n)) & 1
        if unsigned:
            rows.append(unsigned)

    rank = len(code).bit_length() - 1
    row = 0
    for column in range(2 * n - 1, -1, -1):
        if row >= rank:
            break
        pivot = next(
            (i for i in range(row, len(rows)) if (rows[i] >> column) & 1),
            -1,
        )
        if pivot < 0:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        pivot_row = rows[row]
        for i in range(len(rows)):
            if i != row and ((rows[i] >> column) & 1):
                rows[i] ^= pivot_row
        row += 1
    rows = rows[:rank]
    return tuple(unsigned | (signs[unsigned] << (2 * n)) for unsigned in rows)


def code_from_key(key: Sequence[int], n: int) -> Tuple[int, ...]:
    """Expand independent commuting signed generators into their full group."""
    code = (enc(n, 0, 0, 0),)
    unsigned_mask = (1 << (2 * n)) - 1
    for generator in key:
        unsigned = generator & unsigned_mask
        sign = (generator >> (2 * n)) & 1
        code = extend_code(code, unsigned, sign, n)
    return code


def _insert_basis(basis: List[int], v: int, width: int) -> bool:
    """Insert a binary vector into an in-place echelon basis if independent.

    ``basis[bit]`` stores the vector with that highest pivot bit, or zero when
    no such vector is present. Return whether the span increased.
    """
    for bit in range(width - 1, -1, -1):
        if (v >> bit) & 1:
            if basis[bit]:
                v ^= basis[bit]
            else:
                basis[bit] = v
                return True
    return False


def _nullspace_basis(rows: List[int], width: int) -> List[int]:
    """Return a binary nullspace basis for bit-packed linear constraints."""
    rows = rows[:]
    row_count = len(rows)
    row = 0
    pivots = []
    for column in range(width - 1, -1, -1):
        if row >= row_count:
            break
        pivot = next(
            (i for i in range(row, row_count) if (rows[i] >> column) & 1),
            -1,
        )
        if pivot < 0:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        for i in range(row_count):
            if i != row and ((rows[i] >> column) & 1):
                rows[i] ^= rows[row]
        pivots.append(column)
        row += 1

    rows = rows[:row]
    pivot_set = set(pivots)
    nullspace = []
    for free_column in range(width):
        if free_column in pivot_set:
            continue
        vector = 1 << free_column
        for i, pivot_column in enumerate(pivots):
            if parity(rows[i] & vector):
                vector |= 1 << pivot_column
        nullspace.append(vector)
    return nullspace


def logical_representatives(key: Sequence[int], n: int) -> List[int]:
    """Enumerate one unsigned Pauli per nonzero logical coset S-perp / S.

    For stabilizer rank r there are exactly ``4**(n-r) - 1`` choices. Each
    commutes with the stabilizer and is not fixed on its code space. Two
    representatives need not commute: they are alternative next measurements.
    """
    unsigned_mask = (1 << (2 * n)) - 1
    qubit_mask = (1 << n) - 1
    stabilizers = [generator & unsigned_mask for generator in key]

    # Swap x/z halves: ordinary binary dot products with these rows compute
    # the symplectic commutation constraints on a candidate Pauli.
    constraints = []
    for unsigned in stabilizers:
        x_mask, z_mask = unsigned & qubit_mask, unsigned >> n
        constraints.append(z_mask | (x_mask << n))
    centralizer = _nullspace_basis(constraints, 2 * n)

    # Extend a basis of S to one of S-perp. The added vectors span a direct
    # complement and therefore select unique representatives of the quotient.
    span_basis = [0] * (2 * n)
    for unsigned in stabilizers:
        _insert_basis(span_basis, unsigned, 2 * n)
    complement = []
    for vector in centralizer:
        if _insert_basis(span_basis, vector, 2 * n):
            complement.append(vector)

    representatives = []
    for selection in range(1, 1 << len(complement)):
        representative = 0
        for i, vector in enumerate(complement):
            if (selection >> i) & 1:
                representative ^= vector
        representatives.append(representative)
    return representatives
