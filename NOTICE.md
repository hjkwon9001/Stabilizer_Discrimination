# Source provenance

This standalone project derives from the user's Stabilizer Discrimination
Suite and its revised Python library. The Python code contains the exact
arithmetic, Pauli algebra, ensemble definitions and Bellman optimizer. Native
C++ implementations are included as source. No earlier output CSVs,
probability catalogues, saved root lists or executable binaries are included.

The constants defining the quantum ensembles and the exact arithmetic
denominators are inputs to the calculations. They are not stored answers.
Known small values in tests and documentation are checks; they are not used
to produce optimizer results.

All-root native execution removes the dependency on saved symmetry-reduced
first-measurement lists. This trades speed for a fully enumerated root search.
The original native source still exposes additional low-level options when
invoked directly; the Python interface enforces a fresh complete run.

The E8 XOR native solver has been generalized to one through four copies.
It uses multiword stabilizer keys, factored Pauli coefficients, and the
Python library's qubit-label ordering. Search limits abort without reporting
an optimum. Three- and four-copy full searches have not been completed as
part of validation; small cases, coefficients, key representations, and
bounded searches are tested independently.

The four-state Clifford+Toffoli script checks a supplied circuit; it does not
implement a four-T injection protocol.

No authors, copyright holder, citation identifier or license have been
invented. Add the authors' chosen license and citation before public release.
