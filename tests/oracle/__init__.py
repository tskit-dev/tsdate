"""
Plain-Python reference implementations ("oracles") of the functions of the C
extension ``_tsdate``.

Each function has an oracle in the module of the same name here, e.g.
``oracle.approx.moments`` for ``_tsdate.moments``. Equivalence tests
(``tests/test_oracle_*.py``) call the ``_tsdate`` functions and the oracles with the
same arguments and compare the outcomes.

Rules:

* An oracle is written as array code in plain Python with the operation order of the
  C library. Where plain Python semantics differ from C (e.g. division by zero
  raising instead of giving inf/nan, numpy scalar semantics), a comment explains the
  compensation, once in the module header where it applies throughout.
* Oracles call other oracles, never ``_tsdate``, except for ``gammaincinv``
  (``hypergeo.gammainc_inv``), which they take from the C library as they take libm,
  and which is compared with SciPy instead, and the messages of the library's error
  codes (``errors``, as ``_tsdate.strerror`` gives them).
* Oracles raise errors in the library's terms: where the C library returns an error
  code, the oracle raises ``errors.library_error(code)``, the ``_tsdate.LibraryError``
  with that code's message; for the shapes and sizes of arrays that ``_tsdate``
  checks, the oracle raises ``ValueError`` with its message. The equivalence tests
  compare the type and message of the exceptions. A ``ZeroDivisionError`` (from a
  Python float division) stands for the library's ``TSD_ERR_ZERO_DIVISION``. Internal
  invariants, which the library does not report, are plain ``assert`` statements.
* Names, arguments and results are those of ``_tsdate``'s functions: the same
  argument lists, and results of the same types and layout (floats where ``_tsdate``
  returns floats, the same tuples). The EP state is the tuple of factor arrays that
  ``_tsdate`` takes.
"""
