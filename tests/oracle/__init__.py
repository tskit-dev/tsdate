"""
Plain-Python reference implementations ("oracles") of the tree-sweep functions of the
C extension ``_tsdate``.

Each function has an oracle in the module of the same name here, e.g.
``oracle.phasing.block_singletons`` for ``_tsdate.block_singletons``. Equivalence
tests (``tests/test_oracle_*.py``) call the ``_tsdate`` functions and the oracles with
the same arguments and compare the outcomes.

Rules:

* An oracle is written as array code in plain Python with the operation order of the
  C library.
* Oracles call other oracles, never ``_tsdate``, except for the messages of the
  library's error codes (``errors``, as ``_tsdate.strerror`` gives them).
* Oracles raise errors in the library's terms: where the C library returns an error
  code, the oracle raises ``errors.library_error(code)``, the ``_tsdate.LibraryError``
  with that code's message; for the shapes and sizes of arrays that ``_tsdate``
  checks, the oracle raises ``ValueError`` with its message. The equivalence tests
  compare the type and message of the exceptions. Internal invariants, which the
  library does not report, are plain ``assert`` statements.
* Names, arguments and results are those of ``_tsdate``'s functions.
"""
