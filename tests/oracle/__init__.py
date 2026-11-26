"""
Plain-Python reference implementations ("oracles") of the functions of the C
extension ``_tsdate``.

Each function has an oracle in the module of the same name here, e.g.
``oracle.approx.moments``. Equivalence tests compare the C implementation
(``oracle.implementations.C``) against the oracle.

Rules:

* An oracle is written as array code in plain Python with the operation order of the
  C library. Where plain Python semantics differ from C (e.g. division by zero
  raising instead of giving inf/nan, numpy scalar semantics), a comment explains the
  compensation, once in the module header where it applies throughout.
* Oracles call other oracles, never the C implementation.
* Names and signatures are those of ``tsdate``'s functions of the same names, where
  they exist.
"""
