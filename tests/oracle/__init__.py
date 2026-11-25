"""
Plain-Python reference implementations ("oracles") of the functions of the C
extension ``_tsdate``, made from the numba code that the extension replaced.

Each function has an oracle in the module of the same name here, e.g.
``oracle.approx.moments``. Equivalence tests compare the C implementation
(``oracle.implementations.C``) against the oracle.

Rules:

* An oracle is the numba source with the decorators stripped. Change the body only
  where plain Python differs from compiled numba (e.g. division by zero raising
  instead of giving inf/nan, numpy scalar semantics), and put a comment on each change.
* Oracles call other oracles, never the C implementation.
* Names and signatures match the numba functions they were made from.
"""
