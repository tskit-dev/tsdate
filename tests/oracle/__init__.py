"""
Plain-Python reference implementations ("oracles") of tsdate's jitted functions.

Each function that is compiled with numba (or later ported to C) has an oracle in
the module of the same name here, e.g. ``oracle.approx.moments``. Equivalence tests
compare every implementation listed in ``oracle.implementations.IMPLEMENTATIONS``
against the oracle.

Rules:

* An oracle is the numba source with the decorators stripped. Change the body only
  where plain Python differs from compiled numba (e.g. division by zero raising
  instead of giving inf/nan, numpy scalar semantics), and put a comment on each change.
* Oracles call other oracles, never tsdate's numba functions.
* Names and signatures match the tsdate functions exactly.
"""
