"""
The error codes of the C library that the oracles raise (``lib/err.h``, without the
``TSD_ERR_`` prefix), and the exceptions with which ``_tsdate`` reports them.
"""

import _tsdate

UNPHASED_INDIVIDUAL_EDGES = -16
BLOCK_OUT_OF_BOUNDS = -17
PHASE_OUT_OF_RANGE = -18
UNPHASED_LIKELIHOOD_NOT_CONSERVED = -19
MUTATION_NODE_NOT_IN_TREE = -20
FIXED_NODES_VIOLATE_CONSTRAINT = -21


def library_error(code):
    """
    The exception with which ``_tsdate`` reports error ``code``.
    """
    message = _tsdate.strerror(code)
    return _tsdate.LibraryError(message)
