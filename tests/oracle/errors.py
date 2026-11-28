"""
The error codes of the C library (``lib/err.h``, without the ``TSD_ERR_`` prefix),
and the exceptions with which ``_tsdate`` reports them.
"""

import _tsdate

GENERIC = -1
NO_MEMORY = -2
BAD_PARAM = -3
ZERO_DIVISION = -4
INVALID_HYPERGEO = -5
NONPOSITIVE_OR_NONFINITE_MOMENTS = -6
JENSEN_VIOLATION = -7
KL_MAX_ITERATIONS = -8
KL_INVALID_SHAPE = -9
NONPOSITIVE_CENTRAL_MOMENTS = -10
UNSORTED_QUANTILES = -11
QUANTILE_MAX_ITERATIONS = -12
NEGATIVE_SHAPE = -13
NONPOSITIVE_MOMENTS = -14
INVALID_FIXED_AGE = -15
UNPHASED_INDIVIDUAL_EDGES = -16
BLOCK_OUT_OF_BOUNDS = -17
PHASE_OUT_OF_RANGE = -18
UNPHASED_LIKELIHOOD_NOT_CONSERVED = -19
MUTATION_NODE_NOT_IN_TREE = -20
FIXED_NODES_VIOLATE_CONSTRAINT = -21
NONPOSITIVE_INTERVALS = -22
ZERO_EDGE_SPAN = -24
NON_INCREASING_BREAKPOINTS = -25
INVALID_QUANTILE_WIDTH = -26
INVALID_POSTERIOR = -27
TIME_BELOW_FIRST_BREAKPOINT = -28
NO_POSTERIORS = -29
INVALID_MAX_SHAPE = -30
INVALID_MIN_STEP = -31
INVALID_NODE_POSTERIOR = -32
INVALID_DAMPING = -33
TWIN_EDGE_IN_PHASED_PASS = -34
NONPOSITIVE_PENALTY = -35
NODE_OUT_OF_BOUNDS = -36
EDGE_OUT_OF_BOUNDS = -37
INDIVIDUAL_OUT_OF_BOUNDS = -38
BAD_EDGE_INTERVAL = -39
BAD_EDGE_INDEXES = -40
EDGES_FORM_CYCLE = -41
TOO_FEW_TIPS = -42
NO_BREAKPOINTS = -43

# -23 is unused
CODES = [code for code in range(NO_BREAKPOINTS, 0) if code != -23]


def library_error(code):
    """
    The exception with which ``_tsdate`` reports error ``code``.
    """
    message = _tsdate.strerror(code)
    return _tsdate.LibraryError(message)


def code_of(exception):
    """
    The error code of a ``_tsdate.LibraryError``, from its message.
    """
    message = str(exception)
    (code,) = [code for code in CODES if _tsdate.strerror(code) == message]
    return code
