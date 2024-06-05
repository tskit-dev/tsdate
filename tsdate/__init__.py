# MIT License
#
# Copyright (c) 2020 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from .cache import *  # NOQA: F401,F403
from .core import date  # NOQA: F401
from .core import estimation_methods  # NOQA: F401
from .core import inside_outside  # NOQA: F401
from .core import maximization  # NOQA: F401
from .core import variational_gamma  # NOQA: F401
from .prior import parameter_grid as build_parameter_grid  # NOQA: F401
from .prior import prior_grid as build_prior_grid  # NOQA: F401
from .provenance import __version__  # NOQA: F401
from .util import add_sampledata_times  # NOQA: F401
from .util import preprocess_ts  # NOQA: F401
from .util import sites_time_from_ts  # NOQA: F401

# Bit 20 is set in node flags when they are samples not at time zero in the sampledata
# file. This should match the node flag in tsinfer.

#: Node flag value indicating that this is a non-contemporary sample node
NODE_IS_HISTORICAL_SAMPLE = 1 << 20


# Since tsdate is often used together with tsinfer, we try not to use the tsinfer
# node flag constants here, and start from 1 << 30 rather than 1 << 16

#: Node flag value indicating that this was a disjoint node that was then split
NODE_SPLIT_BY_PREPROCESS = 1 << 30
