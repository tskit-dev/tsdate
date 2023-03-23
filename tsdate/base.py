# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Base classes and constants used by tsdate
"""
import logging

import numpy as np


FLOAT_DTYPE = np.float64
LIN = "linear"
LOG = "logarithmic"
PAR = "parameter"

# Bit 20 is set in node flags when they are samples not at time zero in the sampledata
# file
NODE_IS_HISTORIC_SAMPLE = 1 << 20


class NodeGridValues:
    """
    A class to store times or discretised distributions of times for node ids. For nodes
    with fixed times, only a single time value needs to be stored. For non-fixed nodes,
    an array of len(timepoints) probabilies is required.

    :ivar num_nodes: The number of nodes that will be stored in this object
    :vartype num_nodes: int
    :ivar nonfixed_nodes: a (possibly empty) numpy array of unique positive node ids each
        of which must be less than num_nodes. Each will have an array of grid_size
        associated with it. All others (up to num_nodes) will be associated with a single
        scalar value instead.
    :vartype nonfixed_nodes: numpy.ndarray
    :ivar timepoints: Array of time points
    :vartype timepoints: numpy.ndarray
    :ivar fill_value: What should we fill the data arrays with to start with
    :vartype fill_value: numpy.scalar
    """

    def __init__(
        self,
        num_nodes,
        nonfixed_nodes,
        timepoints,
        fill_value=np.nan,
        dtype=FLOAT_DTYPE,
    ):
        """
        :param numpy.ndarray grid: The input numpy.ndarray.
        """
        if nonfixed_nodes.ndim != 1:
            raise ValueError("nonfixed_nodes must be a 1D numpy array")
        if np.any((nonfixed_nodes < 0) | (nonfixed_nodes >= num_nodes)):
            raise ValueError(
                "All non fixed node ids must be between zero and the total node number"
            )
        grid_size = len(timepoints) if type(timepoints) is np.ndarray else timepoints
        self.timepoints = timepoints
        # Make timepoints immutable so no risk of overwritting them with copy
        self.timepoints.setflags(write=False)
        self.num_nodes = num_nodes
        self.nonfixed_nodes = nonfixed_nodes
        self.num_nonfixed = len(nonfixed_nodes)
        self.grid_data = np.full(
            (self.num_nonfixed, grid_size), fill_value, dtype=dtype
        )
        self.fixed_data = np.full(
            num_nodes - self.num_nonfixed, fill_value, dtype=dtype
        )
        self.row_lookup = np.empty(num_nodes, dtype=np.int64)
        # non-fixed nodes get a positive value, indicating lookup in the grid_data array
        self.row_lookup[nonfixed_nodes] = np.arange(self.num_nonfixed)
        # fixed nodes get a negative value from -1, indicating lookup in the scalar array
        self.row_lookup[
            np.logical_not(np.isin(np.arange(num_nodes), nonfixed_nodes))
        ] = (-np.arange(num_nodes - self.num_nonfixed) - 1)
        self.probability_space = LIN

    def force_probability_space(self, probability_space):
        """
        probability_space can be "logarithmic" or "linear": this function will force
        the current probability space to the desired type
        """
        descr = (
            self.probability_space,
            " probabilities into",
            probability_space,
            "space",
        )
        if probability_space == LIN:
            if self.probability_space == LIN:
                pass
            elif self.probability_space == LOG:
                self.grid_data = np.exp(self.grid_data)
                self.fixed_data = np.exp(self.fixed_data)
                self.probability_space = LIN
            else:
                logging.warning("Cannot force", *descr)
        elif probability_space == LOG:
            if self.probability_space == LOG:
                pass
            elif self.probability_space == LIN:
                with np.errstate(divide="ignore", invalid="ignore"):
                    self.grid_data = np.log(self.grid_data)
                    self.fixed_data = np.log(self.fixed_data)
                self.probability_space = LOG
            else:
                logging.warning("Cannot force", *descr)
        elif probability_space == PAR:
            if self.probability_space == PAR:
                pass
            else:
                logging.warning("Cannot force", *descr)
        else:
            logging.warning("Cannot force", *descr)

    def standardize(self):
        """
        Standardize grid data so the max for each row is one (in linear space) or zero
        (in logarithmic space)

        TODO - is it clear why we omit the first element of the
        """
        rowmax = self.grid_data[:, 1:].max(axis=1)
        if self.probability_space == LIN:
            self.grid_data = self.grid_data / rowmax[:, np.newaxis]
        elif self.probability_space == LOG:
            self.grid_data = self.grid_data - rowmax[:, np.newaxis]
        else:
            raise RuntimeError("Probability space is not", LIN, "or", LOG)

    def to_probabilities(self):
        """
        Change grid data into probabilities (i.e. each row sums to one in linear or zero
        in logarithmic space)
        """
        if self.probability_space != LIN:
            raise NotImplementedError(
                "Can only convert to probabilities in linear space"
            )
        assert not np.any(self.grid_data < 0)
        self.grid_data = self.grid_data / self.grid_data.sum(axis=1)[:, np.newaxis]

    def __getitem__(self, node_id):
        index = self.row_lookup[node_id]
        if index < 0:
            return self.fixed_data[1 + index]
        else:
            return self.grid_data[index, :]

    def __setitem__(self, node_id, value):
        index = self.row_lookup[node_id]
        if index < 0:
            self.fixed_data[1 + index] = value
        else:
            self.grid_data[index, :] = value

    def clone_with_new_data(
        self, grid_data=np.nan, fixed_data=None, probability_space=None
    ):
        """
        Take the row indices etc from an existing NodeGridValues object and make a new
        similar one but with different data. If grid_data is a single number, fill the
        entire data array with that, otherwise assume the data is a numpy array of the
        correct size to fill the gridded data. If grid_data is None, fill with NaN

        If fixed_data is None and grid_data is a single number, use the same value as
        grid_data for the fixed data values. If fixed_data is None and grid_data is an
        array, set the fixed data to np.nan
        """

        def fill_fixed(orig, fixed_data):
            if type(fixed_data) is np.ndarray:
                if orig.fixed_data.shape != fixed_data.shape:
                    raise ValueError(
                        "The fixed data array must be the same shape as the original"
                    )
                return fixed_data
            else:
                return np.full(
                    orig.fixed_data.shape, fixed_data, dtype=orig.fixed_data.dtype
                )

        new_obj = NodeGridValues.__new__(NodeGridValues)
        new_obj.num_nodes = self.num_nodes
        new_obj.nonfixed_nodes = self.nonfixed_nodes
        new_obj.num_nonfixed = self.num_nonfixed
        new_obj.row_lookup = self.row_lookup
        new_obj.timepoints = self.timepoints
        if type(grid_data) is np.ndarray:
            if self.grid_data.shape != grid_data.shape:
                raise ValueError(
                    "The grid data array must be the same shape as the original"
                )
            new_obj.grid_data = grid_data
            new_obj.fixed_data = fill_fixed(
                self, np.nan if fixed_data is None else fixed_data
            )
        else:
            if grid_data == 0:  # Fast allocation
                new_obj.grid_data = np.zeros(
                    self.grid_data.shape, dtype=self.grid_data.dtype
                )
            else:
                new_obj.grid_data = np.full(
                    self.grid_data.shape, grid_data, dtype=self.grid_data.dtype
                )
            new_obj.fixed_data = fill_fixed(
                self, grid_data if fixed_data is None else fixed_data
            )
        if probability_space is None:
            new_obj.probability_space = self.probability_space
        else:
            new_obj.probability_space = probability_space
        return new_obj
