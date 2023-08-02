# MIT License
#
# Copyright (c) 2020 University of Oxford
# Copyright (c) 2021-2023 Tskit Developers
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
Routines and classes for manipulating demographic histories in tsdate
"""
import numpy as np
import scipy.stats


class PopulationSizeHistory:
    """
    Stores a piecewise constant population size history and tranforms time from
    a natural (generational) scale to a coalescent one
    """

    @staticmethod
    def _change_time_measure(time_ago, breakpoints, time_measure):
        """
        Rescales time given a piecewise-constant time measure. To convert from
        generations to coalescent units, the time measure per epoch should be 2 *
        effective population size.  To convert from coalescent units to
        generations, the time measure should be the coalescent rate ``1/(2 * Ne)``.

        :param np.ndarray time_ago: An increasing vector of time points
        :param np.ndarray breakpoints: Start times of pieces
        :param np.ndarray time_measure: Time measure within pieces

        :return: Inputs in new time measure
        """

        assert np.all(np.diff(breakpoints) > 0.0)
        assert np.min(breakpoints) == 0.0
        assert np.all(time_ago >= 0.0)
        assert np.all(time_measure > 0.0)
        assert breakpoints.size == time_measure.size
        index = np.searchsorted(breakpoints, time_ago, side="right") - 1
        step = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    breakpoints[1:] * (1.0 / time_measure[:-1] - 1.0 / time_measure[1:])
                ),
            ]
        )
        new_time_ago = time_ago * 1.0 / time_measure[index] + step[index]
        new_breakpoints = breakpoints * 1.0 / time_measure + step
        new_time_measure = 1.0 / time_measure
        return new_time_ago, new_breakpoints, new_time_measure

    def __init__(self, population_size, time_breaks=None):
        """
        :param array_like population_size: An array containing diploid
            population sizes per epoch
        :param array_like time_breaks: A sorted array containing time
            breaks that divide epochs, measured in units of generations in the
            past
        """

        if time_breaks is None:
            time_breaks = []

        if isinstance(population_size, (int, float)):
            population_size = np.array([population_size], dtype=float)
        else:
            try:
                population_size = np.array(population_size, dtype=float)
            except (ValueError, TypeError) as e:
                raise e.__class__(
                    "Population sizes must be convertable to a numpy float array"
                ) from e
        if not np.all(population_size > 0.0):
            raise ValueError("Population sizes must be greater than 0")
        if not np.all(np.isfinite(population_size)):
            raise ValueError("Population sizes must be finite")

        try:
            time_breaks = np.array(time_breaks, dtype=float)
        except (ValueError, TypeError) as e:
            raise e.__class__(
                "Time breaks must be convertable to a numpy float array"
            ) from e
        if not time_breaks.size == population_size.size - 1:
            raise ValueError(
                "The length of the population size array must be one less "
                "than the number of epoch time breaks"
            )
        if time_breaks.size > 0:
            if not np.all(time_breaks > 0.0):
                raise ValueError("Epoch time breaks must be greater than 0")
            if not np.all(np.diff(time_breaks) > 0.0):
                raise ValueError(
                    "Epoch time breaks must be unique and in increasing order"
                )

        self.time_breaks = np.append([0.0], time_breaks.flatten())
        self.population_size = 2 * population_size.flatten()
        _, coalescent_breaks, coalescent_rate = self._change_time_measure(
            self.time_breaks, self.time_breaks, self.population_size
        )
        self.coalescent_breaks = coalescent_breaks
        self.coalescent_rate = coalescent_rate

    def as_dict(self):
        """
        Return the population size history as a dictionary of parameters
        that can be used to initialise a new object
        """
        ret_val = {"population_size": list(self.population_size / 2)}
        assert self.time_breaks[0] == 0.0
        if len(self.time_breaks) > 1:
            ret_val["time_breaks"] = list(self.time_breaks[1:])
        return ret_val

    def to_natural_timescale(self, coalescent_time_ago):
        """
        Convert a vector of times from coalescent units to generations

        :param np.ndarray coalescent_time_ago: Times in the past, in coalescent units
        :return: Times in the past, in generations
        """

        if not isinstance(coalescent_time_ago, np.ndarray):
            raise ValueError("Times must be in a numpy array")
        time_ago, _, _ = self._change_time_measure(
            coalescent_time_ago,
            self.coalescent_breaks,
            self.coalescent_rate,
        )
        return time_ago

    def to_coalescent_timescale(self, time_ago):
        """
        Convert a vector of times from generations to coalescent units

        :param np.ndarray time_ago: Times in the past, in generations
        :return: Times in the past, in coalescent units
        """

        if not isinstance(time_ago, np.ndarray):
            raise ValueError("Times must be in a numpy array")
        coalescent_time_ago, _, _ = self._change_time_measure(
            time_ago,
            self.time_breaks,
            self.population_size,
        )
        return coalescent_time_ago

    # TODO: multiprecision implementation -- remove at some point

    # @staticmethod
    # def _Gamma(z, a=0, b=np.inf):
    #    """
    #    log |Re| of generalized incomplete gamma function with bounds `a,b`
    #    and argument `z`
    #    """
    #    val = mpmath.log(mpmath.gammainc(z, a=a, b=b))
    #    return float(val)

    # def to_gamma(self, shape=1, rate=1):
    #    """
    #    Given a gamma distribution on a coalescent timescale with parameters
    #    `shape` and `rate`, return a gamma approximation to the distribution
    #    under a change of measure to a generational timescale.

    #    :param float shape: Shape parameter of gamma
    #    :param float rate: Rate parameter of gamma (inverse of scale)
    #    :return: Shape and rate parameters after change of measure
    #    """
    #    assert shape > 0, "Gamma shape parameter must be positive"
    #    assert rate > 0, "Gamma rate parameter must be positive"
    #    C = shape * np.log(rate) - self._Gamma(shape)
    #    mn = 0.0
    #    va = 0.0
    #    for a, b, n, t in zip(
    #        self.coalescent_breaks,
    #        np.append(self.coalescent_breaks[1:], [np.inf]),
    #        self.population_size,
    #        self.time_breaks,
    #    ):
    #        gamma = [
    #            np.exp(
    #                self._Gamma(shape + i, rate * a, rate * b)
    #                - (shape + i) * np.log(rate)
    #                + C
    #            )
    #            for i in range(3)
    #        ]
    #        mn += (t - n * a) * gamma[0] + n * gamma[1]
    #        va += (
    #            gamma[0] * (t - n * a) ** 2
    #            + 2 * gamma[1] * n * (t - n * a)
    #            + gamma[2] * n**2
    #        )
    #    va -= mn**2
    #    new_shape = mn**2 / va
    #    new_rate = mn / va
    #    return new_shape, new_rate

    def to_gamma(self, shape=1, rate=1):
        """
        Given a gamma distribution on a coalescent timescale with parameters
        `shape` and `rate`, return a gamma approximation to the distribution
        under a change of measure to a generational timescale.

        :param float shape: Shape parameter of gamma
        :param float rate: Rate parameter of gamma (inverse of scale)
        :return: Shape and rate parameters after change of measure
        """
        assert shape > 0, "Gamma shape parameter must be positive"
        assert rate > 0, "Gamma rate parameter must be positive"
        C = np.exp(shape * np.log(rate) - scipy.special.loggamma(shape))
        gamma_cdf = scipy.special.gammainc
        cdf_breaks = np.append(self.coalescent_breaks, [np.inf])
        cdf_0 = (
            C
            * scipy.special.gamma(shape)
            / rate**shape
            * np.diff(gamma_cdf(shape + 0, rate * cdf_breaks))
        )
        mn_coef_0 = self.time_breaks - self.population_size * self.coalescent_breaks
        va_coef_0 = mn_coef_0**2
        cdf_1 = (
            C
            * scipy.special.gamma(shape + 1)
            / rate ** (shape + 1)
            * np.diff(gamma_cdf(shape + 1, rate * cdf_breaks))
        )
        mn_coef_1 = self.population_size
        va_coef_1 = mn_coef_0 * mn_coef_1 * 2
        cdf_2 = (
            C
            * scipy.special.gamma(shape + 2)
            / rate ** (shape + 2)
            * np.diff(gamma_cdf(shape + 2, rate * cdf_breaks))
        )
        va_coef_2 = mn_coef_1**2
        mn = np.sum(mn_coef_1 * cdf_1 + mn_coef_0 * cdf_0)
        va = np.sum(va_coef_2 * cdf_2 + va_coef_1 * cdf_1 + va_coef_0 * cdf_0)
        va -= mn**2
        new_shape = mn**2 / va
        new_rate = mn / va
        return new_shape, new_rate

    # TODO:
    # @staticmethod
    # def from_demes(filename):
    #    """
    #    Create a `PopulationSizeHistory` instance from a `demes` format YAML
    #    """

    # TODO:
    # output a PopulationSizeHistory as a json object, so we can store it
    # in the provenance. See https://github.com/tskit-dev/tsdate/issues/274
