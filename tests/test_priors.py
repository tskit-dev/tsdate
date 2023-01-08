# MIT License
#
# Copyright (c) 2023 Tskit Developers
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
Test cases for prior functionality used in tsdate
"""
from tsdate.prior import ConditionalCoalescentTimes
from tsdate.prior import PriorParams


class TestConditionalCoalescentTimes:
    def test_str(self):
        # Test how a CC times object is printed out (for debug purposes)
        priors = ConditionalCoalescentTimes(None, 2, "gamma")
        lines = str(priors).split("\n")
        assert len(lines) == 1
        assert "gamma" in lines[0]

        priors.add(2)
        lines = str(priors).split("\n")
        assert len(lines) == 5
        assert lines[1].startswith("2")
        for n in PriorParams._fields:
            assert n in lines[1]
        for i, line in enumerate(lines[2:]):
            assert line.startswith(f" {i} descendants")
            assert line.endswith("]")
