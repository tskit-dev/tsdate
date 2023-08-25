# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for metadata setting functionality in tsdate.
"""
import json
import logging

import numpy as np
import pytest
import tskit
import utility_functions

from tsdate.core import date
from tsdate.metadata import node_md_struct
from tsdate.metadata import save_node_metadata

struct_obj_only_example = tskit.MetadataSchema(
    {
        "codec": "struct",
        "type": "object",
        "properties": {
            "node_id": {"type": "integer", "binaryFormat": "i"},
        },
        "additionalProperties": False,
    }
)

struct_bad_mn = tskit.MetadataSchema(
    {
        "codec": "struct",
        "type": "object",
        "properties": {
            "mn": {"type": "integer", "binaryFormat": "i"},
        },
        "additionalProperties": False,
    }
)

struct_bad_vr = tskit.MetadataSchema(
    {
        "codec": "struct",
        "type": "object",
        "properties": {
            "vr": {"type": "string", "binaryFormat": "10p"},
        },
        "additionalProperties": False,
    }
)


class TestBytes:
    """
    Tests for when existing node metadata is in raw bytes
    """

    def test_no_existing(self):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        assert ts.node(root).metadata == b""
        assert ts.table_metadata_schemas.node == tskit.MetadataSchema(None)
        ts = date(ts, mutation_rate=1, population_size=1)
        assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
        assert ts.node(root).metadata["vr"] > 0

    def test_append_existing(self):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        assert ts.table_metadata_schemas.node == tskit.MetadataSchema(None)
        tables = ts.dump_tables()
        tables.nodes.clear()
        for nd in ts.nodes():
            tables.nodes.append(nd.replace(metadata=b'{"node_id": %d}' % nd.id))
        ts = tables.tree_sequence()
        assert json.loads(ts.node(root).metadata.decode())["node_id"] == root
        ts = date(ts, mutation_rate=1, population_size=1)
        assert ts.node(root).metadata["node_id"] == root
        assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
        assert ts.node(root).metadata["vr"] > 0

    def test_replace_existing(self):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        assert ts.table_metadata_schemas.node == tskit.MetadataSchema(None)
        tables = ts.dump_tables()
        tables.nodes.clear()
        for nd in ts.nodes():
            tables.nodes.append(nd.replace(metadata=b'{"mn": 1.0}'))
        ts = tables.tree_sequence()
        assert json.loads(ts.node(root).metadata.decode())["mn"] == pytest.approx(1.0)
        ts = date(ts, mutation_rate=1, population_size=1)
        assert ts.node(root).metadata["mn"] != pytest.approx(1.0)
        assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
        assert ts.node(root).metadata["vr"] > 0

    def test_existing_bad(self):
        ts = utility_functions.single_tree_ts_n2()
        assert ts.table_metadata_schemas.node == tskit.MetadataSchema(None)
        tables = ts.dump_tables()
        tables.nodes.clear()
        for nd in ts.nodes():
            tables.nodes.append(nd.replace(metadata=b"!!"))
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="Cannot modify"):
            date(ts, mutation_rate=1, population_size=1)

    def test_erase_existing_bad(self, caplog):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        assert ts.table_metadata_schemas.node == tskit.MetadataSchema(None)
        tables = ts.dump_tables()
        tables.nodes.clear()
        for nd in ts.nodes():
            tables.nodes.append(nd.replace(metadata=b"!!"))
        ts = tables.tree_sequence()
        # Should be able to replace using set_metadat=True
        with caplog.at_level(logging.WARNING):
            ts = date(ts, mutation_rate=1, population_size=1, set_metadata=True)
            assert "Erasing existing node metadata" in caplog.text
        assert ts.table_metadata_schemas.node.schema["codec"] == "struct"
        assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
        assert ts.node(root).metadata["vr"] > 0


class TestStruct:
    """
    Tests for when existing node metadata is as a struct
    """

    def test_append_existing(self):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        tables = ts.dump_tables()
        tables.nodes.metadata_schema = struct_obj_only_example
        tables.nodes.packset_metadata(
            [
                tables.nodes.metadata_schema.validate_and_encode_row({"node_id": i})
                for i in range(ts.num_nodes)
            ]
        )
        ts = tables.tree_sequence()
        assert ts.node(root).metadata["node_id"] == root
        ts = date(ts, mutation_rate=1, population_size=1)
        assert ts.node(root).metadata["node_id"] == root
        assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
        assert ts.node(root).metadata["vr"] > 0

    def test_replace_existing(self, caplog):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        tables = ts.dump_tables()
        tables.nodes.metadata_schema = node_md_struct
        tables.nodes.packset_metadata(
            [
                tables.nodes.metadata_schema.validate_and_encode_row(None)
                for _ in range(ts.num_nodes)
            ]
        )
        ts = tables.tree_sequence()
        assert ts.node(root).metadata is None
        with caplog.at_level(logging.INFO):
            ts = date(ts, mutation_rate=1, population_size=1)
            assert ts.table_metadata_schemas.node.schema["codec"] == "struct"
            assert "Replacing 'mn'" in caplog.text
            assert "Replacing 'vr'" in caplog.text
            assert "Schema modified" in caplog.text
            assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
            assert ts.node(root).metadata["vr"] > 0
            sample = ts.samples()[0]
            assert ts.node(sample).metadata is None

    def test_existing_bad_mn(self, caplog):
        ts = utility_functions.single_tree_ts_n2()
        tables = ts.dump_tables()
        tables.nodes.metadata_schema = struct_bad_mn
        tables.nodes.packset_metadata(
            [
                tables.nodes.metadata_schema.validate_and_encode_row({"mn": 1})
                for _ in range(ts.num_nodes)
            ]
        )
        ts = tables.tree_sequence()
        with pytest.raises(
            ValueError, match=r"Cannot change type of node.metadata\['mn'\]"
        ):
            date(ts, mutation_rate=1, population_size=1)

    def test_existing_bad_vr(self, caplog):
        ts = utility_functions.single_tree_ts_n2()
        tables = ts.dump_tables()
        tables.nodes.metadata_schema = struct_bad_vr
        tables.nodes.packset_metadata(
            [
                tables.nodes.metadata_schema.validate_and_encode_row({"vr": "foo"})
                for _ in range(ts.num_nodes)
            ]
        )
        ts = tables.tree_sequence()
        with pytest.raises(
            ValueError, match=r"Cannot change type of node.metadata\['vr'\]"
        ):
            date(ts, mutation_rate=1, population_size=1)


class TestJson:
    """
    Tests for when existing node metadata is json encoded
    """

    def test_replace_existing(self, caplog):
        ts = utility_functions.single_tree_ts_n2()
        root = ts.first().root
        tables = ts.dump_tables()
        schema = tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.nodes.packset_metadata(
            [
                schema.validate_and_encode_row(
                    {f"node {i}": 1, "mn": "foo", "vr": "bar"}
                )
                for i in range(ts.num_nodes)
            ]
        )
        ts = tables.tree_sequence()
        assert "node 0" in ts.node(0).metadata
        assert ts.node(0).metadata["mn"] == "foo"
        with caplog.at_level(logging.INFO):
            ts = date(ts, mutation_rate=1, population_size=1)
            assert ts.table_metadata_schemas.node.schema["codec"] == "json"
            assert "Schema modified" in caplog.text
            sample = ts.samples()[0]
            assert f"node {sample}" in ts.node(sample).metadata
            # Should have deleted mn and vr
            assert "mn" not in ts.node(sample).metadata
            assert "vr" not in ts.node(sample).metadata
            assert f"node {root}" in ts.node(root).metadata
            assert ts.node(root).metadata["mn"] == pytest.approx(ts.nodes_time[root])
            assert ts.node(root).metadata["vr"] > 0


class TestNoSetMetadata:
    """
    Tests for when metadata is not saved
    """

    @pytest.mark.parametrize(
        "method", ["inside_outside", "maximization", "variational_gamma"]
    )
    def test_empty(self, method):
        ts = utility_functions.single_tree_ts_n2()
        assert len(ts.tables.nodes.metadata) == 0
        ts = date(
            ts, mutation_rate=1, population_size=1, method=method, set_metadata=False
        )
        assert len(ts.tables.nodes.metadata) == 0

    @pytest.mark.parametrize(
        "method", ["inside_outside", "maximization", "variational_gamma"]
    )
    def test_random_md(self, method):
        ts = utility_functions.single_tree_ts_n2()
        assert len(ts.tables.nodes.metadata) == 0
        tables = ts.dump_tables()
        tables.nodes.packset_metadata([(b"random %i" % u) for u in range(ts.num_nodes)])
        ts = tables.tree_sequence()
        assert len(ts.tables.nodes.metadata) > 0
        dts = date(
            ts, mutation_rate=1, population_size=1, method=method, set_metadata=False
        )
        assert len(ts.tables.nodes.metadata) == len(dts.tables.nodes.metadata)


class TestFunctions:
    """
    Test internal metadata functions
    """

    def test_bad_save_node_metadata(self):
        ts = utility_functions.single_tree_ts_n2()
        bad_arr = np.zeros(ts.num_nodes + 1)
        good_arr = np.zeros(ts.num_nodes)
        for m, v in ([bad_arr, good_arr], [good_arr, bad_arr]):
            with pytest.raises(ValueError, match="arrays of length ts.num_nodes"):
                save_node_metadata(ts, m, v, fixed_node_set=set(ts.samples()))
