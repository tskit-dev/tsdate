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
Functions for setting or merging schemas in tsdate-generated tree sequences. Note that
tsdate will only add metadata to the node table, so this is the only relevant schema.
"""
import json
import logging

import tskit

MEAN_KEY = "mn"
VARIANCE_KEY = "vr"

node_md_mean = {
    "description": (
        "The mean time of this node, calculated from the tsdate posterior "
        "probabilities. This may not be the same as the node time, as it "
        "is not constrained by parent-child order."
    ),
    "type": "number",
    "binaryFormat": "d",
}

node_md_variance = {
    "description": (
        "The variance in times of this node, calculated from the tsdate "
        "posterior probabilities"
    ),
    "type": "number",
    "binaryFormat": "d",
}

node_md_struct = tskit.MetadataSchema(
    {
        "codec": "struct",
        "type": ["object", "null"],
        "default": None,
        "properties": {MEAN_KEY: node_md_mean, VARIANCE_KEY: node_md_variance},
        "additionalProperties": False,
    }
)


def set_tsdate_node_md_schema(ts, set_metadata=None):
    """
    Taken from the ``tsdate.date()`` docs:
        If set_metadata is ``True``, replace all existing node metadata with
        details of the times (means and variances) for each node. If ``False``,
        do not touch any existing metadata in the tree sequence. If ``None``
        (default), attempt to modify any existing node metadata to add the
        times (means and variances) for each node, overwriting only those specific
        metadata values. If no node metadata schema has been set, this will be possible
        only if either (a) the raw metadata can be decoded as JSON, in which case the
        schema is set to permissive_json or (b) no node metadata exists (in which
        case a default schema will be set), otherwise an error will be raised.

    Returns:
        A tuple of the tree sequence and a boolean indicating whether the
        metadata should be saved or not.
    """
    if set_metadata is not None and not set_metadata:
        logging.debug("Not setting node metadata")
        return ts, False

    tables = ts.dump_tables()
    if set_metadata:
        # Erase existing metadata, force schema to be node_md_struct
        if len(tables.nodes.metadata) != 0:
            logging.warning("Erasing existing node metadata")
            tables.nodes.packset_metadata([b"" for _ in range(tables.nodes.num_rows)])
        tables.nodes.metadata_schema = node_md_struct
        return tables.tree_sequence(), True

    # set_metadata is None: try to set or modify any existing metadata schema
    schema_object = node_md_struct
    if tables.nodes.metadata_schema == tskit.MetadataSchema(schema=None):
        if len(tables.nodes.metadata) > 0:
            # For backwards compatibility, if the node metadata is bytes (schema=None)
            # but can be decoded as JSON, we change the schema to permissive_json
            schema_object = tskit.MetadataSchema.permissive_json()
            try:
                for node in ts.nodes():
                    _ = json.loads(node.metadata.decode() or "{}")
            except json.JSONDecodeError:
                raise ValueError(
                    "Cannot modify node metadata if schema is "
                    "None and non-JSON node metadata already exists"
                )
            logging.info("Null schema now set to permissive_json")
    else:
        # Make new schema on basis of existing
        schema = tables.nodes.metadata_schema.schema
        if "properties" not in schema:
            schema["properties"] = {}
        prop = schema["properties"]
        for key, dfn in zip((MEAN_KEY, VARIANCE_KEY), (node_md_mean, node_md_variance)):
            if key in prop:
                if not prop[key].get("type") in {dfn["type"], None}:
                    raise ValueError(
                        f"Cannot change type of node.metadata['{key}'] in schema"
                    )
                else:
                    logging.info(f"Replacing '{key}' in existing node metadata schema")

            prop[key] = dfn.copy()
            if schema["codec"] == "struct":
                #  If we are adding to an existing struct codec, a "null" entry may
                #  not be allowed, so we need to add null defaults for the new fields
                logging.info("Adding NaN default to schema")
                prop[key]["default"] = float("NaN")

        # Repack, erasing old metadata present in the target keys
        schema_object = tskit.MetadataSchema(schema)
        metadata_array = []
        for node in ts.nodes():
            md = node.metadata
            for key in [MEAN_KEY, VARIANCE_KEY]:
                try:
                    del md[key]
                    logging.debug(f"Deleting existing '{key}' value in node metadata")
                except (KeyError, TypeError):
                    pass
            metadata_array.append(schema_object.validate_and_encode_row(md))
        tables.nodes.packset_metadata(metadata_array)
        logging.info("Schema modified")

    tables.nodes.metadata_schema = schema_object
    return tables.tree_sequence(), True


def save_node_metadata(ts, means, variances, fixed_node_set):
    """
    Assign means and variances (both arrays of length ts.num_nodes)
    to the node metadata and return the resulting tree sequence.

    Assumes that the metadata schema in the tree sequence allows
    MEAN_KEY and VARIANCE_KEY to be set to numbers
    """
    if len(means) != ts.num_nodes or len(variances) != ts.num_nodes:
        raise ValueError("means and variances must be arrays of length ts.num_nodes")
    tables = ts.dump_tables()
    nodes = tables.nodes
    metadata_array = []
    for node, mean, var in zip(ts.nodes(), means, variances):
        md = node.metadata
        if node.id not in fixed_node_set:
            if md is None:
                md = {}
            md[MEAN_KEY] = mean
            md[VARIANCE_KEY] = var
        metadata_array.append(nodes.metadata_schema.validate_and_encode_row(md))
    nodes.packset_metadata(metadata_array)
    return tables.tree_sequence()
