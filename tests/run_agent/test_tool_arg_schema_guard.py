"""Tests for stripping hallucinated tool-call arguments before dispatch.

Some frontier models (reported for Opus 4.8 and Sonnet 5 circa mid-2026 by
Armin Ronacher / Simon Willison, "Better Models, Worse Tools") add extra
fields to tool-call arguments for custom (non-native) tool schemas -- fields
that were never declared in the tool's JSON Schema ``parameters.properties``.
Handlers that unpack ``**args`` into a strict function signature then raise
``TypeError: unexpected keyword argument``, silently breaking the tool call.

strip_unrecognized_tool_args() drops unknown top-level arguments only for
simple schemas that explicitly close the object with
``"additionalProperties": false``. Open, composed, referenced, or
pattern-based schemas pass through unchanged so valid arguments are never
silently discarded.
"""

from unittest.mock import patch

import pytest

from model_tools import strip_unrecognized_tool_args


class TestStripUnrecognizedToolArgs:
    """Unit tests for strip_unrecognized_tool_args."""

    def _mock_schema(self, properties, additional_properties: object = False):
        parameters = {"type": "object", "properties": properties}
        if additional_properties is not None:
            parameters["additionalProperties"] = additional_properties
        return {
            "name": "test_tool",
            "description": "test",
            "parameters": parameters,
        }

    def test_all_known_keys_pass_through_unchanged(self):
        schema = self._mock_schema({"path": {"type": "string"}, "limit": {"type": "integer"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"path": "/tmp/file.txt", "limit": 10}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == {"path": "/tmp/file.txt", "limit": 10}

    def test_strips_single_hallucinated_field(self):
        schema = self._mock_schema({"path": {"type": "string"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"path": "/tmp/file.txt", "recursive_depth": 3}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == {"path": "/tmp/file.txt"}
            assert "recursive_depth" not in result

    def test_strips_multiple_hallucinated_fields(self):
        schema = self._mock_schema({"query": {"type": "string"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"query": "hello", "max_results": 5, "sort_order": "desc"}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == {"query": "hello"}

    def test_respects_additional_properties_true(self):
        """Tools that explicitly declare additionalProperties: true (e.g. the
        raw CDP bridge or the generic MCP tool_call bridge) must not be
        filtered -- freeform args are the intended contract there."""
        schema = self._mock_schema({"name": {"type": "string"}}, additional_properties=True)
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"name": "click", "x": 100, "y": 200}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == {"name": "click", "x": 100, "y": 200}

    def test_respects_schema_valued_additional_properties(self):
        schema = self._mock_schema(
            {"name": {"type": "string"}},
            additional_properties={"type": "string"},
        )
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"name": "click", "selector": "#submit"}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == args

    def test_open_schema_without_additional_properties_is_not_filtered(self):
        schema = self._mock_schema({"name": {"type": "string"}}, None)
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"name": "click", "selector": "#submit"}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == args

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("allOf", [{"properties": {"extra": {"type": "string"}}}]),
            ("anyOf", [{"properties": {"extra": {"type": "string"}}}]),
            ("oneOf", [{"properties": {"extra": {"type": "string"}}}]),
            ("$ref", "#/$defs/toolArgs"),
            ("patternProperties", {"^x-": {"type": "string"}}),
        ],
    )
    def test_composed_or_pattern_schema_is_not_filtered(self, keyword, value):
        schema = self._mock_schema({"name": {"type": "string"}})
        schema["parameters"][keyword] = value
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"name": "click", "extra": "valid-via-schema"}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == args

    def test_empty_properties_schema_strips_all_hallucinated_args(self):
        """A zero-arg tool (properties: {}) still must not silently accept
        invented fields."""
        schema = self._mock_schema({})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"unexpected": "value"}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == {}

    def test_unknown_tool_returns_args_unchanged(self):
        with patch("model_tools.registry.get_schema", return_value=None):
            args = {"anything": "goes"}
            result = strip_unrecognized_tool_args("unknown_tool", args)
            assert result == {"anything": "goes"}

    def test_schema_missing_parameters_key_is_noop(self):
        schema = {"name": "test_tool", "description": "test"}
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"anything": "goes"}
            result = strip_unrecognized_tool_args("test_tool", args)
            assert result == {"anything": "goes"}

    def test_empty_args_returns_unchanged(self):
        schema = self._mock_schema({"path": {"type": "string"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            result = strip_unrecognized_tool_args("test_tool", {})
            assert result == {}

    def test_non_dict_args_passthrough(self):
        schema = self._mock_schema({"path": {"type": "string"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            assert strip_unrecognized_tool_args("test_tool", None) is None

    def test_logs_warning_with_tool_name_and_stripped_keys(self, caplog):
        import logging

        schema = self._mock_schema({"path": {"type": "string"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            with caplog.at_level(logging.WARNING, logger="model_tools"):
                strip_unrecognized_tool_args("read_file", {"path": "/tmp/x", "bogus_field": 1})
        assert any(
            "read_file" in record.message and "bogus_field" in record.message
            for record in caplog.records
        )

    def test_real_open_read_file_schema_is_not_silently_closed(self):
        """Missing additionalProperties means open under JSON Schema semantics."""
        args = {"file_path": "/tmp/x", "hallucinated_extra_param": True}
        result = strip_unrecognized_tool_args("read_file", args)
        assert result == args
