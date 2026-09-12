# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from deadreckoning.canonical import (
    CanonicalisationError,
    canonical_json,
    content_hash,
    redact,
)


def test_key_order_does_not_change_the_bytes() -> None:
    a = {"zebra": 1, "alpha": {"y": 2, "x": 3}, "middle": [1, 2]}
    b = {"middle": [1, 2], "alpha": {"x": 3, "y": 2}, "zebra": 1}
    assert canonical_json(a) == canonical_json(b)
    assert content_hash(a) == content_hash(b)


def test_no_insignificant_whitespace() -> None:
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_unicode_is_utf8_not_escaped() -> None:
    assert canonical_json({"k": "café"}) == '{"k":"café"}'.encode()


def test_float_uses_repr_and_round_trips() -> None:
    value = 0.1 + 0.2
    encoded = canonical_json({"x": value})
    assert encoded == b'{"x":0.30000000000000004}'
    assert json.loads(encoded)["x"] == value


def test_list_order_is_significant() -> None:
    assert canonical_json([1, 2]) != canonical_json([2, 1])


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_floats_are_refused(bad: float) -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json({"x": bad})


def test_unknown_types_are_refused_rather_than_coerced() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json({"x": {1, 2}})


def test_non_string_keys_are_refused() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json({1: "a"})


def test_redaction_is_case_insensitive_and_recursive() -> None:
    out = redact({"Authorization": "Bearer x", "n": [{"TOKEN": "t"}]})
    assert out["Authorization"] == "[REDACTED]"
    assert out["n"][0]["TOKEN"] == "[REDACTED]"


def test_redaction_keeps_environment_variable_names() -> None:
    # api_key_env holds the NAME of a variable, not a secret. Substring matching
    # would destroy it and make the config unreadable.
    assert redact({"api_key_env": "FRONTIER_API_KEY"})["api_key_env"] == "FRONTIER_API_KEY"


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.text()
    | st.floats(allow_nan=False, allow_infinity=False),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=12,
)


@given(json_values)
def test_canonical_form_round_trips(value: object) -> None:
    assert json.loads(canonical_json(value)) == value


@given(json_values)
def test_canonical_form_is_stable(value: object) -> None:
    assert canonical_json(value) == canonical_json(json.loads(canonical_json(value)))
