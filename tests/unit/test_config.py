# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from deadreckoning.config import Config, ConfigError, load_config

BASE = '[node]\nnode_id = "n"\ndata_dir = "./d"\n'


def _load(tmp_path: Path, text: str) -> Config:
    path = tmp_path / "dr.toml"
    path.write_text(text)
    return load_config(path)


def test_the_shipped_example_validates(config: Config) -> None:
    assert config.node.node_id == "truck-7"
    assert [t.rank for t in config.tiers] == [0, 1, 2]
    assert config.task_classes["dispatch"].min_rank == 1


def test_an_unknown_key_is_an_error_not_a_warning(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="canary_intervals_s"):
        _load(tmp_path, BASE + "[health]\ncanary_intervals_s = 15\n")


def test_the_error_names_section_and_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _load(tmp_path, BASE + "[sync]\npage_size = 0\n")
    assert "sync.page_size" in str(caught.value)


def test_an_inline_secret_is_refused_with_a_usable_message(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="secrets must never appear"):
        _load(
            tmp_path,
            BASE + '[[dependencies]]\nname = "h"\ntype = "SYNC_HUB"\ntoken = "hunter2"\n',
        )


def test_naming_the_environment_variable_is_accepted(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        BASE + '[[dependencies]]\nname = "h"\ntype = "SYNC_HUB"\ntoken_env = "DR_HUB_TOKEN"\n',
    )
    assert config.dependencies[0].token_env == "DR_HUB_TOKEN"


def test_chaos_cannot_be_enabled_in_production(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="production"):
        _load(tmp_path, 'profile = "production"\n' + BASE + "[chaos]\nenabled = true\n")


def test_chaos_is_allowed_in_the_demo_profile(tmp_path: Path) -> None:
    assert _load(tmp_path, BASE + "[chaos]\nenabled = true\n").chaos.enabled


def test_two_tiers_cannot_claim_one_rank(tmp_path: Path) -> None:
    tier = '[[tiers]]\nname = "{}"\nrank = 0\nkind = "local"\nmodel = "m"\nbase_url = "u"\n'
    with pytest.raises(ConfigError, match="rank 0 is claimed"):
        _load(tmp_path, BASE + tier.format("a") + tier.format("b"))


def test_a_task_class_floor_no_tier_can_meet_is_refused(tmp_path: Path) -> None:
    # This is the shape of defect that would otherwise surface only at runtime,
    # in the field, as an unexplained escalation.
    with pytest.raises(ConfigError, match="cannot be met by any declared tier"):
        _load(
            tmp_path,
            BASE
            + '[[tiers]]\nname = "a"\nrank = 2\nkind = "local"\nmodel = "m"\nbase_url = "u"\n'
            + "[task_classes.dispatch]\nmin_rank = 1\nreview_above_rank = 0\n",
        )


def test_a_tier_that_is_not_scripted_needs_an_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="base_url is required"):
        _load(tmp_path, BASE + '[[tiers]]\nname = "a"\nrank = 0\nkind = "remote"\nmodel = "m"\n')


def test_a_dependency_may_not_shadow_a_tier_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="already a tier name"):
        _load(
            tmp_path,
            BASE
            + '[[tiers]]\nname = "frontier"\nrank = 0\nkind = "local"\n'
            + 'model = "m"\nbase_url = "u"\n'
            + '[[dependencies]]\nname = "frontier"\ntype = "TOOL_BACKEND"\n',
        )


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no such configuration file"):
        load_config(tmp_path / "absent.toml")


def test_broken_toml_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        _load(tmp_path, "this is not = = toml")


def test_configured_redact_keys_extend_the_defaults(tmp_path: Path) -> None:
    config = _load(tmp_path, BASE + '[redaction]\nkeys = ["pin"]\n')
    assert "pin" in config.redact_keys()
    assert "authorization" in config.redact_keys()


def test_a_tier_that_cannot_be_probed_is_refused(tmp_path: Path) -> None:
    """A tier with no canary is a deadlock, not a saving.

    It starts UNKNOWN, the router will only pick a tier that is HEALTHY or SLOW,
    so it is never called and never observed, so it stays UNKNOWN. It also blocks
    the whole node from reaching CONNECTED, because that needs every declared
    dependency healthy.
    """
    with pytest.raises(ConfigError, match="cannot be probed"):
        _load(
            tmp_path,
            BASE + '[[tiers]]\nname = "a"\nrank = 0\nkind = "local"\nmodel = "m"\n'
            'base_url = "u"\ncanary = false\n',
        )


def test_a_scripted_tier_needs_no_canary(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        BASE + '[[tiers]]\nname = "s"\nrank = 0\nkind = "scripted"\nmodel = "m"\ncanary = false\n',
    )
    assert config.tiers[0].canary is False
