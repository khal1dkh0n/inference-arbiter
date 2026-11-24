from __future__ import annotations

import hashlib
import os

import pytest
import yaml

from inference_arbiter.config import REQUIRED_TIERS, ConfigError, load_config, load_env_file

from conftest import REAL_CONFIG_PATH


def write_config(tmp_path, config_dict):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    return config_path


def test_shipped_config_is_valid():
    config = load_config(REAL_CONFIG_PATH)
    for tier_name in REQUIRED_TIERS:
        assert tier_name in config.tiers
    assert config.confidence_threshold == 0.80
    assert config.fallback_tier == "mid"
    assert config.escalation == {"local": "mid", "code": "mid", "mid": "frontier"}
    assert config.tiers["mid"].model.endswith(":cloud")
    assert config.tiers["mid"].provider == "ollama"
    assert config.tiers["mid"].cloud is True
    assert config.tiers["frontier"].provider == "openrouter"
    assert config.providers["openrouter"].api_key_env == "OPENROUTER_API_KEY"


def test_shipped_config_pins_context_for_every_local_tier():
    config = load_config(REAL_CONFIG_PATH)
    for tier_name, tier in config.tiers.items():
        if tier.cloud:
            continue
        assert "-ctx" in tier.model or tier.num_ctx is not None, tier_name


def test_sha256_matches_file_bytes():
    config = load_config(REAL_CONFIG_PATH)
    assert config.sha256 == hashlib.sha256(REAL_CONFIG_PATH.read_bytes()).hexdigest()


def test_model_swap_needs_only_a_yaml_edit(tmp_path, base_config_dict):
    base_config_dict["tiers"]["local"]["model"] = "some-other-model:7b"
    config = load_config(write_config(tmp_path, base_config_dict))
    assert config.tiers["local"].model == "some-other-model:7b"


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml")


def test_unknown_provider_is_rejected(tmp_path, base_config_dict):
    base_config_dict["tiers"]["local"]["provider"] = "nowhere"
    with pytest.raises(ConfigError, match="unknown provider"):
        load_config(write_config(tmp_path, base_config_dict))


def test_missing_required_tier_is_rejected(tmp_path, base_config_dict):
    del base_config_dict["tiers"]["code"]
    base_config_dict["escalation"].pop("code")
    with pytest.raises(ConfigError, match="missing required tier"):
        load_config(write_config(tmp_path, base_config_dict))


def test_on_error_must_point_at_another_answer_tier(tmp_path, base_config_dict):
    base_config_dict["tiers"]["local"]["on_error"] = "local"
    with pytest.raises(ConfigError, match="points at itself"):
        load_config(write_config(tmp_path, base_config_dict))

    base_config_dict["tiers"]["local"]["on_error"] = "triage"
    with pytest.raises(ConfigError, match="cannot be a triage tier"):
        load_config(write_config(tmp_path, base_config_dict))


def test_unknown_key_is_rejected(tmp_path, base_config_dict):
    base_config_dict["tiers"]["local"]["api_key"] = "should-never-be-here"
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, base_config_dict))


def test_threshold_out_of_range_is_rejected(tmp_path, base_config_dict):
    base_config_dict["confidence_threshold"] = 1.5
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, base_config_dict))


def test_openrouter_provider_requires_key_name(tmp_path, base_config_dict):
    del base_config_dict["providers"]["openrouter"]["api_key_env"]
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(write_config(tmp_path, base_config_dict))


def test_env_loader_sets_names_without_overriding(tmp_path, monkeypatch):
    monkeypatch.setenv("QAH_TEST_EXISTING", "from-shell")
    monkeypatch.delenv("QAH_TEST_NEW", raising=False)
    monkeypatch.delenv("QAH_TEST_QUOTED", raising=False)
    monkeypatch.delenv("QAH_TEST_BLANK", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\n"
        "QAH_TEST_EXISTING=from-file\n"
        "QAH_TEST_NEW=plain-value\n"
        'export QAH_TEST_QUOTED="quoted value"\n'
        "QAH_TEST_BLANK=\n"
        "not a key value line\n",
        encoding="utf-8",
    )

    loaded_names = load_env_file(env_path)

    assert loaded_names == ["QAH_TEST_NEW", "QAH_TEST_QUOTED"]
    assert os.environ["QAH_TEST_EXISTING"] == "from-shell"
    assert os.environ["QAH_TEST_NEW"] == "plain-value"
    assert os.environ["QAH_TEST_QUOTED"] == "quoted value"
    assert "QAH_TEST_BLANK" not in os.environ
    monkeypatch.delenv("QAH_TEST_NEW")
    monkeypatch.delenv("QAH_TEST_QUOTED")


def test_env_loader_ignores_missing_file(tmp_path):
    assert load_env_file(tmp_path / "absent.env") == []


def test_model_placeholder_is_filled_from_the_environment(tmp_path, base_config_dict, monkeypatch):
    base_config_dict["tiers"]["frontier"]["model"] = "${QAH_TEST_FRONTIER_MODEL}"
    monkeypatch.setenv("QAH_TEST_FRONTIER_MODEL", "vendor/frontier-model")

    config = load_config(write_config(tmp_path, base_config_dict))

    assert config.tiers["frontier"].model == "vendor/frontier-model"


def test_model_placeholder_stays_when_the_variable_is_unset_or_empty(tmp_path, base_config_dict, monkeypatch):
    base_config_dict["tiers"]["frontier"]["model"] = "${QAH_TEST_FRONTIER_MODEL}"
    monkeypatch.delenv("QAH_TEST_FRONTIER_MODEL", raising=False)
    assert load_config(write_config(tmp_path, base_config_dict)).tiers["frontier"].model == "${QAH_TEST_FRONTIER_MODEL}"

    monkeypatch.setenv("QAH_TEST_FRONTIER_MODEL", "")
    assert load_config(write_config(tmp_path, base_config_dict)).tiers["frontier"].model == "${QAH_TEST_FRONTIER_MODEL}"


def test_shipped_config_takes_the_frontier_model_from_the_environment(monkeypatch):
    monkeypatch.setenv("FRONTIER_MODEL", "vendor/frontier-model")
    assert load_config(REAL_CONFIG_PATH).tiers["frontier"].model == "vendor/frontier-model"

    monkeypatch.delenv("FRONTIER_MODEL")
    assert load_config(REAL_CONFIG_PATH).tiers["frontier"].model == "${FRONTIER_MODEL}"


def test_shipped_config_writes_solution_and_tests_on_two_different_free_models():
    config = load_config(REAL_CONFIG_PATH)
    code_test = config.code_test

    assert (code_test.interface_tier, code_test.solution_tier, code_test.tests_tier) == ("local", "mid", "local")
    assert config.tiers[code_test.solution_tier].model != config.tiers[code_test.tests_tier].model
    for tier_name in (code_test.interface_tier, code_test.solution_tier, code_test.tests_tier):
        assert config.tiers[tier_name].provider == "ollama", "codetest mode makes no paid call as shipped"
    assert code_test.max_repairs == 1
    assert code_test.max_program_bytes > config.execute.max_program_bytes
