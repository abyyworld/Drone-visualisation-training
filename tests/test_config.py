"""The configuration schema: defaults, strictness, and the unsafe combinations.

Two of these matter more than the rest.

*Unknown keys are rejected.* A typo in a config key on a laptop at an incident
would otherwise leave a safety-relevant default silently in place --
``max_overlay_age_s`` misspelt means stale boxes over live video, and nothing
anywhere would say so.

*``overlay_buffer_s`` must be at least ``max_overlay_age_s``.* A buffer shorter
than the staleness limit throws away payloads the tablet is still allowed to
draw, so the overlay would blink out while the pipeline was perfectly healthy.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from station.core.config import (
    Config,
    IncidentLogConfig,
    InferenceConfig,
    SourceConfig,
    StreamConfig,
    TemporalConfig,
    load_config,
    validate,
)


@pytest.fixture
def write_config(tmp_path: Path):
    """Write a YAML config file and return its path."""

    def _write(text: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path

    return _write


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------


class TestDefaults:
    def test_load_config_with_no_path_returns_defaults(self):
        cfg = load_config()
        assert cfg.source == SourceConfig()
        assert cfg.inference == InferenceConfig()
        assert cfg.temporal == TemporalConfig()
        assert cfg.stream == StreamConfig()
        assert cfg.incident_log == IncidentLogConfig()

    def test_source_defaults(self):
        s = SourceConfig()
        assert (s.type, s.uri, s.reconnect_s, s.target_fps, s.loop) == ("file", "", 2.0, None, False)

    def test_inference_defaults(self):
        i = InferenceConfig()
        assert i.imgsz == 640
        # Deliberately low: recall matters more than precision here, and the
        # temporal filter (not this threshold) suppresses the resulting flicker.
        assert i.conf_threshold == 0.25
        assert i.iou_nms == 0.45
        assert i.device == "auto"
        assert i.max_fps == 10.0

    def test_temporal_defaults_are_three_of_five(self):
        t = TemporalConfig()
        assert (t.n, t.m) == (3, 5)
        assert t.iou_match == 0.30
        assert t.max_age == 5
        assert t.emit_unconfirmed is False

    def test_stream_defaults(self):
        s = StreamConfig()
        assert s.port == 8443
        assert s.max_overlay_age_s == 1.0
        assert s.overlay_buffer_s == 3.0
        assert s.status_interval_s == 1.0
        assert s.stall_after_s == 3.0

    def test_ice_servers_default_to_an_explicit_empty_list(self):
        # Empty and not None: an explicit empty list is what turns off
        # aiortc's default public STUN server, which on a LAN with no
        # internet only adds a timeout to every connection.
        a, b = StreamConfig(), StreamConfig()
        assert a.ice_servers == [] and b.ice_servers == []
        a.ice_servers.append("stun:example")
        assert b.ice_servers == [], "ice_servers is shared between instances"

    def test_incident_log_defaults_to_enabled(self):
        # A station that records nothing by default would leave the
        # false-negative audit in docs/VALIDATION.md with no data at all.
        i = IncidentLogConfig()
        assert i.enabled is True
        assert i.record_video is True
        assert i.flush_every == 1
        assert i.dir == "incidents"

    def test_config_sections_are_independent_between_instances(self):
        a, b = Config(), Config()
        a.temporal.n = 1
        assert b.temporal.n == 3


# --------------------------------------------------------------------------
# loading YAML
# --------------------------------------------------------------------------


class TestLoading:
    def test_loads_a_partial_file_and_keeps_other_defaults(self, write_config):
        cfg = load_config(write_config(
            """
            source:
              type: rtsp
              uri: rtsp://192.168.144.25:8554/main.264
            """
        ))
        assert cfg.source.type == "rtsp"
        assert cfg.source.uri == "rtsp://192.168.144.25:8554/main.264"
        assert cfg.temporal == TemporalConfig()  # untouched section keeps defaults

    def test_an_empty_file_is_all_defaults(self, write_config):
        cfg = load_config(write_config("\n"))
        assert cfg.source == SourceConfig()

    def test_a_null_section_is_that_section_s_defaults(self, write_config):
        cfg = load_config(write_config("temporal:\ninference:\n"))
        assert cfg.temporal == TemporalConfig()
        assert cfg.inference == InferenceConfig()

    def test_a_non_mapping_top_level_is_rejected(self, write_config):
        with pytest.raises(TypeError, match="top level"):
            load_config(write_config("- one\n- two\n"))

    def test_a_non_mapping_section_is_rejected(self, write_config):
        with pytest.raises(TypeError, match="expected a mapping"):
            load_config(write_config("temporal: 5\n"))

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does-not-exist.yaml")

    def test_the_shipped_example_config_loads(self, repo_root):
        # config.example.yaml is what an operator copies. If it does not load,
        # every deployment starts with an error.
        cfg = load_config(repo_root / "config.example.yaml")
        assert isinstance(cfg, Config)
        validate(cfg)

    def test_the_shipped_example_config_lists_every_field(self, repo_root):
        """Every schema field must appear in the example, and nothing else.

        Loading it already proves there are no *extra* keys (unknown keys are
        rejected); this proves there are no *missing* ones, so an operator
        reading the example sees the whole surface -- including the safety
        knobs they would otherwise never learn exist.
        """
        import yaml
        from dataclasses import fields

        raw = yaml.safe_load((repo_root / "config.example.yaml").read_text(encoding="utf-8"))
        defaults = Config()
        for f in fields(Config):
            if f.name == "station_name":
                assert f.name in raw
                continue
            section = raw.get(f.name)
            assert isinstance(section, dict), f"section {f.name} missing from config.example.yaml"
            # `f.type` is a string under `from __future__ import annotations`,
            # so introspect the constructed default rather than the annotation.
            expected = {sf.name for sf in fields(getattr(defaults, f.name))}
            assert set(section) == expected, f"{f.name}: {expected ^ set(section)}"


class TestOverrides:
    def test_dotted_overrides_reach_sections(self):
        cfg = load_config(None, {"temporal.n": 2, "stream.port": 9000})
        assert cfg.temporal.n == 2
        assert cfg.stream.port == 9000

    def test_undotted_overrides_reach_the_root(self):
        assert load_config(None, {"station_name": "engine 41"}).station_name == "engine 41"

    def test_overrides_win_over_the_file(self, write_config):
        path = write_config("temporal:\n  n: 4\n  m: 5\n")
        assert load_config(path, {"temporal.n": 2}).temporal.n == 2

    def test_an_override_is_validated_like_anything_else(self):
        with pytest.raises(ValueError, match="max_overlay_age_s"):
            load_config(None, {"stream.max_overlay_age_s": 0})

    def test_an_unknown_override_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(None, {"stream.max_overlay_age": 2.0})


# --------------------------------------------------------------------------
# strictness: unknown keys
# --------------------------------------------------------------------------


class TestUnknownKeysAreRejected:
    @pytest.mark.parametrize(
        "section,key",
        [
            ("source", "fps"),
            ("inference", "confidence"),
            ("temporal", "min_hits"),
            ("stream", "max_overlay_age"),      # the misspelling that matters
            ("incident_log", "path"),
        ],
    )
    def test_a_typo_in_any_section_fails_loudly(self, write_config, section, key):
        path = write_config(f"{section}:\n  {key}: 1\n")
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(path)

    def test_the_error_names_the_key_and_the_valid_ones(self, write_config):
        path = write_config("stream:\n  max_overlay_age: 2.0\n")
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        message = str(excinfo.value)
        assert "max_overlay_age" in message
        assert "max_overlay_age_s" in message, "the error must show the correct spelling"
        assert "stream" in message

    def test_several_unknown_keys_are_all_reported(self, write_config):
        path = write_config("temporal:\n  nn: 1\n  mm: 2\n")
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        assert "'mm'" in str(excinfo.value) and "'nn'" in str(excinfo.value)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG in the frozen contract file station/core/config.py: unknown keys are "
            "rejected *inside* a section but not at the top level. load_config reads only "
            "the five section names it knows via raw.get(...), so a misspelt section header "
            "-- `strem:` for `stream:`, `temperal:` for `temporal:` -- is silently ignored "
            "and that whole section reverts to defaults. That is exactly the failure "
            "_build's own docstring says must not happen: `max_overlay_age_s` misspelt "
            "means stale boxes over live video, and here the entire stream section can "
            "vanish without a word. Fix: in load_config, check "
            "`set(raw) - {'station_name', 'source', 'inference', 'temporal', 'stream', "
            "'incident_log'}` and raise. Not fixed here because station/core/ is the "
            "committed contract and this agent was told not to edit it."
        ),
    )
    def test_an_unknown_top_level_section_is_rejected(self, write_config):
        path = write_config("streaming:\n  port: 1\n")
        with pytest.raises((ValueError, TypeError)):
            load_config(path)

    def test_a_misspelt_section_header_silently_loses_its_settings(self, write_config):
        """Documents the live consequence of the bug above.

        Kept as a passing test rather than folded into the xfail so the damage
        is visible in the suite: this config *looks* like it sets a two-second
        staleness limit and does not.
        """
        cfg = load_config(write_config("strem:\n  max_overlay_age_s: 2.0\n"))
        assert cfg.stream.max_overlay_age_s == 1.0  # the default, silently


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


class TestValidation:
    def test_overlay_buffer_shorter_than_the_age_limit_is_rejected(self):
        cfg = Config()
        cfg.stream.max_overlay_age_s = 2.0
        cfg.stream.overlay_buffer_s = 1.0
        with pytest.raises(ValueError) as excinfo:
            validate(cfg)
        message = str(excinfo.value)
        assert "overlay_buffer_s" in message and "max_overlay_age_s" in message
        # The reason must survive into the message: this discards payloads the
        # tablet is still allowed to draw.
        assert "discards" in message

    def test_overlay_buffer_equal_to_the_age_limit_is_allowed(self):
        cfg = Config()
        cfg.stream.max_overlay_age_s = 1.5
        cfg.stream.overlay_buffer_s = 1.5
        validate(cfg)  # must not raise

    @pytest.mark.parametrize("age", [0.0, -1.0])
    def test_non_positive_overlay_age_is_rejected(self, age):
        # Zero would disable the staleness rule entirely, which is the single
        # most dangerous configuration this file can express.
        cfg = Config()
        cfg.stream.max_overlay_age_s = age
        cfg.stream.overlay_buffer_s = 3.0
        with pytest.raises(ValueError, match="stale boxes render over live video"):
            validate(cfg)

    @pytest.mark.parametrize("n,m", [(6, 5), (0, 5), (-1, 3), (3, 2)])
    def test_bad_n_of_m_is_rejected(self, n, m):
        cfg = Config()
        cfg.temporal.n, cfg.temporal.m = n, m
        with pytest.raises(ValueError, match="1 <= n <= m"):
            validate(cfg)

    @pytest.mark.parametrize("n,m", [(1, 1), (3, 5), (5, 5), (1, 20)])
    def test_valid_n_of_m_passes(self, n, m):
        cfg = Config()
        cfg.temporal.n, cfg.temporal.m = n, m
        validate(cfg)

    @pytest.mark.parametrize("iou", [0.0, 1.0, -0.1, 1.1])
    def test_iou_match_outside_the_open_unit_interval_is_rejected(self, iou):
        cfg = Config()
        cfg.temporal.iou_match = iou
        with pytest.raises(ValueError, match="iou_match"):
            validate(cfg)

    @pytest.mark.parametrize("conf", [0.0, 1.0, -0.5, 2.0])
    def test_conf_threshold_outside_the_open_unit_interval_is_rejected(self, conf):
        cfg = Config()
        cfg.inference.conf_threshold = conf
        with pytest.raises(ValueError, match="conf_threshold"):
            validate(cfg)

    @pytest.mark.parametrize("fps", [0.0, -1.0])
    def test_non_positive_max_fps_is_rejected(self, fps):
        cfg = Config()
        cfg.inference.max_fps = fps
        with pytest.raises(ValueError, match="max_fps"):
            validate(cfg)

    def test_validate_accepts_the_shipped_defaults(self):
        validate(Config())

    def test_load_config_runs_validate(self, write_config):
        # Validation must not be an optional second step somebody forgets.
        path = write_config("stream:\n  overlay_buffer_s: 0.5\n  max_overlay_age_s: 1.0\n")
        with pytest.raises(ValueError, match="overlay_buffer_s"):
            load_config(path)


class TestConfigValuesReachTheirConsumers:
    """A config field nothing reads is a field that silently does nothing."""

    def test_temporal_config_drives_the_filter(self):
        from station.inference.temporal import TemporalFilter

        cfg = TemporalConfig(n=2, m=4, iou_match=0.5, max_age=1)
        filt = TemporalFilter(cfg)
        assert filt.cfg is cfg
        assert repr(filt).count("n=2") == 1

    def test_the_filter_rejects_what_validate_rejects(self):
        from station.inference.temporal import TemporalFilter

        # A TemporalConfig can be built directly, bypassing load_config, so
        # the filter re-checks rather than trusting its caller.
        with pytest.raises(ValueError):
            TemporalFilter(TemporalConfig(n=9, m=5))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG in the frozen contract file station/core/config.py: load_config defaults "
        "station_name with `raw.get('station_name', Config.station_name)`. Config is a "
        "slots=True dataclass, so `Config.station_name` is the slot *descriptor*, not the "
        "default string. Any config omitting station_name yields a <member ...> object, "
        "which then lands in the certificate common name, /config.json, the CLI banner and "
        "every incident meta.json. The one-line fix is "
        "`Config.__dataclass_fields__['station_name'].default`. Not fixed here because "
        "station/core/ is the committed contract and this agent was told not to edit it; "
        "station.cli._repair_station_name works around it for CLI callers only."
    ),
)
def test_station_name_defaults_to_the_dataclass_default():
    assert load_config().station_name == Config().station_name
    assert isinstance(load_config().station_name, str)


def test_station_name_from_a_file_is_correct(write_config):
    # The bug above only bites when the key is absent; an explicit value works.
    assert load_config(write_config("station_name: engine 41\n")).station_name == "engine 41"
