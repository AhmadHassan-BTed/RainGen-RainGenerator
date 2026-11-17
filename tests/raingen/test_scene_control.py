"""Scene and control layer: units, the rainscene schema, presets, the parameter manifest,
the Following/Adjusted/Locked graph, Tier-0 vibe macros, text and METAR parsing, the
timeline, wellness (modes, the Keep-it-soothing governor, the H.870 dose meter, the claims
linter), live-weather providers (network mocked) and OSC dispatch (no sockets except one
loopback test on an ephemeral port).

Reference values carry a one-line source comment. "Plan" means docs/MASTER_PLAN.md.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from raingen.live import providers as LP
from raingen.scene import manifest as mf
from raingen.scene import metar as M
from raingen.scene import textparse as tp
from raingen.scene import timeline as tl
from raingen.scene import units as U
from raingen.scene.graph import MAX_RATIO, MIN_RATIO, ParameterGraph
from raingen.scene.presets import PRESETS, list_presets, preset
from raingen.scene.schema import SCHEMA_VERSION, Keyframe, Scene, SceneObject
from raingen.scene.vibe import apply_vibe, intensity_from_scene, intensity_rate
from raingen.wellness.dose import (
    ALLOWANCE_PA2H,
    DoseMeter,
    _sos_gain,
    a_weighting_sos,
    dose_pa2h,
    hours_until,
)
from raingen.wellness.modes import (
    GOVERNOR_SETTINGS,
    CalmGovernor,
    apply_mode,
    canonical_mode,
)

SR = 48_000


def c_air(temp_c: float) -> float:
    """Speed of sound in dry air, c = 331.3 sqrt(1 + T/273.15) m/s (ISO 9613-1 ideal-gas form)."""
    return 331.3 * math.sqrt(1.0 + temp_c / 273.15)


def es_bolton(t: float) -> float:
    """Saturation vapour pressure over water, hPa (Bolton 1980), an independent Magnus fit."""
    return 6.112 * math.exp(17.67 * t / (t + 243.5))


# =====================================================================================
# Units
# =====================================================================================
class TestUnits:
    def test_exact_conversion_factors(self):
        assert U.KN_TO_MPS == 1852.0 / 3600.0                    # international knot: 1852 m per hour (exact)
        assert U.MPH_TO_MPS == pytest.approx(1609.344 / 3600.0, rel=1e-15)   # international mile 1609.344 m (exact)
        assert U.KMH_TO_MPS == pytest.approx(1 / 3.6, rel=1e-15)
        assert U.IN_TO_MM == 25.4 and U.FT_TO_M == 0.3048        # international inch / foot (1959, exact)
        assert U.MI_TO_KM == 1.609344 and U.NM_TO_KM == 1.852    # exact by definition
        assert U.INHG_TO_HPA == pytest.approx(33.86389, abs=2e-5)  # NIST SP 811: 1 inHg (conventional) = 3386.389 Pa
        assert U.MMHG_TO_HPA == pytest.approx(1.33322387415, rel=1e-12)  # NIST SP 811: 1 mmHg = 133.322 387 415 Pa

    @pytest.mark.parametrize("text, fn, expected, tol", [
        ("5 mm/h", U.parse_rate, 5.0, 1e-12),
        ("0.2 in/hr", U.parse_rate, 5.08, 1e-12),                          # 0.2 x 25.4 mm
        ("Beaufort 6", U.parse_speed, 0.836 * 6 ** 1.5, 1e-12),            # WMO fit v = 0.836 B^1.5 (Plan App. A)
        ("force 6", U.parse_speed, 0.836 * 6 ** 1.5, 1e-12),
        ("20 knots", U.parse_speed, 20 * 1852 / 3600, 1e-12),
        ("20 kt", U.parse_speed, 20 * 1852 / 3600, 1e-12),
        ("36 km/h", U.parse_speed, 10.0, 1e-12),
        ("10 m/s", U.parse_speed, 10.0, 1e-12),
        ("NW", U.parse_direction_deg, 315.0, 0.0),                          # 16-point compass, 22.5 deg steps
        ("WSW", U.parse_direction_deg, 247.5, 0.0),
        ("315°", U.parse_direction_deg, 315.0, 0.0),
        ("12 °C", U.parse_temperature_c, 12.0, 0.0),
        ("54 F", U.parse_temperature_c, (54 - 32) * 5 / 9, 1e-12),          # degF -> degC
        ("3 miles", U.parse_distance_km, 3 * 1.609344, 1e-12),
        ("1013 hPa", U.parse_pressure_hpa, 1013.0, 0.0),
        ("29.92 inHg", U.parse_pressure_hpa, 29.92 * 33.86389, 1e-3),        # ICAO standard altimeter = 1013.2 hPa
    ])
    def test_parse_examples(self, text, fn, expected, tol):
        assert fn(text) == pytest.approx(expected, abs=tol)

    def test_standard_atmosphere_altimeter(self):
        # ICAO ISA: 29.92 inHg and 1013.25 hPa are the same standard setting (to the 0.01 inHg shown)
        assert U.parse_pressure_hpa("29.92 inHg") == pytest.approx(1013.25, abs=0.34)

    @pytest.mark.parametrize("text", ["flash-to-bang 9 s", "9 s flash-to-bang", "flash to bang: 9 seconds",
                                      "about 9 s flash-to-bang", "9 s"])
    def test_flash_to_bang_distance(self, text):
        # 9 s at 20 degC: c = 343.2 m/s -> 3.09 km; Plan App. A: 2.91 s/km at 20 degC
        assert U.parse_distance_km(text, temp_c=20.0) == pytest.approx(9.0 / 2.91, rel=3e-3)

    def test_flash_to_bang_rejects_other_units(self):
        with pytest.raises(ValueError):
            U.parse_distance_km("flash-to-bang 9 km")

    @pytest.mark.parametrize("temp_c, s_per_km", [(-10.0, 3.08), (20.0, 2.91), (35.0, 2.84)])
    def test_flash_to_bang_follows_temperature(self, temp_c, s_per_km):
        assert U.flash_to_bang_s(1.0, temp_c) == pytest.approx(s_per_km, abs=0.01)   # Plan App. A thunder row
        assert U.seconds_to_km(U.flash_to_bang_s(7.3, temp_c), temp_c) == pytest.approx(7.3, rel=1e-12)

    def test_beaufort_fit_lands_in_wmo_bands(self):
        # WMO-No. 306 / Met Office band table (Plan App. A): the 0.836 B^1.5 fit at every integer
        # force falls inside that force's band, and the bands' edges map as published
        for b in range(13):
            assert U.beaufort_force(U.beaufort_to_mps(b)) == b
            assert U.mps_to_beaufort(U.beaufort_to_mps(b)) == pytest.approx(b, abs=1e-9)
        edges = [(0.29, 0), (0.3, 1), (1.6, 2), (3.4, 3), (5.5, 4), (8.0, 5), (10.8, 6), (13.9, 7), (17.2, 8),
                 (20.8, 9), (24.5, 10), (28.5, 11), (32.7, 12)]
        for v, force in edges:
            assert U.beaufort_force(v) == force
        assert U.beaufort_name(8) == "gale" and U.beaufort_name(6) == "strong breeze"

    @pytest.mark.parametrize("rate, wmo, ams", [
        (0.01, "none", "none"),           # below the 0.02 mm/h WMO detection threshold (Plan App. A)
        (1.0, "slight", "light"),
        (2.49, "slight", "light"),        # WMO slight < 2.5; AMS light <= 2.5 mm/h
        (2.6, "moderate", "moderate"),    # AMS moderate 2.6-7.6
        (7.6, "moderate", "moderate"),
        (7.7, "moderate", "heavy"),       # AMS heavy > 7.6
        (9.99, "moderate", "heavy"),      # WMO moderate 2.5-10
        (10.0, "heavy", "heavy"),         # WMO heavy 10-50
        (49.9, "heavy", "heavy"),
        (50.0, "intense", "heavy"),       # WMO intense >= 50
        (100.0, "cloudburst", "heavy"),   # cloudburst >= 100 (Plan 2.5)
    ])
    def test_rain_labels(self, rate, wmo, ams):
        assert U.rain_label(rate, "wmo") == wmo
        assert U.rain_label(rate, "ams") == ams

    @pytest.mark.parametrize("rate, wmo, ams", [
        (0.05, "slight drizzle", "light drizzle"),      # WMO drizzle < 0.1 / 0.1-0.5 / >= 0.5 (Plan App. A)
        (0.1, "moderate drizzle", "light drizzle"),     # AMS drizzle light < 0.3, moderate 0.3-0.5, heavy > 0.5
        (0.3, "moderate drizzle", "moderate drizzle"),
        (0.5, "heavy drizzle", "moderate drizzle"),
        (0.8, "heavy drizzle", "heavy drizzle"),
    ])
    def test_drizzle_labels(self, rate, wmo, ams):
        assert U.rain_label(rate, "wmo", "drizzle") == wmo
        assert U.rain_label(rate, "ams", "drizzle") == ams

    def test_dew_point_and_humidity_are_inverse_and_match_bolton(self):
        for t, rh in [(-5.0, 80.0), (10.0, 55.0), (22.0, 94.0), (30.0, 40.0)]:
            assert U.humidity_from_dew_point(t, U.dew_point_c(t, rh)) == pytest.approx(rh, abs=1e-6)
        # Alduchov & Eskridge vs Bolton (1980) Magnus coefficients agree to < 0.5 %RH at 12/10 degC
        assert U.humidity_from_dew_point(12.0, 10.0) == pytest.approx(100 * es_bolton(10) / es_bolton(12), abs=0.5)

    def test_compass_and_relative_direction(self):
        assert [U.compass_label(d) for d in (0, 11.2, 11.3, 247.5, 315, 359)] == ["N", "N", "NNE", "WSW", "NW", "N"]
        assert U.compass_label(100, points=8) == "E"
        assert U.relative_direction(270.0, 0.0) == "left"           # facing north, wind from the west
        assert U.relative_direction(90.0, 90.0) == "front"

    def test_direction_words(self):
        assert U.parse_direction_deg("from the north-west") == 315.0
        assert U.parse_direction_deg("south-westerly") == 225.0
        assert U.parse_direction_deg("west south west") == 247.5
        assert U.parse_direction_deg("370") == 10.0
        with pytest.raises(ValueError):
            U.parse_direction_deg("westward")                       # a 'to' direction is refused

    def test_decimal_comma_and_thousands(self):
        assert U.parse_rate("12,5 mm/h") == pytest.approx(12.5)
        assert U.parse_length_m("1,000 m") == pytest.approx(1000.0)
        assert U.parse_duration_s("01:00:00:12") == pytest.approx(3600.5)   # SMPTE at 24 fps

    def test_hail_sizes(self):
        assert U.parse_hail_size_mm("golf ball") == pytest.approx(44.5)    # NOAA: golf ball 1.75 in
        assert U.parse_hail_size_mm("1.75 in") == pytest.approx(44.45)
        assert U.parse_hail_size_mm("15 mm (marble)") == pytest.approx(15.0)  # an explicit size wins over a name
        assert U.hail_label(25.4) == "quarter"                              # NOAA: quarter = 1 in

    _ROUND_TRIP = (
        ("weather.precipitation.rate_mm_h", [0.3, 5.0, 23.0, 120.0]),
        ("weather.wind.speed_mps", [0.5, 4.4, 9.4, 18.9, 30.0]),
        ("weather.wind.from_deg", [0.0, 247.5, 315.0, 359.0]),
        ("weather.temperature_c", [-12.0, 0.5, 12.0, 31.0]),
        ("weather.thunder.distance_km", [0.8, 3.0, 8.0, 22.0]),
        ("weather.thunder.approach_kmh", [-30.0, 20.0]),
        ("weather.pressure_hpa", [980.0, 1013.0, 1040.0]),
        ("weather.hour", [0.0, 6.5, 22.25]),
        ("weather.visibility_km", [0.2, 5.0, 30.0]),
        ("listener.height_m", [0.4, 1.7, 12.0]),
        ("weather.precipitation.hail_size_mm", [6.4, 15.0, 44.5]),
        ("weather.humidity_pct", [45.0, 93.0]),
    )

    @pytest.mark.parametrize("system", ["metric", "uk", "us"])
    @pytest.mark.parametrize("path, values", _ROUND_TRIP)
    def test_format_parse_round_trip(self, path, values, system):
        # every readout parses back to the stored value within the precision it shows:
        # two significant figures (units._sig) or whole units (degrees, °C/°F, %, hPa, minutes)
        for v in values:
            text = U.format_value(path, v, system=system)
            back = U.parse_value(path, text)
            leaf = path.rsplit(".", 1)[-1]
            if leaf in ("from_deg",):
                assert abs((back - v + 180) % 360 - 180) <= 0.5, text
            elif leaf == "hour":
                assert back == pytest.approx(v, abs=1 / 60), text
            elif leaf in ("temperature_c",):
                assert back == pytest.approx(v, abs=0.56 if system == "us" else 0.5), text   # 1 degF = 0.56 K
            elif leaf in ("humidity_pct", "pressure_hpa"):
                assert back == pytest.approx(v, abs=0.5 if system != "us" else 0.34), text   # 0.01 inHg
            elif leaf == "approach_kmh":
                assert back == pytest.approx(v, abs=0.5 * (1.609344 if system != "metric" else 1.0)), text
            elif leaf == "speed_mps" and system != "metric":
                assert back == pytest.approx(v, abs=0.5 * 0.44704), text                    # whole mph shown
            elif leaf == "hail_size_mm" and system != "us":
                assert back == pytest.approx(v, abs=0.5), text                              # whole mm shown
            else:
                assert back == pytest.approx(v, rel=0.05), text

    def test_readout_reads_its_most_precise_part(self):
        assert U.parse_value("weather.wind.speed_mps", "force 5, fresh breeze, 9.4 m/s (34 km/h)") == 9.4
        assert U.parse_value("weather.wind.speed_mps", "force 5, fresh breeze") == pytest.approx(0.836 * 5 ** 1.5)
        assert U.parse_value("weather.precipitation.rate_mm_h", "light, steady rain") == 5.0   # words still parse
        assert U.format_value("weather.wind.speed_mps", 9.4) == "force 5, fresh breeze, 9.4 m/s (34 km/h)"

    def test_format_labels(self):
        assert U.format_value("weather.precipitation.rate_mm_h", 5.0) == "5.0 mm/h, moderate"
        assert U.format_value("weather.precipitation.rate_mm_h", 5.0, "us").endswith("moderate")   # AMS words
        assert "23 s flash-to-bang" in U.format_value("weather.thunder.distance_km", 8.0)   # 8 km x 2.91 s/km
        assert U.format_value("weather.wind.gust_factor", None) == "auto"


# =====================================================================================
# Scene schema (rainscene format)
# =====================================================================================
def _rich_scene() -> Scene:
    s = Scene(name="round trip", seed=2 ** 63 + 5)
    s.place.objects.append(SceneObject(id="oak", type="tree", material="oak", position=(3.0, -2.5, 0.0),
                                       size=(4.0, 4.0, 9.0), rotation_deg=30.0, params={"species": "oak"}))
    s.timeline = [Keyframe(at=0.0, path="weather.precipitation.rate_mm_h", value=2.0),
                  Keyframe(at="00:30", path="weather.precipitation.rate_mm_h", value="20 mm/h", curve="log"),
                  Keyframe(at=12.5, path="listener.position", value=[1.0, 2.0, 0.0], curve="smooth"),
                  Keyframe(at=40.0, path="weather.thunder.enabled", value=True, curve="hold")]
    s.overrides = {"weather.wind.speed_mps": {"state": "pinned", "value": 7.5}}
    s.vibe = {"intensity": 55.0}
    s.weather.wind.gust_factor = 1.6
    return s


class TestSchema:
    def test_json_round_trip_is_equal(self):
        s = _rich_scene()
        back = Scene.from_json(s.to_json())
        assert back == s
        assert isinstance(back.place.objects[0], SceneObject) and back.place.objects[0].position == (3.0, -2.5, 0.0)
        assert isinstance(back.place.objects[0].size, tuple)
        assert all(isinstance(k, Keyframe) for k in back.timeline)
        assert isinstance(back.listener.position, tuple)

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_file_round_trip_is_equal(self, tmp_out, ext):
        s = _rich_scene()
        path = str(tmp_out / f"scene{ext}")
        s.save(path)
        assert Scene.load(path) == s

    def test_rainscene_version_key(self):
        d = Scene().to_dict()
        assert d["rainscene"] == SCHEMA_VERSION and "version" not in d    # the file key is 'rainscene' (Plan 3.2)
        assert Scene.from_dict({"rainscene": 1, "name": "x"}).version == 1
        assert Scene.from_dict({"version": 1, "name": "x"}).name == "x"

    def test_unknown_keys_are_ignored_at_every_level(self):
        d = _rich_scene().to_dict()
        d["future_feature"] = {"a": 1}
        d["weather"]["aurora"] = True
        d["weather"]["precipitation"]["colour"] = "blue"
        d["place"]["objects"][0]["glow"] = 3
        d["timeline"][0]["easing_v2"] = "spring"
        d["render"]["codec"] = "opus"
        back = Scene.from_dict(d)
        assert back == _rich_scene()

    def test_partial_dict_takes_defaults(self):
        s = Scene.from_dict({"weather": {"precipitation": {"rate_mm_h": 12.0}}})
        assert s.weather.precipitation.rate_mm_h == 12.0
        assert s.weather.precipitation.type == Scene().weather.precipitation.type
        assert s.weather.wind == Scene().weather.wind
        import yaml
        empty = Scene.from_dict(yaml.safe_load("rainscene: 1\nweather:\nplace:\n  objects:\ntimeline:\n"))
        assert empty == Scene()                                            # empty YAML sections keep defaults

    def test_from_dict_does_not_alias_its_input(self):
        d = Scene().to_dict()
        s = Scene.from_dict(d)
        s.place.surfaces["gravel"] = 99.0
        s.place.emitters.append({"type": "chime"})
        assert "gravel" not in d["place"]["surfaces"] and len(d["place"]["emitters"]) == 2

    def test_get_set_paths(self):
        s = Scene()
        s.set("weather.wind.speed_mps", 3.0)
        s.set("place.surfaces.gravel", 5.0)
        assert s.get("weather.wind.speed_mps") == 3.0 and s.get("place.surfaces.gravel") == 5.0
        assert s.copy() == s and s.copy() is not s

    def test_every_preset_round_trips(self):
        for name in PRESETS:
            sc = preset(name)
            assert Scene.from_json(sc.to_json()) == sc, name


# =====================================================================================
# Presets
# =====================================================================================
def _check_in_manifest(scene: Scene, where: str) -> list[str]:
    bad = []
    for spec in mf.PARAMS:
        try:
            v = scene.get(spec.id)
        except (KeyError, AttributeError):
            continue
        if v is None:
            if not spec.nullable:
                bad.append(f"{where}: {spec.id} is None")
            continue
        if spec.type == "vec3":
            if not (isinstance(v, tuple) and len(v) == 3 and all(math.isfinite(float(x)) for x in v)):
                bad.append(f"{where}: {spec.id} = {v!r}")
        elif spec.choices:
            if v not in spec.choices:
                bad.append(f"{where}: {spec.id} = {v!r} not in {spec.choices}")
        elif spec.is_numeric and spec.min is not None and spec.max is not None:
            ok = spec.min <= float(v) <= spec.max or (spec.zero_ok and float(v) == 0.0)
            if not ok or not math.isfinite(float(v)):
                bad.append(f"{where}: {spec.id} = {v!r} outside {spec.min}..{spec.max}")
    for mat, share in scene.place.surfaces.items():
        if not 0.0 <= float(share) <= 100.0:
            bad.append(f"{where}: surface {mat} = {share}")
    return bad


class TestPresets:
    def test_there_are_28_presets_and_all_load(self):
        assert len(PRESETS) == 28
        cards = list_presets()
        assert len(cards) == 28 and len({c["title"] for c in cards}) == 28        # one name everywhere (Plan 2.4)
        for c in cards:
            assert isinstance(preset(c["name"]), Scene)

    def test_every_value_is_inside_the_manifest_ranges(self):
        bad = []
        for name in PRESETS:
            bad += _check_in_manifest(preset(name), name)
        assert not bad, "\n".join(bad)

    def test_rain_rate_zero_only_without_precipitation(self):
        for name in PRESETS:
            p = preset(name).weather.precipitation
            assert (p.rate_mm_h == 0.0) == (p.type == "none"), name

    def test_surface_mix_is_percent_coverage(self):
        for name in PRESETS:
            assert sum(preset(name).place.surfaces.values()) == pytest.approx(100.0, abs=0.1), name

    def test_mode_presets_honour_their_guard_rails(self):
        sleep = preset("window-rain-bedroom")
        assert sleep.mode == "sleep" and sleep.vibe["timer_min"] == 45.0 and sleep.vibe["startle_guard"]
        focus = preset("focus-steady-roof")
        assert not focus.weather.thunder.enabled and focus.place.emitters == []
        assert focus.weather.wind.gust_factor <= 1.25                      # Plan 8.3: gust swells <= +-2 dB

    def test_presets_are_deterministic_and_independent(self):
        a = preset("tin-roof-cabin-storm")
        a.weather.precipitation.rate_mm_h = 1.0
        assert preset("tin-roof-cabin-storm").weather.precipitation.rate_mm_h == 22.0

    def test_tags_and_aliases(self):
        assert {c["name"] for c in list_presets("sleep")} >= {"window-rain-bedroom", "steady-rain-leaves-night"}
        assert preset("Steady Rain Leaves Night") == preset("steady-rain-leaves-night")
        with pytest.raises(KeyError):
            preset("no-such-scene")


# =====================================================================================
# Manifest
# =====================================================================================
_JSON_TYPES = {
    "object": lambda x: isinstance(x, dict),
    "array": lambda x: isinstance(x, (list, tuple)),
    "string": lambda x: isinstance(x, str),
    "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "integer": lambda x: (isinstance(x, int) and not isinstance(x, bool)) or (isinstance(x, float) and x.is_integer()),
    "boolean": lambda x: isinstance(x, bool),
    "null": lambda x: x is None,
}
_KEYWORDS = {"$schema", "$id", "title", "description", "type", "properties", "additionalProperties", "items",
             "minItems", "maxItems", "required", "enum", "const", "minimum", "maximum", "default"}


def _schema_errors(inst, sch, path="$") -> list[str]:
    """A small JSON Schema 2020-12 validator for the keywords the manifest emits."""
    errs = []
    if "const" in sch and inst != sch["const"]:
        errs.append(f"{path}: {inst!r} != const {sch['const']!r}")
    if "enum" in sch and not any(inst == e and type(inst) is type(e) or
                                 (_JSON_TYPES["number"](inst) and _JSON_TYPES["number"](e) and inst == e)
                                 for e in sch["enum"]):
        errs.append(f"{path}: {inst!r} not in enum")
    if "type" in sch:
        types = sch["type"] if isinstance(sch["type"], list) else [sch["type"]]
        if not any(_JSON_TYPES[t](inst) for t in types):
            return errs + [f"{path}: {inst!r} is not {types}"]
    if _JSON_TYPES["number"](inst):
        if "minimum" in sch and inst < sch["minimum"]:
            errs.append(f"{path}: {inst} < {sch['minimum']}")
        if "maximum" in sch and inst > sch["maximum"]:
            errs.append(f"{path}: {inst} > {sch['maximum']}")
    if isinstance(inst, dict):
        props = sch.get("properties", {})
        errs += [f"{path}: missing {k}" for k in sch.get("required", []) if k not in inst]
        for k, v in inst.items():
            if k in props:
                errs += _schema_errors(v, props[k], f"{path}.{k}")
            elif isinstance(sch.get("additionalProperties"), dict):
                errs += _schema_errors(v, sch["additionalProperties"], f"{path}.{k}")
    if isinstance(inst, (list, tuple)):
        if len(inst) < sch.get("minItems", 0) or len(inst) > sch.get("maxItems", 10 ** 9):
            errs.append(f"{path}: {len(inst)} items")
        if isinstance(sch.get("items"), dict):
            for i, v in enumerate(inst):
                errs += _schema_errors(v, sch["items"], f"{path}[{i}]")
    return errs


def _walk_keywords(sch, path="$") -> list[str]:
    bad = [f"{path}.{k}" for k in sch if k not in _KEYWORDS and not k.startswith("x-")]
    for k, v in sch.get("properties", {}).items():
        bad += _walk_keywords(v, f"{path}.{k}")
    for k in ("items", "additionalProperties"):
        if isinstance(sch.get(k), dict):
            bad += _walk_keywords(sch[k], f"{path}.{k}")
    return bad


class TestManifest:
    def test_every_param_path_resolves_on_a_default_scene(self):
        s = Scene()
        for spec in mf.PARAMS:
            parent, _, leaf = spec.id.rpartition(".")
            container = s.get(parent) if parent else s
            if isinstance(container, dict):
                # open-ended mappings (vibe macros, surface shares): a missing key means the spec default
                assert leaf in container or spec.default is not None, spec.id
            else:
                assert hasattr(container, leaf), spec.id
                v = s.get(spec.id)
                d = spec.default
                assert (list(v) if isinstance(v, tuple) else v) == (list(d) if isinstance(d, tuple) else d), spec.id

    def test_ids_unique_and_specs_consistent(self):
        ids = [p.id for p in mf.PARAMS]
        assert len(ids) == len(set(ids))
        for p in mf.PARAMS:
            assert p.tier in mf.TIER_NAMES and p.audibility in mf.AUDIBILITY
            assert p.scale in ("lin", "log", "beaufort", "choice")
            if p.scale == "log":
                assert p.min > 0                          # a log axis needs a positive floor
            if p.min is not None and p.max is not None:
                assert p.min < p.max, p.id
                if p.default is not None and p.is_numeric:
                    assert p.min <= p.default <= p.max or (p.zero_ok and p.default == 0), p.id
            if p.choices and p.default is not None:
                assert p.default in p.choices, p.id

    def test_smoothing_times_follow_the_plan(self):
        # Plan 2.6 Smoothing: time to 99.5 %: rain 3 s, wind 1 s, distance 2 s, macros 0.25 s
        assert mf.get_param("weather.precipitation.rate_mm_h").smoothing_s == 3.0
        assert mf.get_param("weather.wind.speed_mps").smoothing_s == 1.0
        assert mf.get_param("weather.thunder.distance_km").smoothing_s == 2.0
        assert mf.get_param("vibe.intensity").smoothing_s == 0.25
        assert mf.get_param("place.shelter").smoothing_s == 0.0         # discrete values switch, never glide

    def test_render_layouts_cover_the_decoders(self):
        from raingen.spatial.decoders import SPEAKER_LAYOUTS
        layouts = mf.get_param("listener.render").choices
        assert set(SPEAKER_LAYOUTS) <= set(layouts)
        assert {"stereo", "binaural", "mono", "foa", "hoa3"} <= set(layouts)

    def test_destinations_match_the_master_profiles(self):
        from raingen.master.destinations import DESTINATIONS
        assert set(mf.get_param("render.destination").choices) == set(DESTINATIONS)

    def test_json_schema_structure(self):
        sch = mf.to_json_schema()
        json.dumps(sch)                                                   # serialisable
        assert sch["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert sch["type"] == "object" and sch["properties"]["rainscene"]["const"] == SCHEMA_VERSION
        assert not _walk_keywords(sch), _walk_keywords(sch)
        for spec in mf.PARAMS:                                            # every parameter at its nested location
            node = sch
            for part in spec.id.split("."):
                node = node["properties"][part]
            assert node["x-osc"] == spec.osc and node["title"] == spec.label
            if spec.is_numeric and spec.min is not None:
                assert node["maximum"] == spec.max

    def test_every_preset_validates_against_the_json_schema(self):
        sch = mf.to_json_schema()
        for name in PRESETS:
            errs = _schema_errors(json.loads(preset(name).to_json()), sch)
            assert not errs, (name, errs)
        rich = _rich_scene()
        ParameterGraph(rich).set_offset(RATE, 1.0)                         # an Adjusted entry with its ratio
        assert not _schema_errors(json.loads(rich.to_json()), sch)

    def test_json_schema_rejects_bad_scenes(self):
        sch = mf.to_json_schema()
        d = json.loads(Scene().to_json())
        d["weather"]["precipitation"]["rate_mm_h"] = 500.0
        d["place"]["shelter"] = "spaceship"
        d["listener"]["position"] = [1.0, 2.0]
        errs = _schema_errors(d, sch)
        assert len(errs) == 3, errs

    def test_osc_namespace_unique_and_well_formed(self):
        ns = mf.osc_namespace()
        addr_re = re.compile(r"^/rg(/[a-z0-9_]+)+$")
        tags = set("ifsbhtdScrmTFNI")                                      # OSC 1.0 / 1.1 type tags
        for addr, entry in ns.items():
            assert addr_re.match(addr), addr
            assert entry["type"] and set(entry["type"]) <= tags, (addr, entry["type"])
        oscs = [p.osc for p in mf.PARAMS]
        assert len(oscs) == len(set(oscs)) and set(oscs) <= set(ns)
        for p in mf.PARAMS:
            assert mf.param_by_osc(p.osc) is p
            assert p.osc == "/rg/" + p.id.replace(".", "/")               # one canonical id everywhere (Plan 2.12)

    def test_wwise_export_well_formed(self):
        w = mf.wwise_manifest()
        names = [r["name"] for r in w["RTPCs"]] + [g["group"] for g in w["Switches"] + w["States"]]
        assert len(names) == len(set(names))
        assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names)
        for r in w["RTPCs"]:
            assert r["min"] < r["max"] and r["min"] <= r["default"] <= r["max"], r
            assert r["slew_s"] >= 0.0
        for g in w["Switches"]:
            assert g["switches"] and g["default"] in g["switches"], g
        for g in w["States"]:
            assert g["states"] and g["default"] in g["states"], g
        preset_group = next(g for g in w["States"] if g["group"] == "RG_Preset")
        assert len(preset_group["states"]) == 28
        json.dumps(w)

    def test_fmod_export_well_formed(self):
        f = mf.fmod_manifest()
        names = [p["name"] for p in f["parameters"]]
        assert len(names) == len(set(names))
        for p in f["parameters"]:
            assert p["type"] in ("continuous", "discrete", "labeled")
            if p["type"] == "labeled":
                assert p["labels"] and p["max"] == len(p["labels"]) - 1 and 0 <= p["default"] <= p["max"]
            else:
                assert p["min"] < p["max"] and p["min"] <= p["default"] <= p["max"], p
                spec = mf.get_param(p["path"])
                if spec.smoothing_s > 0:
                    assert p["seek_speed"] == pytest.approx((spec.max - spec.min) / spec.smoothing_s)
        json.dumps(f)

    def test_markdown_reference_renders(self):
        md = mf.markdown_reference()
        rows = [ln for ln in md.splitlines() if ln.startswith("| `")]
        assert len(rows) == len(mf.PARAMS)
        assert all(ln.count("|") == 10 for ln in rows)                   # 9 columns, no stray pipes
        for tier, name in mf.TIER_NAMES.items():
            assert f"## Tier {tier}: {name}" in md
        assert len(json.loads(mf.manifest_json())["params"]) == len(mf.PARAMS)

    def test_normalize_is_perceptual_and_invertible(self):
        rate = mf.get_param("weather.precipitation.rate_mm_h")
        wind = mf.get_param("weather.wind.speed_mps")
        assert mf.denormalize(rate, 0.5) == pytest.approx(math.sqrt(0.02 * 150.0), rel=1e-9)   # geometric midpoint
        b_mid = 0.5 * U.mps_to_beaufort(40.0)
        assert mf.denormalize(wind, 0.5) == pytest.approx(U.beaufort_to_mps(b_mid), rel=1e-9)   # Beaufort midpoint
        for spec in mf.PARAMS:
            if not spec.is_numeric or spec.min is None or spec.max is None:
                continue
            xs = np.linspace(0, 0.99 if spec.wrap else 1.0, 11)               # 360 deg wraps to 0 deg
            vals = [mf.denormalize(spec, x) for x in xs]
            assert all(b >= a for a, b in itertools.pairwise(vals)), spec.id
            for x in (0.1, 0.37, 0.9):
                v = mf.denormalize(spec, x)
                tol = 0.5 / (spec.max - spec.min) + 1e-9 if spec.type == "int" else 1e-9   # half a step
                assert mf.normalize(spec, v) == pytest.approx(x, abs=tol), spec.id

    def test_14_bit_midi_resolves_the_log_rain_slider(self):
        # Plan 2.6: at 7 bits each step of a log rain slider is a ~7 % jump; 14-bit pairs fix it
        rate = mf.get_param("weather.precipitation.rate_mm_h")
        assert rate.midi == "cc14"
        step7 = mf.midi_to_value(rate, 64) / mf.midi_to_value(rate, 63) - 1.0
        step14 = mf.midi_to_value(rate, 64, 1) / mf.midi_to_value(rate, 64, 0) - 1.0
        assert step7 == pytest.approx((150 / 0.02) ** (1 / 127) - 1, rel=1e-6) and 0.06 < step7 < 0.08
        assert step14 < 1e-3

    def test_clamp_and_coerce(self):
        rate = mf.get_param("weather.precipitation.rate_mm_h")
        assert mf.clamp(rate, 0.0) == 0.0                                   # no rain stays no rain
        assert mf.clamp(rate, 500.0) == 150.0 and mf.clamp(rate, 0.001) == 0.02
        assert mf.clamp(mf.get_param("weather.wind.from_deg"), 370.0) == pytest.approx(10.0)
        assert mf.clamp(mf.get_param("place.vegetation.conifer"), 7.6) == 8
        assert mf.coerce(mf.get_param("weather.wind.speed_mps"), "30 km/h") == pytest.approx(30 / 3.6)
        assert mf.coerce(mf.get_param("weather.wind.from_deg"), "NW") == 315.0
        assert mf.coerce(mf.get_param("place.shelter"), "tin roof") == "attic"
        assert mf.coerce(mf.get_param("weather.thunder.enabled"), "on") is True
        assert mf.coerce(mf.get_param("listener.position"), "1, 2, 3.5") == (1.0, 2.0, 3.5)
        assert mf.coerce(mf.get_param("weather.wind.gust_factor"), "auto") is None
        assert mf.coerce(mf.get_param("mode"), "Relax") == "calm"
        with pytest.raises(ValueError):
            mf.coerce(mf.get_param("place.shelter"), "spaceship")


# =====================================================================================
# Parameter graph (Following / Adjusted / Locked)
# =====================================================================================
RATE = "weather.precipitation.rate_mm_h"
WIND = "weather.wind.speed_mps"


class TestGraph:
    def test_log_offset_is_a_ratio_bounded_to_quarter_and_four(self, scene):
        g = ParameterGraph(scene)
        g.set_offset(RATE, 1.0)                                            # one doubling
        assert g.overrides[RATE]["ratio"] == pytest.approx(2.0) and g.ratio(RATE) == pytest.approx(2.0)
        assert g.adjustment_text(RATE) == "×2"
        assert g.set_offset(RATE, 5.0) == pytest.approx(2.0) and g.ratio(RATE) == pytest.approx(MAX_RATIO)
        assert g.set_offset(RATE, -9.0) == pytest.approx(-2.0) and g.ratio(RATE) == pytest.approx(MIN_RATIO)
        assert (MIN_RATIO, MAX_RATIO) == (0.25, 4.0)
        for _ in range(10):                                                # nudging cannot escape the bound
            g.nudge(RATE, 1.0)
        assert g.ratio(RATE) == pytest.approx(4.0)

    def test_offset_follows_the_driver_multiplicatively(self, scene):
        g = ParameterGraph(scene)
        g.set_offset(RATE, 3.0, perceptual=False, driver=4.0)              # "+3 mm/h" on 4 mm/h -> x1.75
        assert g.ratio(RATE) == pytest.approx(7.0 / 4.0)
        for driver in (0.5, 4.0, 20.0):
            assert g.resolve(RATE, driver) == pytest.approx(driver * 1.75)
        assert g.resolve(RATE, 0.0) == 0.0                                 # x1.75 of no rain is no rain

    def test_ratio_in_the_scene_file_wins_and_round_trips(self, scene):
        g = ParameterGraph(scene)
        g.set_offset(RATE, math.log2(1.5))
        back = Scene.from_json(scene.to_json())
        g2 = ParameterGraph(back)
        assert g2.resolve(RATE, 4.0) == pytest.approx(6.0)
        back.overrides[RATE] = {"state": "offset", "ratio": 3.0}            # hand-edited file: ratio only
        assert ParameterGraph(back).resolve(RATE, 4.0) == pytest.approx(12.0)
        back.overrides[RATE] = {"state": "offset", "ratio": 100.0}          # out of bound -> x4
        assert ParameterGraph(back).resolve(RATE, 4.0) == pytest.approx(16.0)

    def test_beaufort_and_linear_offsets_add(self, scene):
        g = ParameterGraph(scene)
        g.set_offset(WIND, 1.0)                                            # +1 Beaufort (Plan 2.7)
        assert U.mps_to_beaufort(g.resolve(WIND, U.beaufort_to_mps(3))) == pytest.approx(4.0)
        assert g.adjustment_text(WIND) == "+1 Beaufort" and g.ratio(WIND) is None
        g.set_offset("weather.temperature_c", -2.0)
        assert g.resolve("weather.temperature_c", 12.0) == pytest.approx(10.0)

    def test_locked_holds_when_tier0_macros_change(self, scene):
        g = ParameterGraph(scene)
        locked = g.set_manual(WIND, "force 2")
        assert locked == pytest.approx(U.beaufort_to_mps(2))
        rates = []
        for i in (10, 40, 70, 100):
            out = g.apply(apply_vibe(scene, intensity=i), source="Macro")
            assert out.weather.wind.speed_mps == pytest.approx(locked)
            rates.append(out.weather.precipitation.rate_mm_h)
        assert rates == sorted(rates) and rates[-1] > rates[0]               # the unlocked rate still follows
        assert g.label(WIND) == "Locked" and g.provenance(WIND) == "Manual"

    def test_reset_returns_to_following(self, scene):
        g = ParameterGraph(scene)
        g.set_offset(RATE, 1.0)
        g.set_manual(WIND, 12.0)
        driven = apply_vibe(scene, intensity=60)
        assert g.label(RATE) == "Adjusted" and g.state(RATE) == "offset"
        g.relink(RATE)
        assert g.state(RATE) == "auto" and g.label(RATE) == "Following scene"
        assert g.apply(driven).weather.precipitation.rate_mm_h == driven.weather.precipitation.rate_mm_h
        g.relink_all()
        assert g.overrides == {} and g.apply(driven).weather.wind.speed_mps == driven.weather.wind.speed_mps
        g.apply(driven, source="Live")
        assert g.label(WIND) == "Following live weather"

    def test_labels_never_show_internal_state_words(self, scene):
        g = ParameterGraph(scene)
        g.set_offset(RATE, 2.0)
        g.apply(apply_vibe(scene, intensity=100))                          # 100 mm/h x 4 is past 150: clamped
        assert g.at_limit(RATE) and g.label(RATE) == "Adjusted (at limit)"
        for path in (RATE, WIND, "place.shelter"):
            assert not re.search(r"\b(auto|offset|pinned)\b", g.label(path).lower())

    def test_discrete_values_only_follow_or_lock(self, scene):
        g = ParameterGraph(scene)
        with pytest.raises(ValueError):
            g.set_offset("place.shelter", 1.0)
        g.set_manual("weather.thunder.enabled", "off")
        out = g.apply(apply_vibe(scene, intensity=100))                    # the storm path would turn thunder on
        assert out.weather.thunder.enabled is False

    def test_automation_beats_a_lock(self, scene):
        scene.timeline = [Keyframe(at=0.0, path=WIND, value=15.0)]
        g = ParameterGraph(scene)
        g.set_manual(WIND, 2.0)
        assert g.apply(tl.evaluate(scene, 1.0)).weather.wind.speed_mps == 15.0   # Plan 2.7 priority
        assert g.provenance(WIND) == "Automation"
        assert g.label(WIND) == "Automated: your lock is ignored while this lane is active"   # Plan 2.7 wording
        assert g.label(RATE) == "Following scene"

    def test_one_gesture_is_one_undo_step(self, scene):
        g = ParameterGraph(scene)
        with g.gesture():
            for _ in range(5):
                g.nudge(RATE, 0.1)
        g.set_manual(WIND, 3.0)
        assert g.undo() and WIND not in g.overrides and g.state(RATE) == "offset"
        assert g.undo() and g.overrides == {}
        assert not g.undo()
        assert g.redo() and g.state(RATE) == "offset" and g.ratio(RATE) == pytest.approx(2 ** 0.5)


# =====================================================================================
# Vibe macros
# =====================================================================================
_SHELTER_RANK = {"open": 0, "tree": 1, "umbrella": 1, "porch": 2, "tent": 2, "car": 2, "attic": 2, "cave": 2,
                 "underpass": 2, "room_open": 3, "room_closed": 4}


class TestVibe:
    @pytest.mark.parametrize("i, rate, label", [
        (20, 0.3, "moderate drizzle"),      # Plan 2.3 "How much rain" detents
        (40, 5.0, "moderate"),
        (60, 20.0, "heavy"),
        (80, 50.0, "intense"),
        (100, 100.0, "cloudburst"),
    ])
    def test_intensity_anchors(self, i, rate, label):
        assert intensity_rate(i) == pytest.approx(rate, rel=1e-9)
        kind = "drizzle" if rate < 0.5 else "rain"
        assert U.rain_label(rate, "wmo", kind) == label

    def test_intensity_is_monotonic(self, scene):
        rows = []
        for i in range(0, 101, 5):
            w = apply_vibe(scene, intensity=i).weather
            rows.append((w.precipitation.rate_mm_h, w.wind.speed_mps, w.thunder.enabled, w.thunder.distance_km,
                         w.thunder.flashes_per_min))
        rates, winds = [r[0] for r in rows], [r[1] for r in rows]
        assert rates == sorted(rates) and rates[0] == 0.0 and rates[-1] == pytest.approx(100.0)
        assert winds == sorted(winds)
        thunder = [r for r in rows if r[2]]
        assert [r[2] for r in rows] == sorted(r[2] for r in rows)          # once on, stays on
        dists, flashes = [r[3] for r in thunder], [r[4] for r in thunder]
        assert dists == sorted(dists, reverse=True) and flashes == sorted(flashes)
        assert dists[-1] == pytest.approx(1.5) and dists[0] >= 5.0          # Plan 2.3: ~5 km at 80, 1.5 km at 100

    def test_closeness_is_monotonic(self, scene):
        ranks, windows = [], []
        for c in range(0, 101, 5):
            p = apply_vibe(scene, closeness=c).place
            ranks.append(_SHELTER_RANK[p.shelter])
            if p.shelter.startswith("room"):
                windows.append(p.window_open)
        assert ranks == sorted(ranks) and ranks[0] == 0 and ranks[-1] == 4
        assert windows == sorted(windows, reverse=True) and windows[-1] == 0.0

    def test_wildness_is_monotonic(self, scene):
        base = apply_vibe(scene, intensity=85)
        rows = [apply_vibe(base, wildness=w).weather for w in range(0, 101, 10)]
        g = [w.wind.gust_factor for w in rows]
        assert g == sorted(g) and g[0] == pytest.approx(1.15) and g[-1] == pytest.approx(1.9)
        for w in rows:
            ti = w.wind.turbulence_intensity                                # G = 1 + 3 I (Plan App. A)
            assert ti == pytest.approx((w.wind.gust_factor - 1) / 3, abs=1e-9) and 0.05 <= ti <= 0.5
        rates = [w.wind.gust_rate_per_min for w in rows]
        flashes = [w.thunder.flashes_per_min for w in rows]
        assert rates == sorted(rates) and flashes == sorted(flashes) and flashes[-1] > flashes[0]

    def test_macros_are_idempotent_and_pure(self, scene):
        before = scene.copy()
        a = apply_vibe(scene, intensity=70, closeness=60, wildness=40)
        assert scene == before
        b = apply_vibe(apply_vibe(a, intensity=20), intensity=70)          # the base snapshot is reused
        assert b.weather == a.weather and b.place == a.place

    def test_intensity_from_scene_inverts_the_rate_anchors(self, scene):
        for i in (10, 25, 40, 55, 75, 95):
            assert intensity_from_scene(apply_vibe(scene, intensity=i)) == pytest.approx(i, abs=1e-6)


# =====================================================================================
# Text parsing
# =====================================================================================
class TestTextParse:
    def test_heavy_rain_on_a_tin_roof_with_distant_thunder(self):
        s = tp.scene_from_text("heavy rain on a tin roof with distant thunder")
        assert s.weather.precipitation.rate_mm_h >= 8.0                    # AMS heavy > 7.6 mm/h
        assert U.rain_label(s.weather.precipitation.rate_mm_h) == "heavy"
        assert s.place.shelter == "attic"                                  # the attic shelter is the tin roof
        assert s.weather.thunder.enabled and s.weather.thunder.distance_km > 5.0

    def test_light_drizzle_in_a_pine_forest_at_dawn(self):
        s = tp.scene_from_text("light drizzle in a pine forest at dawn")
        p = s.weather.precipitation
        assert p.type == "drizzle" and p.drops == "fine"
        assert U.rain_label(p.rate_mm_h, "wmo", "drizzle") == "slight drizzle"   # WMO slight drizzle < 0.1 mm/h
        assert s.place.preset == "conifer_forest" and s.place.vegetation["conifer"] > s.place.vegetation["deciduous"]
        assert 5.0 <= s.weather.hour <= 7.0

    def test_gale_force_winds_from_the_northwest(self):
        s = tp.scene_from_text("gale force winds from the northwest")
        assert U.beaufort_force(s.weather.wind.speed_mps) == 8              # gale = force 8 (WMO-No. 306)
        assert s.weather.wind.from_deg == 315.0

    def test_strict_mode_raises_on_gibberish(self):
        with pytest.raises(tp.TextParseError):
            tp.parse_text("qwzx blorft snargle", strict=True)
        with pytest.raises(ValueError):
            tp.scene_from_text("zzkrb vlimp", strict=True)
        assert tp.parse_text("qwzx blorft snargle").assignments == {}      # non-strict: nothing, no error
        assert tp.parse_text("rain please", strict=True).assignments["weather.precipitation.type"] == "rain"

    def test_api_strict_resolution(self):
        from raingen.api import SceneParseError, _resolve
        with pytest.raises(SceneParseError):
            _resolve("qwzx blorft snargle", strict=True)

    @pytest.mark.parametrize("text, path, expected, tol", [
        ("wind 20 kt from WSW", "weather.wind.speed_mps", 20 * 1852 / 3600, 1e-9),
        ("wind 20 kt from WSW", "weather.wind.from_deg", 247.5, 0.0),
        ("force 6 westerly", "weather.wind.speed_mps", 0.836 * 6 ** 1.5, 1e-9),
        ("force 6 westerly", "weather.wind.from_deg", 270.0, 0.0),
        ("36 km/h wind", "weather.wind.speed_mps", 10.0, 1e-9),
        ("thunder 9 seconds after the flash", "weather.thunder.distance_km", 9 * c_air(12.0) / 1000, 1e-9),
        ("rain at 8 mm/h", "weather.precipitation.rate_mm_h", 8.0, 0.0),
        ("0.2 in/h of rain", "weather.precipitation.rate_mm_h", 5.08, 1e-9),
        ("drizzle, 54 F", "weather.temperature_c", (54 - 32) * 5 / 9, 1e-9),
        ("thunder 3 miles away", "weather.thunder.distance_km", 3 * 1.609344, 1e-9),
    ])
    def test_vocabulary(self, text, path, expected, tol):
        s = tp.scene_from_text(text)
        assert s.get(path) == pytest.approx(expected, abs=tol)

    def test_gusts_and_relative_mean(self):
        r = tp.parse_text("30 mph gusts")
        g = r.assignments["weather.wind.gust_factor"]
        mean = r.assignments["weather.wind.speed_mps"]
        assert mean * g == pytest.approx(30 * 0.44704, rel=1e-6)

    def test_rain_below_freezing_becomes_snow(self):
        s = tp.scene_from_text("rain at -6 °C")
        assert s.weather.precipitation.type == "snow" and s.weather.temperature_c == -6.0

    def test_modes_and_layouts(self):
        s = tp.scene_from_text("gentle rain for sleep")
        assert s.mode == "sleep" and s.vibe["timer_min"] == 45.0
        assert tp.scene_from_text("asmr rain on an umbrella").listener.render == "binaural"

    def test_pasted_metar_routes_to_the_metar_parser(self):
        s = tp.scene_from_text("KJFK 121755Z 21016G24KT 1SM +TSRA BR BKN015CB 18/17 A2990")
        assert s.weather.thunder.enabled and s.weather.wind.from_deg == 210.0
        assert s.name.startswith("METAR KJFK")

    def test_parse_does_not_touch_the_base(self, scene):
        before = scene.copy()
        tp.scene_from_text("downpour on a tent, force 7", base=scene)
        assert scene == before


# =====================================================================================
# METAR
# =====================================================================================
def _metar(text):
    parsed = M.parse_metar(text)
    return parsed, M.metar_assignments(parsed), M.scene_from_metar(text)


class TestMetar:
    def test_egll_light_rain(self):
        p, _, s = _metar("METAR EGLL 161250Z 24015G25KT 9999 -RA BKN012 OVC020 12/10 Q1004")
        assert p["station"] == "EGLL" and p["time"] == {"day": 16, "hour": 12, "minute": 50}
        assert s.weather.precipitation.type == "rain"
        assert U.rain_label(s.weather.precipitation.rate_mm_h) == "slight"          # '-' = light (WMO slight)
        assert s.weather.wind.speed_mps == pytest.approx(15 * 1852 / 3600, rel=1e-12)
        assert s.weather.wind.from_deg == 240.0
        assert s.weather.wind.gust_factor == pytest.approx(25 / 15)
        assert s.weather.temperature_c == 12.0
        assert s.weather.humidity_pct == pytest.approx(100 * es_bolton(10) / es_bolton(12), abs=0.5)   # Magnus
        assert s.weather.pressure_hpa == 1004.0                                       # Q group: hPa
        assert s.weather.cloud_okta == 8.0                                            # OVC = 8 oktas
        assert s.weather.visibility_km == pytest.approx(9.999)
        assert not s.weather.thunder.enabled

    def test_kjfk_heavy_thunderstorm_rain(self):
        p, _, s = _metar("METAR KJFK 161251Z 18022G34KT 2SM +TSRA BR BKN008CB 22/21 A2981")
        w = s.weather
        assert w.precipitation.type in ("rain", "showers") and w.precipitation.character == "showery"
        assert U.rain_label(w.precipitation.rate_mm_h) == "heavy"                    # '+' = heavy (WMO 10-50)
        assert w.precipitation.drops == "fat"
        assert w.thunder.enabled and w.thunder.distance_km <= 5 * 1.852               # ASOS: TS within 5 NM
        assert w.wind.speed_mps == pytest.approx(22 * 1852 / 3600) and w.wind.from_deg == 180.0
        assert w.wind.gust_factor == pytest.approx(34 / 22)
        assert w.pressure_hpa == pytest.approx(29.81 * 33.86389, abs=0.01)            # A group: inHg
        assert w.humidity_pct == pytest.approx(100 * es_bolton(21) / es_bolton(22), abs=0.5)
        assert p["visibility_m"] == pytest.approx(2 * 1609.344)                       # statute miles
        assert any(g["phenomena"] == ["BR"] for g in p["weather"])                    # mist is parsed, not rain

    @pytest.mark.parametrize("text, ptype, label", [
        ("METAR CYYZ 161300Z 03012KT 1SM -SN BKN010 OVC025 M02/M04 A2995", "snow", None),
        ("METAR KORD 161251Z 07009KT 3SM -FZRA BR OVC006 M01/M02 A3002", "freezing_rain", "slight"),
        ("METAR KDFW 161253Z 21025G40KT 1/2SM +TSGR OVC015CB 24/19 A2970", "hail", None),
        ("METAR EHAM 161255Z 25018KT 9999 SHRA FEW020CB 10/07 Q0998", "showers", "moderate"),
        ("METAR EGPF 161250Z 22008KT 4000 DZ BR OVC004 09/09 Q1010", "drizzle", "moderate drizzle"),
        ("METAR EGPF 161250Z 22008KT 4000 -DZ BR OVC004 09/09 Q1010", "drizzle", "slight drizzle"),
        ("METAR KMIA 161253Z 00000KT 10SM VCTS SCT030CB 28/24 A2992", "none", None),
    ])
    def test_present_weather_groups(self, text, ptype, label):
        _, _, s = _metar(text)
        p = s.weather.precipitation
        assert p.type == ptype
        if label is not None:
            kind = "drizzle" if ptype == "drizzle" else "rain"
            assert U.rain_label(p.rate_mm_h, "wmo", kind) == label
        if ptype == "hail":
            assert s.weather.thunder.enabled and p.hail_size_mm >= 6.4             # GR: hail >= 1/4 in (FMH-1)

    def test_negative_temperatures(self):
        p, _, s = _metar("METAR CYYZ 161300Z 03012KT 1SM -SN BKN010 OVC025 M02/M04 A2995")
        assert (p["temp_c"], p["dew_c"]) == (-2.0, -4.0)
        assert s.weather.humidity_pct == pytest.approx(100 * es_bolton(-4) / es_bolton(-2), abs=0.5)

    def test_calm_and_variable_wind(self):
        _, a, s = _metar("METAR KMIA 161253Z 00000KT 10SM SKC 28/24 A2992")
        assert s.weather.wind.speed_mps == 0.0 and "weather.wind.from_deg" not in a   # calm has no direction
        assert s.weather.precipitation.type == "none" and s.weather.cloud_okta == 0.0
        p, a, s = _metar("METAR KDEN 161253Z VRB03KT 10SM CLR M02/M04 A3010")
        assert p["wind"]["variable"] and p["wind"]["dir_deg"] is None
        assert s.weather.wind.speed_mps == pytest.approx(3 * 1852 / 3600)
        assert "weather.wind.from_deg" not in a and 0.05 <= a["weather.wind.turbulence_intensity"] <= 0.5
        _, a, _ = _metar("METAR LFPG 161230Z 18010KT 150V230 CAVOK 15/08 Q1015 NOSIG")
        assert a["weather.wind.from_deg"] == 180.0 and a["weather.visibility_km"] == 10.0

    def test_thunder_distance_follows_asos(self):
        # ASOS / FMH-1: TS within 5 NM, VCTS 5-10 NM, LTG DSNT 10-30 NM (Plan 2.8)
        _, _, s = _metar("METAR KMIA 161253Z 00000KT 10SM VCTS SCT030CB 28/24 A2992")
        assert 5 * 1.852 <= s.weather.thunder.distance_km <= 10 * 1.852 and s.weather.thunder.enabled
        _, _, s = _metar("METAR KMIA 161253Z 09005KT 10SM SCT030CB 28/24 A2992 RMK AO2 LTG DSNT SE")
        assert 10 * 1.852 <= s.weather.thunder.distance_km <= 30 * 1.852
        assert s.weather.thunder.bearing_deg == 135.0

    def test_remarks(self):
        text = ("METAR KDFW 161253Z 21025G40KT 1/2SM +TSGR SQ OVC015CB 24/19 A2970 "
                "RMK AO2 PK WND 22045/1240 FRQ LTGICCG OHD TS OHD MOV E")
        p, a, s = _metar(text)
        rmk = p["remarks"]
        assert rmk["peak_wind"] == {"dir_deg": 220.0, "speed_kt": 45.0, "time": "1240"}
        assert rmk["lightning"]["frequency"] == "frequent" and rmk["lightning"]["types"] == ["IC", "CG"]
        assert rmk["thunderstorm"]["where"] == "OHD" and rmk["thunderstorm"]["moving"] == "E"
        assert s.weather.thunder.distance_km == 1.0 and s.weather.thunder.flashes_per_min == 3.0
        assert s.weather.wind.gust_factor >= 1.5                                       # squall (FMH-1)
        _, a, _ = _metar("METAR KXYZ 161253Z 00000KT 10SM SCT050 20/10 A2992 RMK TS SE MOV NW")
        assert a["weather.thunder.bearing_deg"] == 135.0 and a["weather.thunder.approach_kmh"] > 0   # cell coming in

    def test_recent_rain_means_aftermath(self):
        _, a, _ = _metar("SPECI KBOS 161312Z AUTO 36005MPS 1 1/2SM RERA OVC008 05/04 A2988")
        assert a["weather.precipitation.type"] == "none" and a["weather.storm_arc"] == "clearing"
        assert a["weather.wind.speed_mps"] == 5.0                                      # MPS unit
        assert a["weather.visibility_km"] == pytest.approx(1.5 * 1.609344)

    def test_metar_detection_and_utc_offset(self):
        assert M.looks_like_metar("EGLL 161250Z 24015G25KT 9999 -RA")
        assert not M.looks_like_metar("heavy rain on a tent")
        s = M.scene_from_metar("EGLL 161250Z 24015KT 9999 -RA 12/10 Q1004", utc_offset_h=1.0)
        assert s.weather.hour == pytest.approx(13 + 50 / 60)


# =====================================================================================
# Timeline
# =====================================================================================
def _lane_scene(keyframes, **weather):
    s = Scene()
    for k, v in weather.items():
        s.set(k, v)
    s.timeline = keyframes
    return s


class TestTimeline:
    def test_linear(self):
        s = _lane_scene([Keyframe(0.0, RATE, 2.0), Keyframe(10.0, RATE, 12.0)])
        assert tl.evaluate(s, 2.5).weather.precipitation.rate_mm_h == pytest.approx(4.5)

    def test_log_is_geometric(self):
        s = _lane_scene([Keyframe(0.0, RATE, 1.0), Keyframe(10.0, RATE, 100.0, "log")])
        assert tl.evaluate(s, 5.0).weather.precipitation.rate_mm_h == pytest.approx(10.0)     # geometric mean
        r = [tl.evaluate(s, t).weather.precipitation.rate_mm_h for t in (2, 4, 6, 8)]
        assert np.allclose(np.diff(np.log(r)), np.log(r[1] / r[0]))                          # constant ratio per s

    def test_hold_and_smooth(self):
        s = _lane_scene([Keyframe(0.0, RATE, 2.0), Keyframe(10.0, RATE, 12.0, "hold")])
        assert tl.evaluate(s, 9.999).weather.precipitation.rate_mm_h == 2.0
        assert tl.evaluate(s, 10.0).weather.precipitation.rate_mm_h == 12.0
        s = _lane_scene([Keyframe(0.0, RATE, 2.0), Keyframe(10.0, RATE, 12.0, "smooth")])
        assert tl.evaluate(s, 2.5).weather.precipitation.rate_mm_h == pytest.approx(2.0 + 10 * 0.15625)  # smoothstep(1/4)
        assert tl.interpolate(0.0, 1.0, 0.5, "smooth") == 0.5

    def test_unsorted_keyframes_and_string_times_and_units(self):
        s = _lane_scene([Keyframe("00:20", RATE, "2 mm/h"), Keyframe(0.0, RATE, 10.0), Keyframe(10.0, RATE, 4.0)])
        assert tl.evaluate(s, 5.0).weather.precipitation.rate_mm_h == pytest.approx(7.0)
        assert tl.evaluate(s, 15.0).weather.precipitation.rate_mm_h == pytest.approx(3.0)
        assert tl.evaluate(s, 99.0).weather.precipitation.rate_mm_h == pytest.approx(2.0)     # holds after the last

    def test_first_keyframe_travels_from_the_scene_value(self):
        s = _lane_scene([Keyframe(10.0, RATE, 8.0, "log")], **{RATE: 2.0})
        assert tl.evaluate(s, 0.0).weather.precipitation.rate_mm_h == pytest.approx(2.0)
        assert tl.evaluate(s, 5.0).weather.precipitation.rate_mm_h == pytest.approx(4.0)
        assert tl.evaluate(s, -1.0).weather.precipitation.rate_mm_h == pytest.approx(2.0)

    def test_circular_values_take_the_short_way(self):
        s = _lane_scene([Keyframe(0.0, "weather.wind.from_deg", 350.0), Keyframe(10.0, "weather.wind.from_deg", 10.0)])
        assert tl.evaluate(s, 5.0).weather.wind.from_deg == pytest.approx(0.0, abs=1e-9)
        assert tl.evaluate(s, 7.5).weather.wind.from_deg == pytest.approx(5.0)

    def test_discrete_values_hold_and_coerce(self):
        s = _lane_scene([Keyframe(0.0, "place.shelter", "porch"), Keyframe(10.0, "place.shelter", "tin roof"),
                         Keyframe(5.0, "weather.thunder.enabled", "on")])
        assert tl.evaluate(s, 9.0).place.shelter == "porch"
        assert tl.evaluate(s, 10.0).place.shelter == "attic"
        assert tl.evaluate(s, 6.0).weather.thunder.enabled is True

    def test_values_are_clamped_to_the_manifest(self):
        s = _lane_scene([Keyframe(0.0, RATE, 400.0), Keyframe(0.0, "place.vegetation.conifer", 7.6)])
        out = tl.evaluate(s, 1.0)
        assert out.weather.precipitation.rate_mm_h == 150.0 and out.place.vegetation["conifer"] == 8

    def test_evaluate_is_pure_and_deterministic(self):
        s = _rich_scene()
        before = s.copy()
        a, b = tl.evaluate(s, 17.0), tl.evaluate(s, 17.0)
        assert a == b and s == before
        assert a.listener.position == pytest.approx((1.0, 2.0, 0.0))

    def test_validate(self):
        good = _rich_scene()
        assert tl.validate(good) == []
        bad = _lane_scene([Keyframe(0.0, "weather.rain_amount", 3.0),
                           Keyframe(1.0, RATE, 5.0, "bouncy"),
                           Keyframe(-2.0, RATE, 5.0),
                           Keyframe(3.0, RATE, "lots"),
                           Keyframe(4.0, RATE, 900.0),
                           Keyframe(5.0, "place.shelter", "spaceship"),
                           Keyframe("soon", RATE, 3.0)])
        problems = tl.validate(bad)
        assert len(problems) == 7, problems
        for word in ("unknown parameter path", "unknown curve", "must be >= 0", "not a value", "outside",
                     "not one of", "bad time"):
            assert any(word in p for p in problems), word
        with pytest.raises(ValueError):
            tl.validate(bad, strict=True)

    def test_storm_arc_keyframes(self):
        kfs = tl.storm_arc_keyframes("full", 10.0, thunder_km=(15.0, 2.0))
        s = Scene()
        s.timeline = kfs
        assert tl.validate(s) == []
        assert tl.evaluate(s, 0.0).weather.storm_arc == "arriving"
        assert tl.evaluate(s, 3 * 60.0 + 1).weather.storm_arc == "steady"              # arriving = 30 % of 10 min
        assert tl.evaluate(s, 3 * 60.0).weather.thunder.distance_km == pytest.approx(2.0)


# =====================================================================================
# Wellness: modes and the governor
# =====================================================================================


def _wild_scene() -> Scene:
    s = preset("hailstorm-car")
    s.weather.thunder.distance_km = 1.0
    s.weather.thunder.flashes_per_min = 8.0
    s.weather.thunder.approach_kmh = 30.0
    s.weather.precipitation.rate_mm_h = 80.0
    s.weather.wind.speed_mps = 25.0
    s.weather.wind.gust_factor = 2.2
    s.place.emitters = [{"type": "chime", "distance_m": 2.0}, {"type": "gutter", "distance_m": 2.0},
                        {"type": "wire", "distance_m": 9.0}]
    return s


def _envelope_db(x: np.ndarray, sr: int, win_s: float = 0.005) -> np.ndarray:
    n = int(win_s * sr)
    frames = x[: x.size // n * n].reshape(-1, n)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)


def _noise_bed(rng, seconds: float, level: float = 0.01) -> np.ndarray:
    from scipy import signal
    w = rng.standard_normal(int(seconds * SR))
    b, a = signal.butter(2, 2000 / (SR / 2))                              # a darker-than-white bed
    y = signal.lfilter(b, a, w)
    return level * y / np.sqrt(np.mean(y ** 2))


class TestModes:
    def test_sleep_caps_and_softens(self):
        s = apply_mode(_wild_scene(), "sleep")
        w = s.weather
        # Plan 2.3 / 8.3 Sleep: rain tops out at heavy; wind <= force 5; thunder >= 8 km, <= 1 per 2-5 min
        assert U.rain_label(w.precipitation.rate_mm_h) in ("slight", "moderate", "heavy")
        assert w.precipitation.type == "rain" and w.precipitation.drops != "fat"
        assert w.precipitation.character == "continuous"
        assert U.beaufort_force(w.wind.speed_mps) <= 5
        assert w.thunder.distance_km >= 8.0 and w.thunder.flashes_per_min <= 0.5
        assert w.thunder.approach_kmh <= 0.0 and w.thunder.stroke_intensity <= 0.4
        assert w.wind.gust_factor <= 1.3
        assert {e["type"] for e in s.place.emitters} == {"gutter"} and s.place.emitters[0]["distance_m"] >= 6.0
        assert s.vibe["volume_db"] <= -18.0 and s.vibe["timer_min"] == 45.0 and s.vibe["descent_min"] == 15.0
        assert s.vibe["startle_guard"] is True                             # "No sudden loud sounds" always on
        guard = s.vibe["guard"]
        assert guard["event_to_bed_db"] == 3.0 and guard["thunder_lowpass_hz"] <= 400.0
        assert guard["thunder_attack_ms"] >= 300.0                         # Plan 8.2: attack >= 200 ms
        assert s.place.biophony["birds"] == "off"

    def test_focus_and_relax(self):
        f = apply_mode(_wild_scene(), "focus")
        assert not f.weather.thunder.enabled and f.place.emitters == []    # Plan 8.3: no thunder, no rhythmic gutters
        assert f.place.biophony["birds"] == "off" and f.weather.wind.gust_factor <= 1.25
        assert f.vibe["guard"]["no_pitched_drips"]
        r = apply_mode(_wild_scene(), "relax")
        assert r.mode == "calm" and r.weather.thunder.distance_km >= 8.0 and r.weather.thunder.flashes_per_min <= 0.5
        assert r.weather.precipitation.rate_mm_h == 80.0                    # Relax keeps full ranges

    def test_asmr_triggers_are_close_and_removable(self):
        s = Scene()
        s.vibe = {"asmr_exclude": ["tin"]}
        a = apply_mode(s, "asmr")
        assert a.listener.render == "binaural"
        objs = [o for o in a.place.objects if o.params.get("asmr")]
        names = {o.params["trigger"] for o in objs}
        assert "tin" not in names and names
        for o in objs:
            assert 0.05 <= o.params["distance_m"] <= 0.5                    # Plan 8.3: 5-50 cm
            assert 0.0 <= o.params["motion_cm_s"] <= 8.0
        back = apply_mode(a, "none")
        assert not [o for o in back.place.objects if o.params.get("asmr")]

    def test_mode_names(self):
        assert canonical_mode("Relax") == "calm" and canonical_mode("Cinematic") == "none"
        with pytest.raises(ValueError):
            canonical_mode("party")

    def test_apply_mode_is_pure_and_idempotent(self):
        s = _wild_scene()
        before = s.copy()
        once = apply_mode(s, "sleep")
        assert s == before and apply_mode(once, "sleep") == once


class TestGovernor:
    def test_sleep_softens_a_sudden_transient(self, rng):
        x = _noise_bed(rng, 3.2)
        t0 = int(2.8 * SR)
        x[t0:t0 + int(0.03 * SR)] += 0.3 * rng.standard_normal(int(0.03 * SR))   # ~+30 dB burst
        out = {}
        for mode in ("none", "sleep"):
            g = CalmGovernor(SR, mode)
            y = g.process(x.copy())
            env = _envelope_db(y[g.latency:] if g.latency else y, SR)
            bed = np.median(env[int(2.0 / 0.005):int(2.75 / 0.005)])
            peak = env[int(2.79 / 0.005):int(2.85 / 0.005)].max()
            out[mode] = peak - bed
            assert np.all(np.isfinite(y))
        assert out["none"] > 25.0
        # Plan 8.2: event-to-bed <= 3 dB in Sleep; +3 dB allowance for the 5 ms vs 10 ms envelope estimators
        assert out["sleep"] <= GOVERNOR_SETTINGS["sleep"]["event_to_bed_db"] + 3.0, out

    def test_never_boosts_and_eases_a_loud_stream(self, rng):
        x = _noise_bed(rng, 4.0, level=0.3)                                # far above the -30 LUFS Sleep ceiling
        g = CalmGovernor(SR, "sleep")
        gains = []
        for k in range(0, x.size, 4800):
            g.process(x[k:k + 4800])
            gains.append(g.metrics.get("gain_db", 0.0))
        assert max(gains) <= 0.0
        # slew 0.25 dB/s (Sleep): ~-0.9 dB after 4 s, monotonic
        assert gains[-1] == pytest.approx(-0.25 * 4.0 + 0.1, abs=0.2)
        assert all(b <= a + 1e-12 for a, b in itertools.pairwise(gains))
        assert g.badge() is None or "Sleep" in g.badge()

    def test_block_size_independence(self, rng):
        x = np.stack([_noise_bed(rng, 1.5), _noise_bed(rng, 1.5)])
        ref = CalmGovernor(SR, "sleep").process(x)
        for block in (1024, 777, 64):
            g = CalmGovernor(SR, "sleep")
            y = np.concatenate([g.process(x[:, k:k + block]) for k in range(0, x.shape[1], block)], axis=1)
            assert np.allclose(y, ref, atol=1e-12, rtol=0), block

    def test_no_clicks_at_hop_boundaries(self):
        t = np.arange(int(3.0 * SR)) / SR
        x = 0.05 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 5000 * t)   # bright: the shelf moves
        g = CalmGovernor(SR, "focus")
        y = g.process(x)
        assert abs(g.metrics["tilt_correction_db"]) > 0.3                   # the correction did move
        d2 = np.abs(np.diff(y, 2))
        hop = g.hop
        edges = np.zeros(d2.size, bool)
        for k in range(hop, d2.size, hop):
            edges[max(0, k - 4):k + 4] = True
        steady = slice(int(0.5 * SR), None)
        # a coefficient switch or gain step would stand out in the second difference of a smooth signal
        assert d2[steady][edges[steady]].max() <= 1.05 * d2[steady][~edges[steady]].max()

    def test_loudness_proxy_sums_channels_like_bs1770(self, rng):
        x = _noise_bed(rng, 1.0, level=0.05)
        mono, stereo = CalmGovernor(SR, "calm"), CalmGovernor(SR, "calm")
        mono.process(x)
        stereo.process(np.stack([x, x]))
        # BS.1770-4 eq. 2: L = -0.691 + 10 log10(sum_i G_i z_i); two equal channels read +3.01 dB
        assert stereo.metrics["loudness_db"] - mono.metrics["loudness_db"] == pytest.approx(10 * math.log10(2), abs=1e-6)
        assert mono.metrics["loudness_db"] == pytest.approx(20 * math.log10(0.05) - 0.691, abs=0.3)

    def test_output_has_no_dc(self, rng):
        x = np.stack([_noise_bed(rng, 2.0), _noise_bed(rng, 2.0)])
        y = CalmGovernor(SR, "sleep").process(x)
        assert np.all(np.abs(y.mean(axis=1)) < 0.02 * np.sqrt(np.mean(y ** 2)))   # shelf and gains add no offset

    def test_silence_stays_silent_and_finite(self):
        g = CalmGovernor(SR, "sleep")
        y = g.process(np.zeros((2, SR)))
        assert np.all(y == 0.0)

    def test_timer_descent_and_stop(self, rng):
        x = _noise_bed(rng, 13.0)
        g = CalmGovernor(SR, "sleep", timer_min=2.0 / 60.0, descent_min=1.0 / 60.0)
        y = g.process(x)
        lvl = lambda a, b: 10 * np.log10(np.mean(y[int(a * SR):int(b * SR)] ** 2) + 1e-30)
        drop = lvl(1.95, 2.0) - lvl(0.5, 0.95)
        assert -16.0 <= drop <= -10.0                                       # Plan 8.3: -10 to -15 dB fade (descent -12)
        assert g.finished and np.all(y[int(12.05 * SR):] == 0.0)            # silence 10 s after the timer
        assert g.badge() == "Timer finished"

    def test_none_mode_is_a_bypass(self, rng):
        x = _noise_bed(rng, 0.5)
        assert np.array_equal(CalmGovernor(SR, "none").process(x), x)


# =====================================================================================
# Wellness: dose meter (ITU-T H.870) and claims linter
# =====================================================================================


class TestDose:
    def test_h870_reference_exposures(self):
        assert dose_pa2h(80.0, 40.0) == pytest.approx(1.6, rel=1e-3)        # ITU-T H.870 adult: 80 dBA x 40 h
        assert dose_pa2h(75.0, 40.0) == pytest.approx(0.51, rel=0.01)       # child / sensitive: 75 dBA x 40 h
        assert ALLOWANCE_PA2H == {"adult": 1.6, "child": 0.51}

    def test_equal_energy_exchange(self):
        # 3 dB exchange rate: +3 dB halves the time for the same dose (equal-energy principle)
        assert dose_pa2h(83.0, 20.0) == pytest.approx(dose_pa2h(80.0, 40.0), rel=0.005)
        assert hours_until(1.6, 80.0) == pytest.approx(40.0, rel=1e-3)
        assert hours_until(1.6, 90.0) == pytest.approx(4.0, rel=1e-3)
        assert hours_until(1.6, 100.0) == pytest.approx(0.4, rel=1e-3)

    @pytest.mark.parametrize("f, a_db, tol", [
        (31.5, -39.4, 0.5), (63, -26.2, 0.3), (125, -16.1, 0.2), (250, -8.6, 0.2), (500, -3.2, 0.2),
        (1000, 0.0, 0.01), (2000, 1.2, 0.2), (4000, 1.0, 0.3), (8000, -1.1, 1.0),
    ])
    def test_a_weighting_matches_iec_61672(self, f, a_db, tol):
        # IEC 61672-1 Table 3 nominal A-weighting (bilinear warping near Nyquist stays inside class 1)
        assert 20 * math.log10(_sos_gain(a_weighting_sos(SR), f, SR)) == pytest.approx(a_db, abs=tol)

    def test_meter_reads_80_dba_and_doses_correctly(self):
        m = DoseMeter(SR, channels=2, calibration_db_spl=100.0, start_time_s=1.0e9)
        t = np.arange(SR) / SR
        x = np.stack([0.1 * np.sin(2 * np.pi * 1000 * t)] * 2)            # -20 dB re full-scale sine -> 80 dB SPL
        for _ in range(4):
            m.process(x)
        assert m.laeq_db() == pytest.approx(80.0, abs=0.1)
        assert m.weekly_dose() == pytest.approx(1.6 * (4.0 / 3600.0) / 40.0, rel=0.01)
        assert m.percent() == pytest.approx(100 * 4.0 / (40 * 3600), rel=0.01)
        assert m.remaining_hours(80.0) == pytest.approx(40.0 - 4.0 / 3600.0, rel=1e-3)
        assert not m.exceeded() and m.recommended_gain() == 1.0

    def test_any_block_length_gives_the_same_dose(self, rng):
        x = rng.standard_normal((2, SR)) * 0.1
        whole = DoseMeter(SR, start_time_s=0.0)
        whole.process(x)
        split = DoseMeter(SR, start_time_s=0.0)
        for k in range(0, SR, 333):
            split.process(x[:, k:k + 333])
        # each ear is integrated on its own, so the louder-ear dose does not depend on the block length
        assert split.session_dose == pytest.approx(whole.session_dose, rel=1e-9)
        assert split.weekly_dose() == pytest.approx(whole.weekly_dose(), rel=1e-9)
        assert split.short_term_laeq_db() == pytest.approx(whole.short_term_laeq_db(), abs=1e-9)

    def test_louder_ear_is_metered(self):
        m = DoseMeter(SR, channels=2, calibration_db_spl=100.0, start_time_s=0.0)
        t = np.arange(SR) / SR
        m.process(np.stack([0.1 * np.sin(2 * np.pi * 1000 * t), np.zeros(SR)]))
        assert m.laeq_db() == pytest.approx(80.0, abs=0.1)

    def test_rolling_week_and_limit(self):
        m = DoseMeter(SR, channels=1, calibration_db_spl=130.0, start_time_s=0.0)
        t = np.arange(SR) / SR
        x = np.sin(2 * np.pi * 1000 * t)                                    # full scale = 130 dB: 1.1 Pa^2h per s
        t0 = 10 * 86400.0
        m.process(x, now_s=t0)
        m.process(x, now_s=t0 + 1)
        assert m.exceeded(now_s=t0 + 2)
        assert m.recommended_gain(now_s=t0 + 2) == pytest.approx(10 ** ((80.0 - 130.0) / 20), rel=0.03)
        assert m.weekly_dose(now_s=t0 + 6 * 86400) > 0.0
        assert m.weekly_dose(now_s=t0 + 8 * 86400) == 0.0                   # out of the rolling 7 days
        back = DoseMeter.from_dict(m.to_dict(), sr=SR, channels=1)
        assert back.weekly_dose(now_s=t0 + 2) == pytest.approx(m.weekly_dose(now_s=t0 + 2))

    def test_child_profile(self):
        m = DoseMeter(SR, profile="child", start_time_s=0.0)
        assert m.allowance == 0.51 and m.reference_dba == 75.0
        with pytest.raises(ValueError):
            DoseMeter(SR, profile="robot")


class TestClaims:
    @pytest.mark.parametrize("text", ["cures insomnia", "treats anxiety", "clinically proven to make you sleep",
                                      "reduces tinnitus", "boosts deep sleep", "helps with ADHD",
                                      "the perfect sleep sound for babies", "a tinnitus masker"])
    def test_flags_medical_claims(self, text):
        from raingen.wellness.claims import is_clean, lint_text
        assert not is_clean(text)                                           # FDA General Wellness (2026) disallows these
        assert any(f["severity"] == "error" for f in lint_text(text))

    @pytest.mark.parametrize("text", ["relaxing rain sounds", "may help you unwind", "masks distractions so you can focus",
                                      "a calm sound for your bedtime routine", "Steady rain on leaves, night"])
    def test_allows_wellness_language(self, text):
        from raingen.wellness.claims import is_clean, lint_text
        assert is_clean(text) and not lint_text(text)

    def test_presets_pass_and_ignore_marker(self):
        from raingen.wellness.claims import lint_presets, lint_text, wellness_terms
        assert lint_presets() == []
        assert lint_text("cures insomnia  # claims: ignore") == []
        assert "unwind" in wellness_terms("helps you unwind and relax")
        f = lint_text("Great rain.\nIt treats anxiety.")[0]
        assert (f["line"], f["column"]) == (2, 4)


# =====================================================================================
# Live providers (network mocked)
# =====================================================================================


class _FakeNet:
    """Stands in for urllib.request.urlopen: records requests and replies from a queue."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        body = r if isinstance(r, bytes) else (r.encode() if isinstance(r, str) else json.dumps(r).encode())
        return io.BytesIO(body)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("tests must not open network sockets")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    return monkeypatch


def _om_payload(code=61, precip=0.5, rain=0.5, showers=0.0, snow=0.0, interval=900, wind=5.0, gust=9.0):
    return {"latitude": 51.5, "longitude": -0.12, "timezone": "Europe/London", "utc_offset_seconds": 0,
            "current_units": {"time": "iso8601", "interval": "seconds", "precipitation": "mm"},
            "current": {"time": "2026-11-16T22:15", "interval": interval, "precipitation": precip, "rain": rain,
                        "showers": showers, "snowfall": snow, "weather_code": code, "wind_speed_10m": wind,
                        "wind_gusts_10m": gust, "wind_direction_10m": 225, "temperature_2m": 11.3,
                        "relative_humidity_2m": 93, "cloud_cover": 100, "is_day": 0}}


class TestLive:
    def test_wmo_code_table(self):
        assert len(LP.WMO_CODES) == 29                                      # Plan 2.8: 28 codes + 97 heavy thunderstorm
        assert all(LP.WMO_CODES[c]["thunder"] for c in (95, 96, 97, 99))
        assert LP.WMO_CODES[97]["thunder"] and LP.WMO_CODES[97].get("heavy")

    def test_open_meteo_parsing_into_a_scene(self, no_network):
        net = _FakeNet([_om_payload()])
        no_network.setattr(urllib.request, "urlopen", net)
        obs = LP.OpenMeteoProvider().current(51.50741, -0.12784)
        assert obs["rain_mm_h"] == pytest.approx(2.0)                       # 0.5 mm per 900 s interval -> 2 mm/h
        assert obs["hour"] == pytest.approx(22.25) and obs["month"] == 11
        s = LP.scene_from_observation(obs, name="London")
        w = s.weather
        assert w.precipitation.type == "rain" and w.precipitation.rate_mm_h == pytest.approx(2.0)
        assert w.wind.speed_mps == 5.0 and w.wind.from_deg == 225.0 and w.wind.gust_factor == pytest.approx(1.8)
        assert w.temperature_c == 11.3 and w.humidity_pct == 93.0 and w.cloud_okta == 8.0
        assert w.season == "autumn" and w.latitude_deg == 51.5 and not w.thunder.enabled
        assert s.name == "Live: London (slight rain)"
        assert _check_in_manifest(s, "live") == []

    def test_request_url_user_agent_and_privacy(self, no_network):
        net = _FakeNet([_om_payload()])
        no_network.setattr(urllib.request, "urlopen", net)
        LP.OpenMeteoProvider().current(51.50741, -0.12784)
        req = net.requests[0]
        ua = req.get_header("User-agent")
        assert ua == LP.USER_AGENT and ua.startswith("raingen/") and "https://" in ua
        url = req.full_url
        assert "latitude=51.5000" in url and "longitude=-0.1000" in url    # rounded to 0.1 deg (~10 km, Plan 2.8)
        assert "wind_speed_unit=ms" in url and "precipitation_unit=mm" in url and "apikey" not in url
        assert url.startswith(LP.OpenMeteoProvider.FREE_URL)
        keyed = LP.OpenMeteoProvider(api_key="k", coarse_location=False).url(51.50741, -0.12784)
        assert keyed.startswith(LP.OpenMeteoProvider.CUSTOMER_URL) and "apikey=k" in keyed and "51.5074" in keyed

    @pytest.mark.parametrize("code, heavy", [(95, False), (97, True)])
    def test_thunder_codes(self, code, heavy):
        obs = LP.demo_observation(code)
        s = LP.scene_from_observation(obs)
        th = s.weather.thunder
        assert th.enabled
        lo, hi = (1.0, 6.0) if heavy else (2.0, 12.0)
        assert lo <= th.distance_km <= hi and 0.0 <= th.stroke_intensity <= 1.0
        assert _check_in_manifest(s, f"code {code}") == []

    def test_snow_fog_and_dry(self):
        snow = LP.demo_observation(73) | {"snow_water_mm_h": 1.0, "rain_mm_h": 0.0, "precip_mm_h": 1.0, "month": 1}
        s = LP.scene_from_observation(snow)
        assert s.weather.precipitation.type == "snow" and s.weather.season == "winter"
        fog = LP.scene_from_observation(LP.demo_observation(45) | {"precip_mm_h": 0.0, "rain_mm_h": 0.0})
        assert fog.weather.precipitation.type == "none" and fog.weather.visibility_km == 0.2
        south = LP.scene_from_observation(LP.demo_observation(61) | {"lat": -33.9, "month": 1})
        assert south.weather.season == "summer"                              # southern hemisphere January

    def test_snowfall_cm_to_water(self, no_network):
        net = _FakeNet([_om_payload(code=73, precip=1.0, rain=0.0, snow=0.7, interval=3600)])
        no_network.setattr(urllib.request, "urlopen", net)
        obs = LP.OpenMeteoProvider().current(60.0, 10.0)
        assert obs["snow_water_mm_h"] == pytest.approx(1.0)                  # Open-Meteo: cm of snow / 7 x 10 = mm water

    def test_subgrid_detail_is_seeded(self):
        obs = LP.demo_observation(95)
        a, b = LP.scene_from_observation(obs), LP.scene_from_observation(obs)
        assert a == b
        other = Scene(seed=7)
        assert LP.scene_from_observation(obs, base=other).weather.thunder.distance_km != a.weather.thunder.distance_km

    @pytest.mark.parametrize("reply, words", [
        (urllib.error.HTTPError("https://api.open-meteo.com/v1/forecast", 429, "Too Many", {}, io.BytesIO(b"limit")),
         "HTTP 429"),
        (urllib.error.URLError("no route"), "cannot reach"),
        (b"<html>not json", "bad JSON"),
        ({"latitude": 1.0}, "no 'current'"),
    ])
    def test_errors_become_live_weather_errors(self, no_network, reply, words):
        no_network.setattr(urllib.request, "urlopen", _FakeNet([reply]))
        with pytest.raises(LP.LiveWeatherError, match=words):
            LP.OpenMeteoProvider().current(51.5, -0.1)

    def test_scene_from_live_with_place_name(self, no_network):
        geo = {"results": [{"latitude": 53.8, "longitude": -1.55, "name": "Leeds", "country": "UK",
                            "timezone": "Europe/London"}]}
        net = _FakeNet([geo, _om_payload(code=80, rain=0.0, showers=0.5, precip=0.5)])
        no_network.setattr(urllib.request, "urlopen", net)
        s = LP.scene_from_live("Leeds")
        assert s.name.startswith("Live: Leeds") and s.weather.precipitation.type == "showers"
        assert s.weather.precipitation.character == "showery"
        assert "geocoding-api.open-meteo.com" in net.requests[0].full_url
        no_network.setattr(urllib.request, "urlopen", _FakeNet([{"results": []}]))
        with pytest.raises(LP.LiveWeatherError, match="no place"):
            LP.geocode("Atlantis")

    def test_metar_provider(self, no_network):
        raw = "KJFK 161251Z 18022G34KT 2SM +TSRA BR BKN008CB 22/21 A2981"
        net = _FakeNet([raw + "\n"])
        no_network.setattr(urllib.request, "urlopen", net)
        s = LP.scene_from_live("KJFK", provider="metar")
        assert s.name.startswith("Live METAR KJFK") and s.weather.thunder.enabled
        assert "ids=KJFK" in net.requests[0].full_url
        assert net.requests[0].get_header("User-agent") == LP.USER_AGENT     # aviationweather.gov asks for a UA
        rows = [{"icaoId": "KLGA", "lat": 40.78, "lon": -73.88, "rawOb": "KLGA 161251Z 18010KT 10SM -RA 20/18 A2990"},
                {"icaoId": "KJFK", "lat": 40.64, "lon": -73.78, "rawOb": raw}]
        no_network.setattr(urllib.request, "urlopen", _FakeNet([rows]))
        assert LP.MetarProvider().nearest(40.64, -73.78)[0] == "KJFK"
        no_network.setattr(urllib.request, "urlopen", _FakeNet([""]))
        with pytest.raises(LP.LiveWeatherError):
            LP.MetarProvider().fetch_raw("KXXX")


# =====================================================================================
# OSC control
# =====================================================================================
class _FakeEngine:
    def __init__(self):
        self.calls = []
        self.loaded = []
        self._scene = Scene()
        self.graph = ParameterGraph(self._scene)

    def set(self, path, value, ramp=None):
        self.calls.append((path, value, ramp))

    @property
    def scene(self):
        return self._scene.copy()

    def set_scene(self, scene):
        self.loaded.append(scene)


def calls_tail(eng, n):
    return eng.calls[-n:]


def _send(disp, address, *args):
    from pythonosc.osc_message_builder import OscMessageBuilder
    b = OscMessageBuilder(address=address)
    for a in args:
        b.add_arg(a)
    disp.call_handlers_for_packet(b.build().dgram, ("127.0.0.1", 9))


class TestOSC:
    @pytest.fixture
    def rig(self):
        from raingen.control.osc import build_dispatcher
        eng = _FakeEngine()
        errors = []
        disp = build_dispatcher(eng, on_error=lambda addr, exc: errors.append((addr, exc)))
        return eng, disp, errors

    def test_every_manifest_address_is_mapped(self, rig):
        _, disp, _ = rig
        for addr in mf.osc_namespace():
            assert list(disp.handlers_for_address(addr)), addr

    def test_values_are_coerced_clamped_and_ramped(self, rig):
        eng, disp, errors = rig
        _send(disp, "/rg/weather/precipitation/rate_mm_h", 8.0)
        _send(disp, "/rg/weather/wind/speed_mps", "30 km/h", 2.0)           # units in strings, ramp seconds
        _send(disp, "/rg/weather/wind/from_deg", "NW")
        _send(disp, "/rg/weather/precipitation/rate_mm_h", 900.0)           # clamped to the 150 mm/h top
        _send(disp, "/rg/weather/thunder/enabled", True)
        _send(disp, "/rg/place/shelter", "tin roof")
        _send(disp, "/rg/listener/position", 1.0, 2.0, 0.5)
        _send(disp, "/rg/weather/precipitation/rate_mm_h/norm", 0.5)
        _send(disp, "/rg/mode", "Relax")
        _send(disp, "/rg/vibe/intensity/norm", 0.25)
        assert not errors, errors
        assert calls_tail(eng, 2) == [("mode", "calm", None), ("vibe.intensity", 25.0, None)]
        calls = eng.calls
        assert calls[0] == (RATE, 8.0, None)
        assert calls[1][0] == WIND and calls[1][1] == pytest.approx(30 / 3.6) and calls[1][2] == 2.0
        assert calls[2] == ("weather.wind.from_deg", 315.0, None)
        assert calls[3] == (RATE, 150.0, None)
        assert calls[4] == ("weather.thunder.enabled", True, None)
        assert calls[5] == ("place.shelter", "attic", None)
        assert calls[6] == ("listener.position", (1.0, 2.0, 0.5), None)
        assert calls[7][1] == pytest.approx(math.sqrt(0.02 * 150.0))       # norm 0.5 = geometric midpoint

    def test_bad_packets_are_reported_not_raised(self, rig):
        eng, disp, errors = rig
        _send(disp, "/rg/place/shelter", "spaceship")
        _send(disp, "/rg/weather/wind/speed_mps", "a lot")
        _send(disp, "/rg/listener/position", 1.0)
        _send(disp, "/rg/nonexistent/thing", 1.0)                            # unmapped: ignored quietly
        assert len(errors) == 3 and eng.calls == []

    def test_commands(self, rig):
        eng, disp, errors = rig
        _send(disp, "/rg/preset", "attic-tin-roof")
        _send(disp, "/rg/text", "heavy rain on a tent")
        _send(disp, "/rg/metar", "EGLL 161250Z 24015G25KT 9999 -RA 12/10 Q1004")
        assert not errors, errors
        assert eng.loaded[0].place.shelter == "attic" and eng.loaded[1].place.shelter == "tent"
        assert eng.loaded[2].weather.wind.from_deg == 240.0
        _send(disp, "/rg/pin", WIND, "force 3")
        _send(disp, "/rg/offset", RATE, 1.0)
        assert eng.graph.label(WIND) == "Locked" and eng.graph.ratio(RATE) == pytest.approx(2.0)
        _send(disp, "/rg/relink", WIND)
        assert eng.graph.state(WIND) == "auto"
        _send(disp, "/rg/preset", "no-such-scene")
        assert len(errors) == 1

    def test_bundle(self, rig):
        from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
        from pythonosc.osc_message_builder import OscMessageBuilder
        eng, disp, _ = rig
        bb = OscBundleBuilder(IMMEDIATELY)
        for addr, v in (("/rg/weather/temperature_c", 4.0), ("/rg/weather/humidity_pct", 70.0)):
            mb = OscMessageBuilder(address=addr)
            mb.add_arg(v)
            bb.add_content(mb.build())
        disp.call_handlers_for_packet(bb.build().dgram, ("127.0.0.1", 9))
        assert [c[0] for c in eng.calls] == ["weather.temperature_c", "weather.humidity_pct"]

    def test_loopback_server_on_an_ephemeral_port(self):
        from pythonosc.udp_client import SimpleUDPClient

        from raingen.control.osc import serve_osc
        eng = _FakeEngine()
        with serve_osc(eng, host="127.0.0.1", port=0) as handle:
            host, port = handle.address
            assert host == "127.0.0.1" and port > 0
            SimpleUDPClient("127.0.0.1", port).send_message("/rg/weather/precipitation/rate_mm_h", 12.0)
            deadline = time.monotonic() + 3.0
            while not eng.calls and time.monotonic() < deadline:
                time.sleep(0.01)
        assert eng.calls == [(RATE, 12.0, None)]
