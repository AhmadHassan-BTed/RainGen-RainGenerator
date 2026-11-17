import pytest
from rain_generator.presets import PRESETS, INTENSITY_TO_PRESET, PRESET_KEYS, StormPreset

def test_preset_definitions():
    assert len(PRESETS) >= 5
    for key, preset in PRESETS.items():
        assert isinstance(preset, StormPreset)
        assert preset.drop_count > 0
        assert preset.rain_speed_min < preset.rain_speed_max
        assert preset.thunder_interval_range[0] < preset.thunder_interval_range[1]
        assert preset.rain_audio_intensity > 0
        assert preset.drop_audio_rate > 0
        assert 0.0 <= preset.cloud_density <= 1.0

def test_intensity_mapping():
    for intensity, preset_key in INTENSITY_TO_PRESET.items():
        assert preset_key in PRESETS

def test_preset_keys():
    for key in PRESET_KEYS:
        assert key in PRESETS


def test_cli_and_ui_offer_exactly_the_presets():
    from rain_generator.simulator import parse_args
    from rain_generator.ui import AudioGeneratorModal
    from rain_generator.config import UIColors

    for key in PRESETS:
        assert parse_args(["--preset", key]).preset == key
    assert AudioGeneratorModal(UIColors()).preset_keys == PRESET_KEYS


def test_intensity_ladder_is_monotonic():
    """Low -> Extreme must mean more and faster rain, more wind and more frequent thunder."""
    ladder = [PRESETS[INTENSITY_TO_PRESET[k]] for k in ("Low", "Medium", "High", "Extreme")]
    for a, b in zip(ladder, ladder[1:]):
        assert a.drop_count < b.drop_count
        assert a.rain_speed_max < b.rain_speed_max
        assert a.rain_audio_intensity < b.rain_audio_intensity
        assert a.drop_audio_rate < b.drop_audio_rate
        assert a.wind_audio_intensity < b.wind_audio_intensity
        assert a.thunder_interval_range[1] > b.thunder_interval_range[1]
