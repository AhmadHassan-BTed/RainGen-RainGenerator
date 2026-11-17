import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import threading

import numpy as np
import pytest
import pygame
import rain_generator.audio as audio_mod
from rain_generator.config import Config
from rain_generator.simulator import ThunderstormSimulator, parse_args, main
from rain_generator.presets import PRESETS

@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()

def test_parse_args_defaults():
    args = parse_args([])
    assert args.preset == "steady"
    assert args.width == 1080
    assert args.height == 720
    assert not args.fullscreen
    assert not args.mute

def test_parse_args_custom():
    args = parse_args(["--preset", "thunderstorm", "--fullscreen", "--mute", "-W", "800", "-H", "600"])
    assert args.preset == "thunderstorm"
    assert args.fullscreen
    assert args.mute
    assert args.width == 800
    assert args.height == 600

def test_simulator_lifecycle():
    config = Config(screen_width=640, screen_height=480, is_muted=True)
    sim = ThunderstormSimulator(config=config, preset_name="drizzle")

    assert sim.current_preset_key == "drizzle"
    assert sim.width == 640
    assert sim.height == 480

    # Preset cycle
    sim.cycle_preset()
    assert sim.current_preset_key != "drizzle"

    # Mute toggle
    assert sim.config.is_muted
    sim.toggle_mute()
    assert not sim.config.is_muted

    # Manual lightning strike
    sim.trigger_manual_lightning()
    assert sim.lightning.in_strike

    # Resize
    sim._on_resize(800, 600)
    assert sim.width == 800
    assert sim.height == 600

    # Simulation step
    sim.update(0.016)

    # Cleanup
    sim.cleanup()


# ---------------------------------------------------------------------------
# headless app: the real event loop on the dummy SDL drivers
# ---------------------------------------------------------------------------
def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0)


def test_headless_run_loop(monkeypatch):
    """ThunderstormSimulator.run() for a dozen frames with keyboard input, modals, a resize
    and a lightning strike, then a QUIT event: no exceptions, clean shutdown."""
    monkeypatch.setattr(audio_mod, "HAS_SOUNDDEVICE", False)          # never open a real output stream
    config = Config(screen_width=480, screen_height=320, is_muted=True, fps=240)
    sim = ThunderstormSimulator(config=config, preset_name="thunderstorm")
    script = {
        2: [_key(pygame.K_l), _key(pygame.K_SPACE), _key(pygame.K_SPACE), _key(pygame.K_2), _key(pygame.K_s)],
        4: [_key(pygame.K_ESCAPE)],                                    # closes settings
        5: [_key(pygame.K_g)],
        7: [_key(pygame.K_ESCAPE)],                                    # closes the generator (nothing exported)
        8: [_key(pygame.K_SLASH)],
        10: [_key(pygame.K_SLASH), _key(pygame.K_h), _key(pygame.K_m)],
        11: [pygame.event.Event(pygame.VIDEORESIZE, w=640, h=400, size=(640, 400))],
        14: [pygame.event.Event(pygame.QUIT)],
    }
    seen = {"frames": 0, "settings": False, "generator": False, "help": False}
    update = sim.update

    def scripted_update(dt):
        update(dt)
        seen["frames"] += 1
        seen["settings"] |= sim.settings_modal.is_open
        seen["generator"] |= sim.generator_modal.is_open
        seen["help"] |= sim.help_modal.is_open
        for ev in script.get(seen["frames"], ()):
            pygame.event.post(ev)
        assert seen["frames"] < 100, "QUIT was not handled"

    monkeypatch.setattr(sim, "update", scripted_update)
    sim.run()
    assert not sim.running
    assert 14 <= seen["frames"] < 100
    assert seen["settings"] and seen["generator"] and seen["help"]       # every modal was drawn
    assert sim.current_preset_key == "steady"                          # key 2
    assert (sim.width, sim.height) == (640, 400)
    assert not sim.config.is_muted                                     # toggled once with M
    assert not pygame.get_init()                                       # cleanup ran


def test_headless_frames_with_storm(monkeypatch):
    """Direct update/draw loop with automatic strikes due every frame and thunder audio."""
    monkeypatch.setattr(audio_mod, "HAS_SOUNDDEVICE", False)
    config = Config(screen_width=320, screen_height=240, is_muted=False)
    sim = ThunderstormSimulator(config=config, preset_name="gale")
    strikes = []
    sim.lightning.on_thunder_trigger = lambda delay, vol: strikes.append((delay, vol))
    try:
        for i in range(30):
            sim.next_strike_time = 0.0 if i % 10 == 0 else sim.next_strike_time
            for event in pygame.event.get():
                sim.handle_event(event)
            sim.update(1.0 / 60.0)
            sim.draw()
        assert strikes, "lightning never asked for thunder"
        assert all(d >= 0.0 and 0.0 < v <= 2.0 for d, v in strikes)
        sim.audio.play_thunder(delay=0.0, volume_scale=1.0)             # WAV clips from sounds/
        sim.audio.process_delayed_thunders()
        out = np.zeros((1024, 2), dtype=np.float32)
        sim.audio._audio_callback(out, 1024, None, None)
        assert np.all(np.isfinite(out)) and np.any(out != 0.0)
    finally:
        sim.cleanup()


def test_export_from_the_app_runs_in_the_background(monkeypatch):
    calls = []
    done = threading.Event()

    def fake_export(**kw):
        calls.append(kw)
        done.set()
        return kw["filepath"]

    monkeypatch.setattr(audio_mod, "HAS_SOUNDDEVICE", False)
    monkeypatch.setattr(audio_mod, "export_audio_file", fake_export)
    sim = ThunderstormSimulator(config=Config(screen_width=320, screen_height=240, is_muted=True),
                                preset_name="drizzle")
    try:
        sim.export_audio(12.0, preset_name="cozy", filepath="x.wav")
        assert done.wait(10.0)
        assert calls == [dict(filepath="x.wav", duration=12.0, preset_name="cozy", volumes=sim.config.volumes)]
    finally:
        sim.cleanup()


def test_cli_export_path(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_export(filepath, duration, preset_name):
        calls.append((filepath, duration, preset_name))
        return str(tmp_path / "out.wav")

    monkeypatch.setattr(audio_mod, "export_audio_file", fake_export)
    main(["--export", str(tmp_path / "out.wav"), "-d", "3.5", "-i", "High"])
    assert calls == [(str(tmp_path / "out.wav"), 3.5, "thunderstorm")]   # intensity High -> thunderstorm
    assert "Exported audio file" in capsys.readouterr().out


def test_every_preset_runs_headless(monkeypatch):
    monkeypatch.setattr(audio_mod, "HAS_SOUNDDEVICE", False)
    sim = ThunderstormSimulator(config=Config(screen_width=320, screen_height=240, is_muted=True))
    try:
        for key in PRESETS:
            sim.apply_preset(key)
            assert sim.current_preset_key == key
            assert len(sim.rain.drops) == PRESETS[key].drop_count
            for _ in range(3):
                sim.update(1.0 / 30.0)
                sim.draw()
            # the live engine follows the preset
            assert sim.audio.rain_intensity_scale == pytest.approx(max(0.1, PRESETS[key].rain_audio_intensity))
    finally:
        sim.cleanup()
