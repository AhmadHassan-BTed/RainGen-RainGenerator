"""End-to-end integration tests: render, determinism, master chain, stems, captions, engine, CLI, core.

Two kinds of programme are used:

* real physics scenes (presets and a light open-meadow rain scene at 'mobile' quality),
  kept to 0.25-2 s because synthesis is slow;
* synthetic test generators installed in place of the physics generators (the World,
  encoder, shelter, decoders and the whole master chain stay real). They make exact
  contract checks cheap: an impulse at a known scene sample must land on that output
  sample, stems must sum to the mix, a stationary noise must hit a loudness target.

Every numeric tolerance names its source; "contract" means the behaviour documented in the
module docstrings (raingen/render.py, raingen/engine.py) or docs/MASTER_PLAN.md.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from raingen.core.constants import BLOCK, SAMPLE_RATE
from raingen.core.types import BlockContext, Emission, EventMarker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYOUT_CHANNELS = {"stereo": 2, "binaural": 2, "5.1": 6, "7.1": 8, "7.1.4": 12, "foa": 4, "hoa3": 16, "mono": 1}
LSB24 = 2.0 ** -23


# ---------------------------------------------------------------------------
# scenes and synthetic generators
# ---------------------------------------------------------------------------
def light_scene(seed: int = 0x5EED, rate: float = 4.0):
    """Open meadow, light steady rain, light air, no trees/emitters/biophony, 'mobile' detail (fast)."""
    from raingen.scene import Scene

    sc = Scene(name="Integration light rain", seed=seed)
    p = sc.weather.precipitation
    p.type, p.rate_mm_h, p.character = "rain", float(rate), "continuous"
    sc.weather.wind.speed_mps = 2.0
    sc.weather.storm_arc = "steady"
    sc.place.preset = "meadow"
    sc.place.shelter = "open"
    sc.place.surfaces = {"grass": 70.0, "puddle": 30.0}
    sc.place.vegetation = {"deciduous": 0, "conifer": 0, "distance_m": 30.0}
    sc.place.emitters = []
    sc.place.biophony = {"birds": "off", "insects": "off", "frogs": "off", "traffic": 0.0,
                         "region": "temperate_europe"}
    sc.render.quality = "mobile"
    return sc


def calm_scene(seed: int = 0xC0FFEE):
    """No precipitation, no wind, no thunder: a quiet World for the synthetic generators."""
    sc = light_scene(seed)
    sc.name = "Integration synthetic"
    sc.weather.precipitation.type = "none"
    sc.weather.precipitation.rate_mm_h = 0.0
    sc.weather.wind.speed_mps = 0.0
    sc.weather.thunder.enabled = False
    return sc


class ImpulseGen:
    """Unit-ish impulses at given scene samples (frontal point source) with one marker each."""

    def __init__(self, at_samples, amp: float = 0.25, stem: str = "rain_near", azimuth: float = 0.0,
                 kinds=("thunder",)):
        self.name = "test.impulse"
        self.at = [int(a) for a in at_samples]
        self.kinds = tuple(kinds)
        self.amp = float(amp)
        self.stem = stem
        self.azimuth = float(azimuth)

    def render(self, ctx: BlockContext) -> list:
        s = np.zeros(ctx.n)
        start = int(round(ctx.t0 * ctx.sr))
        for i, a in enumerate(self.at):
            if start <= a < start + ctx.n:
                s[a - start] = self.amp
                ctx.markers.append(EventMarker(t=a / ctx.sr, kind=self.kinds[i % len(self.kinds)],
                                               data={"distance_km": 0.5, "caption": f"[Test impulse {i}]"}))
        return [Emission(s, azimuth=self.azimuth, elevation=0.0, spread=0.0, stem=self.stem, source_id="imp")]


class NoiseGen:
    """Stationary, rain-like test programme: counter-based Gaussian noise plus sparse clicks.

    Drawn from ctx.rng (layer per generator, Philox counter = block index), so it is
    deterministic, seed-dependent and prefix-stable, like every engine layer (MASTER_PLAN 9.4).
    `rate_gain` scales the level with the instantaneous rain rate (sqrt: incoherent impacts).
    """

    def __init__(self, rms: float = 0.02, stem: str = "rain_far", layer: str = "test.noise", clicks_per_s: float = 40.0,
                 click_amp: float = 0.4, spread: float = 1.0, azimuth: float = 0.4, rate_gain: bool = False):
        self.name = layer
        self.rms, self.stem, self.layer = float(rms), stem, layer
        self.clicks_per_s, self.click_amp = float(clicks_per_s), float(click_amp)
        self.spread, self.azimuth, self.rate_gain = float(spread), float(azimuth), bool(rate_gain)
        self.rates: list[float] = []
        self._g_prev = None

    def render(self, ctx: BlockContext) -> list:
        rng = ctx.rng.get(self.layer, ctx.index)
        x = rng.standard_normal(ctx.n) * self.rms
        k = int(rng.poisson(self.clicks_per_s * ctx.n / ctx.sr))
        if k:
            x[rng.integers(0, ctx.n, k)] += self.click_amp * rng.choice([-1.0, 1.0], k)
        if self.rate_gain:
            r = float(ctx.weather.rain_rate_mm_h)
            self.rates.append(r)
            g = math.sqrt(max(r, 0.0) / 4.0)
            g0 = g if self._g_prev is None else self._g_prev
            x *= g0 + (g - g0) * np.arange(1, ctx.n + 1) / ctx.n         # per-sample level ramp
            self._g_prev = g
        return [Emission(x, azimuth=self.azimuth, elevation=0.2, spread=self.spread, stem=self.stem,
                         source_id=self.layer)]


class SineGen:
    """A steady sine on an absolute sample clock (phase continuous across blocks and pipelines)."""

    def __init__(self, freq: float = 997.0, amp: float = 0.1, stem: str = "rain_near", phase: float = 0.0):
        self.name = "test.sine"
        self.freq, self.amp, self.stem, self.phase = float(freq), float(amp), stem, float(phase)

    def render(self, ctx: BlockContext) -> list:
        start = int(round(ctx.t0 * ctx.sr))
        t = (start + np.arange(ctx.n)) / ctx.sr
        return [Emission(self.amp * np.sin(2.0 * math.pi * self.freq * t + self.phase), azimuth=0.0,
                         stem=self.stem, source_id="sine")]


@pytest.fixture
def synth(monkeypatch):
    """synth(factory): every BlockRenderer built afterwards runs factory() instead of the physics generators."""
    import raingen.synth.registry as registry

    def install(factory):
        monkeypatch.setattr(registry, "make_generators",
                            lambda scene, patches, rngf, precip_types=None, **kw: list(factory()))
    return install


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------
def lufs(x, layout: str = "stereo", sr: int = SAMPLE_RATE) -> float:
    """BS.1770-5 integrated loudness; under 400 ms the meter's provisional (ungated) reading."""
    from raingen.master.loudness import LoudnessMeter

    x = np.atleast_2d(x)
    m = LoudnessMeter(sr, x.shape[0], layout=layout, true_peak=False)
    m.process(x)
    return m.integrated()


def dbtp(x, sr: int = SAMPLE_RATE) -> float:
    from raingen.master.loudness import true_peak_db
    return true_peak_db(np.atleast_2d(x), sr)


def centroid_hz(x, sr: int = SAMPLE_RATE) -> float:
    x = np.atleast_2d(x).sum(axis=0)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    return float(np.sum(f * spec) / max(np.sum(spec), 1e-30))


def jump_ratios(y: np.ndarray, positions, half: int = 96) -> np.ndarray:
    """|x[p] - x[p-1]| at each position over the rms of the first difference around it (max over channels).

    A click (step) at p gives a large ratio; for noise-like audio the ratio is ~|N(0, 1)|.
    """
    y = np.atleast_2d(y)
    d = np.diff(y, axis=1)                                   # d[:, p-1] = y[:, p] - y[:, p-1]
    out = []
    for p in positions:
        p = int(p)
        if p - half - 1 < 0 or p + half >= d.shape[1]:
            continue
        ctx = np.concatenate([d[:, p - 1 - half: p - 3], d[:, p + 2: p + half]], axis=1)
        ref = np.sqrt(np.mean(ctx ** 2, axis=1)) + 1e-12
        out.append(float(np.max(np.abs(d[:, p - 1]) / ref)))
    return np.asarray(out)


def assert_no_block_clicks(y: np.ndarray, offsets=(0,), period: int = BLOCK) -> None:
    """Jumps at block boundaries (k * period + offset) must look like jumps anywhere else."""
    n = np.atleast_2d(y).shape[1]
    for off in offsets:
        bpos = [k * period + off for k in range(1, n // period) if 0 < k * period + off < n]
        ctrl = [k * period + off + period // 2 + j for k in range(n // period) for j in (-200, -77, 0, 51, 173)]
        rb, rc = jump_ratios(y, bpos), jump_ratios(y, ctrl)
        assert rb.size >= 4 and rc.size >= 20
        # a systematic discontinuity at every boundary raises the median (noise: |N(0,1)| median ~0.67)
        assert np.median(rb) <= 1.5 * np.median(rc) + 0.25, (off, np.median(rb), np.median(rc))
        # a single click: far outside what the same audio shows anywhere else
        assert np.max(rb) <= max(2.0 * np.max(rc), 8.0), (off, np.max(rb), np.max(rc))


def assert_clean(y: np.ndarray) -> None:
    """Finite, no DC, no denormals (MASTER_PLAN 9.5: denormal flush)."""
    assert np.all(np.isfinite(y))
    tiny = np.abs(y[(y != 0.0)])
    assert not np.any(tiny < np.finfo(np.float64).tiny)          # subnormal float64
    rms = np.sqrt(np.mean(y ** 2, axis=1)) + 1e-15
    dc = np.abs(np.mean(y, axis=1))
    # DC below 10 % of the rms (a 0.25 s window of rain-like noise has |mean|/rms ~ 1/sqrt(N) << 0.1)
    assert np.all(dc <= 0.1 * rms + 1e-9), dc / rms


# Renders of real scenes are slow; share them between tests (module scope, keyed by arguments).
_CACHE: dict = {}


def cached_render_array(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


# ===========================================================================
# core: constants, RNG, types
# ===========================================================================
class TestCore:
    def test_physical_constants(self):
        from raingen.core import constants as c

        # ISO 9613-1 / Cramer 1993: c = 331.3 sqrt(1 + T/273.15) -> 343.2 m/s at 20 C
        assert c.speed_of_sound(20.0) == pytest.approx(343.21, abs=0.05)
        assert c.speed_of_sound(0.0) == pytest.approx(331.3, abs=1e-9)
        # CRC Handbook: water 998.2 kg/m3 and surface tension 72.75 mN/m at 20 C; air 1.204 kg/m3 (ISO 2533)
        assert c.RHO_WATER == pytest.approx(998.2, abs=0.5)
        assert c.SIGMA_WATER == pytest.approx(0.07275, abs=2e-4)
        assert c.RHO_AIR == pytest.approx(1.204, abs=0.002)
        assert c.P0 == 101_325.0                           # ISO 2533 standard atmosphere
        assert c.G == pytest.approx(9.80665, abs=0.005)    # CGPM 1901 standard gravity
        assert c.GAMMA_AIR == pytest.approx(1.4, abs=0.002)
        # contract: 48 kHz / 1024-frame blocks -> 46.875 Hz control rate; 3rd order -> 16 SH channels
        assert c.CONTROL_RATE == pytest.approx(46.875)
        assert c.N_AMBI == (c.AMBI_ORDER + 1) ** 2 == 16
        assert len(set(c.STEMS)) == len(c.STEMS) == 7

    def test_level_calibration(self):
        from raingen.core import constants as c

        # 20 log10(1 Pa / 20 uPa) = 93.98 dB SPL (IEC 60942 calibrator: 94 dB SPL ~ 1 Pa rms)
        assert c.FULL_SCALE_DB_SPL == pytest.approx(93.979, abs=1e-3)
        assert c.db_spl_to_pa(c.FULL_SCALE_DB_SPL) == pytest.approx(c.PA_PER_FULL_SCALE, rel=1e-12)
        assert c.db_spl_to_pa(94.0) == pytest.approx(1.0024, abs=1e-4)
        assert c.pa_to_db_spl(2e-5) == pytest.approx(0.0, abs=1e-9)
        assert c.pa_to_db_spl(0.0) == float("-inf")
        for db in (0.0, 35.0, 60.0, 94.0, 120.0):
            assert c.pa_to_db_spl(c.db_spl_to_pa(db)) == pytest.approx(db, abs=1e-9)

    def test_level_calibration_in_lufs(self):
        """A 997 Hz tone of L dB SPL in one channel reads L - 93.98 LUFS: BS.1770-5 calibrates
        -0.691 so a 997 Hz full-scale sine (rms -3.01 dBFS) reads -3.01 LUFS, and 1.0 = 1 Pa."""
        from raingen.core import constants as c

        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        for L in (50.0, 70.0):
            x = c.db_spl_to_pa(L) * math.sqrt(2.0) * np.sin(2 * math.pi * 997.0 * t)
            # BS.1770-5 Table 1 / EBU Tech 3341 test 1: +-0.1 LU
            assert lufs(x[None, :], layout="mono") == pytest.approx(L - c.FULL_SCALE_DB_SPL, abs=0.1)

    def test_rng_same_key_same_stream(self):
        from raingen.core.rng import RngFactory

        a, b = RngFactory(1234), RngFactory(1234)
        x = a.get("rain.drops", 7).random(1000)
        assert np.array_equal(x, b.get("rain.drops", 7).random(1000))
        # get() hands out a fresh generator: asking twice gives the same numbers (random access)
        assert np.array_equal(x, a.get("rain.drops", 7).random(1000))

    def test_rng_layers_independent_of_creation_order(self):
        from raingen.core.rng import RngFactory

        a = RngFactory(99)
        a.get("wind", 0).random(10)
        a.get("thunder", 3).random(10)
        ya = a.get("rain", 5).standard_normal(500)
        b = RngFactory(99)                                    # never saw 'wind' or 'thunder'
        assert np.array_equal(ya, b.get("rain", 5).standard_normal(500))

    def test_rng_streams_uncorrelated(self):
        from scipy import stats

        from raingen.core.rng import RngFactory

        f = RngFactory(0x5EED)
        n = 20000
        streams = {
            "layer": (f.get("rain", 0).standard_normal(n), f.get("wind", 0).standard_normal(n)),
            "block": (f.get("rain", 0).standard_normal(n), f.get("rain", 1).standard_normal(n)),
            "seed": (f.get("rain", 0).standard_normal(n), RngFactory(0x5EEE).get("rain", 0).standard_normal(n)),
            "far block": (f.get("rain", 3).standard_normal(n), f.get("rain", 2 ** 40 + 3).standard_normal(n)),
        }
        for name, (x, y) in streams.items():
            r = float(np.corrcoef(x, y)[0, 1])
            # |r| of independent N(0,1) samples: sd 1/sqrt(n) = 0.007; 5 sd bound
            assert abs(r) < 5.0 / math.sqrt(n), (name, r)
        u = f.get("rain", 42).random(n)
        # Kolmogorov-Smirnov against U(0,1): reject only at p < 1e-4
        assert stats.kstest(u, "uniform").pvalue > 1e-4

    def test_rng_child_and_large_seed(self):
        from raingen.core.rng import RngFactory, new_seed

        f = RngFactory(7)
        c1, c2 = f.child("variation/0"), f.child("variation/0")
        assert c1.root_seed == c2.root_seed
        assert c1.root_seed != f.child("variation/1").root_seed
        assert np.array_equal(c1.get("x").random(8), c2.get("x").random(8))
        big = RngFactory(2 ** 127 + 5)                         # 128-bit seeds are valid keys
        assert big.get("x", 2 ** 63).random(4).shape == (4,)
        assert isinstance(new_seed(), int) and 0 <= new_seed() < 2 ** 64

    @pytest.mark.parametrize("bearing, heading, expected_deg", [
        (0.0, 0.0, 0.0),         # straight ahead
        (90.0, 0.0, -90.0),      # east while facing north: on the right (azimuth CCW, +y left)
        (270.0, 0.0, 90.0),      # west: on the left
        (180.0, 0.0, 180.0),     # behind (wrapped to +pi)
        (0.0, 180.0, 180.0),
        (10.0, 350.0, -20.0),    # across north
        (350.0, 10.0, 20.0),
        (720.0 + 45.0, 0.0, -45.0),
    ])
    def test_bearing_to_azimuth(self, bearing, heading, expected_deg):
        from raingen.core.types import bearing_to_azimuth

        az = bearing_to_azimuth(bearing, heading)
        assert -math.pi < az <= math.pi
        assert math.degrees(az) == pytest.approx(expected_deg, abs=1e-9)

    def test_source_patch_geometry(self):
        from raingen.core.types import SourcePatch

        p = SourcePatch("p", "surface", "grass", position=(0.0, 3.0, 4.0))
        assert p.distance == pytest.approx(5.0)
        assert p.azimuth == pytest.approx(math.pi / 2)          # +y = left
        assert p.elevation == pytest.approx(math.atan2(4.0, 3.0))
        assert SourcePatch("q", "surface", "grass", position=(0.0, 0.0, 0.0)).distance == 0.05


# ===========================================================================
# package and API surface
# ===========================================================================
class TestApiSurface:
    def test_import_raingen_then_attribute_api(self):
        """`import raingen; raingen.api` and every lazily exported name, in a fresh interpreter."""
        code = (
            "import raingen, sys\n"
            "assert 'numpy' not in sys.modules, 'import raingen must stay cheap'\n"
            "import types\n"
            "assert isinstance(raingen.api, types.ModuleType)\n"
            "missing = [n for n in raingen.__all__ if not hasattr(raingen, n)]\n"
            "assert not missing, missing\n"
            "for m in raingen._SUBMODULES:\n"
            "    assert isinstance(getattr(raingen, m), types.ModuleType), m\n"
            "assert 'api' in dir(raingen) and 'render_array' in dir(raingen)\n"
            "assert raingen.render_array is raingen.api.render_array\n"
            "assert raingen.render is raingen.api.render\n"
            "print('ok')\n"
        )
        env = dict(os.environ, OMP_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120,
                           env=env)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "ok"

    def test_render_callable_in_any_import_order(self):
        """`from raingen.render import ...` first binds the submodule as raingen.render; it must stay callable."""
        code = (
            "from raingen.render import Renderer\n"
            "import raingen\n"
            "assert callable(raingen.render)\n"
            "try:\n"
            "    raingen.render('light rain', seconds=-1)\n"
            "except ValueError as e:\n"
            "    assert 'positive' in str(e), e\n"
            "else:\n"
            "    raise SystemExit('no error')\n"
            "print('ok')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))
        assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr

    def test_unknown_attribute_raises(self):
        import raingen

        with pytest.raises(AttributeError):
            raingen.no_such_thing  # noqa: B018

    def test_duration_seconds(self):
        from raingen.api import duration_seconds

        assert duration_seconds() == 60.0
        assert duration_seconds(seconds=30, minutes=1) == 90.0
        assert duration_seconds(hours=0.5) == 1800.0
        with pytest.raises(ValueError):
            duration_seconds(seconds=-1)


# ===========================================================================
# master chain contracts (synthetic programme, fast)
# ===========================================================================
class TestLatencyAndMarkers:
    @pytest.mark.parametrize("layout", ["stereo", "mono", "5.1", "foa", "binaural"])
    def test_sample_zero_is_scene_time_zero(self, synth, layout):
        """An impulse at scene sample T lands on output sample T; its marker says T / sr (render.py contract)."""
        from raingen.render import render_array

        T = [3 * BLOCK + 357, 9 * BLOCK + 11]
        synth(lambda: [ImpulseGen(T)])
        y, markers = render_array(calm_scene(), 0.25, layout=layout, destination="raw", bit_depth=32)
        assert y.shape == (LAYOUT_CHANNELS[layout], int(round(0.25 * SAMPLE_RATE)))
        env = np.max(np.abs(y), axis=0)
        for t in T:
            seg = env[t - 400: t + 400]
            onset = t - 400 + int(np.argmax(seg > 0.1 * seg.max()))
            if layout == "binaural":
                # HRIR onset: the ITD of a frontal source is 0, the HRIR rise is < 1 ms (Algazi et al. 2001)
                assert abs(onset - t) <= 48, (layout, onset, t)
            else:
                # matrix decoders are memoryless: exact
                assert onset == t, (layout, onset, t)
        assert [round(m.t * SAMPLE_RATE) for m in markers] == T

    def test_resampled_output_alignment(self, synth):
        """At 44.1 kHz the impulse lands at round(T * 44100 / 48000) (resampler has no added delay)."""
        from raingen.render import render_array

        T = 6 * BLOCK + 500
        synth(lambda: [ImpulseGen([T])])
        y, _ = render_array(calm_scene(), 0.25, layout="mono", destination="raw", bit_depth=32, sample_rate=44100)
        assert y.shape[1] == int(round(0.25 * 44100))
        k = int(np.argmax(np.abs(y[0])))
        # polyphase sinc interpolation: peak within one output sample of the exact position
        assert abs(k - T * 44100 / 48000) <= 1.0

    def test_captions_and_cue_sheet_times(self, synth, tmp_out):
        from raingen.render import render

        # captions.py SDH rules: same-kind events within 10 s merge, cues show >= 1.5 s and overlapping
        # one-liners stack, so two separate cues need different kinds >= 1.5 s apart
        T = [4 * BLOCK + 100, 2 * SAMPLE_RATE + 700]
        synth(lambda: [ImpulseGen(T, kinds=("thunder", "gust"))])
        rep = render(calm_scene(), 2.2, out=str(tmp_out / "cap.wav"), destination="raw",
                     captions=("srt", "vtt", "csv", "edl"))
        assert set(rep["captions"]) == {"srt", "vtt", "csv", "edl"}
        assert rep["markers"] == 2 and rep["marker_kinds"] == {"thunder": 1, "gust": 1}
        with open(rep["captions"]["csv"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        # csv carries the event time to 0.1 ms (4 decimals)
        assert [float(r["time_s"]) for r in rows] == pytest.approx([t / SAMPLE_RATE for t in T], abs=1e-4)
        srt = open(rep["captions"]["srt"]).read()
        starts = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt)
        secs = [int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000 for h, m, s, ms in starts]
        # SRT has millisecond resolution
        assert secs == pytest.approx([t / SAMPLE_RATE for t in T], abs=1e-3)
        assert open(rep["captions"]["vtt"]).read().startswith("WEBVTT")
        assert "TITLE:" in open(rep["captions"]["edl"]).read()


class TestStemsAndFiles:
    def _three_stems(self):
        return [NoiseGen(0.02, "rain_near", "test.near", spread=0.0, azimuth=0.5),
                NoiseGen(0.01, "wind", "test.wind", clicks_per_s=0, spread=1.0),
                ImpulseGen([5 * BLOCK + 3], amp=0.2, stem="thunder")]

    def test_float_stems_sum_to_mix(self, synth, tmp_out):
        """raw / 32-bit float: no limiter, no dither -> stems sum to the mix to float32 precision."""
        import soundfile as sf

        from raingen.io.ucs import UCS_CATIDS
        from raingen.render import render

        synth(self._three_stems)
        rep = render(calm_scene(), 0.5, out=str(tmp_out) + os.sep, destination="raw", bit_depth=32, stems=True)
        mix, sr = sf.read(rep["path"], always_2d=True)
        assert sr == SAMPLE_RATE and mix.shape == (int(0.5 * SAMPLE_RATE), 2)
        total = np.zeros_like(mix)
        from raingen.core.constants import STEMS
        assert set(rep["stems"]) == set(STEMS)
        for stem, path in rep["stems"].items():
            name = os.path.basename(path)
            # UCS v8.2.1: CatID_FXName_CreatorID_SourceID(_UserData).ext
            m = re.fullmatch(r"([A-Za-z]+)_([^_]+)_RGN_raingen_([a-z_\-]+)\.wav", name)
            assert m, name
            assert m.group(1) in UCS_CATIDS, name
            assert m.group(3).replace("-", "_") == stem
            x, _ = sf.read(path, always_2d=True)
            total += x
        assert re.fullmatch(r"[A-Za-z]+_[^_]+_RGN_raingen\.wav", os.path.basename(rep["path"]))
        cat = {stem: os.path.basename(p).split("_", 1)[0] for stem, p in rep["stems"].items()}
        assert "THUNDER" in UCS_CATIDS[cat["thunder"]]       # UCS: THUNDER category (or WEATHER/THUNDER)
        assert UCS_CATIDS[cat["wind"]][0] == "WIND"
        err = np.max(np.abs(total - mix))
        # float32 rounding of 8 files: < 8 ulp at the 0.5-ish peak level
        assert err < 8 * np.finfo(np.float32).eps * max(1.0, np.max(np.abs(mix)))

    def test_pcm24_stems_sum_within_dither(self, synth, tmp_out):
        """raw / 24-bit: each file gets independent +-1 LSB TPDF dither (Lipshitz et al. 1992)."""
        import soundfile as sf

        from raingen.render import render

        synth(self._three_stems)
        rep = render(calm_scene(), 0.5, out=str(tmp_out / "mix24.wav"), destination="raw", stems=True)
        mix, _ = sf.read(rep["path"], always_2d=True)
        total = sum(sf.read(p, always_2d=True)[0] for p in rep["stems"].values())
        err = total - mix
        # 8 independent TPDF-dithered requantisations: error rms = sqrt(8 * LSB^2 / 4) ~ 1.4 LSB
        assert np.sqrt(np.mean(err ** 2)) < 3.0 * LSB24
        assert np.max(np.abs(err)) < 16 * LSB24
        info = sf.info(rep["path"])
        assert info.subtype == "PCM_24" and info.frames == int(0.5 * SAMPLE_RATE)

    def test_flac_and_metadata(self, synth, tmp_out):
        import soundfile as sf

        from raingen.render import render

        synth(lambda: [NoiseGen()])
        rep = render(calm_scene(seed=0xABC), 0.3, out=str(tmp_out / "x.flac"), destination="raw")
        info = sf.info(rep["path"])
        assert rep["path"].endswith(".flac") and info.format == "FLAC" and info.subtype == "PCM_24"
        assert info.frames == int(round(0.3 * SAMPLE_RATE))
        # 7.1.4 cannot go to FLAC (8 channels max, libFLAC): RF64/WAV instead, with a warning
        rep2 = render(calm_scene(), 0.25, out=str(tmp_out / "y.flac"), layout="7.1.4", destination="raw")
        assert rep2["path"].endswith(".wav") and any("FLAC" in w for w in rep2["warnings"])
        assert sf.info(rep2["path"]).channels == 12

    @pytest.mark.parametrize("layout", list(LAYOUT_CHANNELS))
    def test_every_layout_to_file(self, synth, tmp_out, layout):
        import soundfile as sf

        from raingen.io.wav import layout_channel_mask
        from raingen.render import render

        synth(lambda: [NoiseGen(0.01, clicks_per_s=0.0), ImpulseGen([4000], amp=0.2)])
        rep = render(calm_scene(), 0.25, out=str(tmp_out / f"{layout}.wav"), layout=layout, destination="raw")
        info = sf.info(rep["path"])
        assert (info.channels, info.samplerate, info.frames) == (LAYOUT_CHANNELS[layout], 48000, 12000)
        assert info.subtype == "PCM_24" and rep["layout"] == layout
        x, _ = sf.read(rep["path"], always_2d=True)
        assert_clean(x.T)
        assert np.max(np.abs(x)) > 0.01
        raw = open(rep["path"], "rb").read(4096)
        mask = layout_channel_mask(layout, info.channels)
        if info.channels > 2 and mask:
            # WAVE_FORMAT_EXTENSIBLE dwChannelMask (Microsoft KSAUDIO speaker positions)
            assert mask.to_bytes(4, "little") in raw

    def test_wav_embeds_seed_and_scene(self, synth, tmp_out):
        from raingen.render import render

        synth(lambda: [NoiseGen()])
        rep = render(calm_scene(seed=0xBEEF), 0.25, out=str(tmp_out / "meta.wav"), destination="raw")
        raw = open(rep["path"], "rb").read()
        assert b"bext" in raw and b"iXML" in raw
        assert b"seed=0xBEEF" in raw                        # bext CodingHistory (render.py contract)
        assert b"Integration synthetic" in raw              # the scene travels with the file


class TestLoudness:
    from raingen.master.destinations import DESTINATIONS as _D
    NORMALISED = [k for k, v in _D.items() if v["normalize"]]

    @pytest.mark.parametrize("destination", NORMALISED)
    def test_normalised_destinations_hit_target(self, synth, destination):
        """3 s stationary programme: integrated loudness within +-0.5 LU, true peak <= ceiling.

        Targets and ceilings: raingen.master.destinations (EBU R128 -23 LUFS / -1 dBTP, ATSC A/85
        -24 LKFS / -2 dBTP, streaming -14 / -16 LUFS ...). The clicks put the peak-to-loudness
        ratio near 18 dB (steady rain measured 15 dB: -43.5 LUFS, -28.2 dBTP), so loud targets
        make the limiter work.
        """
        from raingen.master.destinations import get_destination
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.02, clicks_per_s=20.0, click_amp=0.2)])
        d = get_destination(destination)
        r = Renderer(calm_scene(), 3.0, layout="stereo", destination=destination)
        y, _ = r.to_array()
        rep = r.summary
        assert r.pass2 == "replay"
        assert rep["integrated_lufs"] == pytest.approx(d["target_lufs"], abs=0.5)
        # independent re-measurement of the delivered array (BS.1770-5 meter)
        assert lufs(y) == pytest.approx(d["target_lufs"], abs=0.5)
        # 8x oversampled true peak may under-read by <= 0.17 dB (MASTER_PLAN 10)
        assert dbtp(y) <= d["tp_dbtp"] + 0.17
        assert rep["true_peak_dbtp"] <= d["tp_dbtp"] + 0.17
        assert "programme is silent" not in " ".join(rep["warnings"])

    def test_limiter_makeup_reaches_target(self, synth):
        """Peaky programme at -14 LUFS: the static gain alone would land low; calibration fixes it."""
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.02, clicks_per_s=10.0, click_amp=0.3)])    # peak-to-loudness ~ 21 dB
        r = Renderer(calm_scene(), 3.0, layout="stereo", destination="youtube")
        y, _ = r.to_array()
        linear = -14.0 - r._measured
        assert r.gain_db > linear + 1.0                       # make-up was needed and applied
        assert r.gain_db <= linear + 6.0 + 1e-9               # CALIBRATION_MAX_MAKEUP_DB
        assert lufs(y) == pytest.approx(-14.0, abs=0.5)
        assert dbtp(y) <= -1.0 + 0.17

    def test_limiter_makeup_is_bounded(self, synth):
        """A programme the limiter cannot bring to -14 LUFS without crushing it gets at most 6 dB of
        make-up, a warning, and still respects the true-peak ceiling."""
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.01, clicks_per_s=30.0, click_amp=0.2)])     # clicks every 33 ms > release
        r = Renderer(calm_scene(), 3.0, layout="stereo", destination="youtube")
        y, _ = r.to_array()
        assert r.gain_db == pytest.approx(-14.0 - r._measured + 6.0, abs=1e-9)
        assert any("not reached" in w for w in r.summary["warnings"])
        assert dbtp(y) <= -1.0 + 0.17

    def test_tape_replay_matches_resynthesis(self, synth):
        """Pass 2 from the tape equals pass 2 by re-synthesis to float32 precision (no limiting)."""
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.01, clicks_per_s=0.0)])
        kw = dict(layout="stereo", destination="broadcast_eu", bit_depth=32)
        a = Renderer(calm_scene(), 1.0, **kw)
        ya, _ = a.to_array()
        b = Renderer(calm_scene(), 1.0, tape=False, **kw)
        yb, _ = b.to_array()
        assert (a.pass2, b.pass2) == ("replay", "render")
        assert a.gain_db == pytest.approx(b.gain_db, abs=1e-9)
        # float32 tape: relative error ~ 2^-24 (-144 dB)
        assert np.max(np.abs(ya - yb)) <= 1e-6 * np.max(np.abs(yb))

    def test_memmap_tape_equals_memory_tape(self, synth, tmp_out, monkeypatch):
        import raingen.render as rr

        synth(lambda: [NoiseGen(0.01)])
        ya, _ = rr.Renderer(calm_scene(), 0.5, destination="youtube").to_array()
        monkeypatch.setattr(rr, "TAPE_MEMORY_BYTES", 0)
        r = rr.Renderer(calm_scene(), 0.5, destination="youtube", tmp_dir=str(tmp_out))
        yb, _ = r.to_array()
        assert r.pass2 == "replay"
        assert np.array_equal(ya, yb)
        assert not [p for p in os.listdir(tmp_out) if p.startswith(".raingen-tape-")]   # cleaned up

    @pytest.mark.parametrize("seconds", [0.1, 0.25, 0.39])
    def test_short_clips_are_normalised(self, synth, seconds):
        """Clips under one 400 ms gating block still reach the target (short-programme measurement)."""
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.01, clicks_per_s=0.0)])
        r = Renderer(calm_scene(), seconds, layout="stereo", destination="broadcast_eu")
        y, _ = r.to_array()
        assert math.isfinite(r._measured)
        assert "programme is silent" not in " ".join(r.warnings)
        # whole-clip K-weighted mean square, same definition as the measurement: +-0.5 LU
        assert lufs(y) == pytest.approx(-23.0, abs=0.5)

    def test_silence_is_reported_not_amplified(self, synth):
        from raingen.render import Renderer

        synth(lambda: [])
        r = Renderer(calm_scene(), 0.5, destination="youtube")
        y, _ = r.to_array()
        assert r.gain_db == 0.0 and not np.any(y)
        assert any("silent" in w for w in r.warnings)

    def test_gain_sits_before_limiter(self, synth):
        """A normalised render is the raw render times one static gain while the limiter is idle."""
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.005, clicks_per_s=0.0)])
        raw, _ = Renderer(calm_scene(), 1.0, destination="raw", bit_depth=32).to_array()
        r = Renderer(calm_scene(), 1.0, destination="broadcast_eu", bit_depth=32)
        y, _ = r.to_array()
        g = 10.0 ** (r.gain_db / 20.0)
        # -23 LUFS on a crest factor ~ 12 dB noise: far below -1 dBTP, limiter idle -> exact up to float32 tape
        assert np.max(np.abs(y - g * raw)) <= 1e-6 * np.max(np.abs(y))

    def test_progress_and_cancel(self, synth):
        from raingen.render import RenderCancelled, Renderer

        synth(lambda: [NoiseGen()])
        seen = []
        Renderer(calm_scene(), 0.5, destination="youtube", progress=lambda f, ph: seen.append((f, ph))).to_array()
        fr = [f for f, _ in seen]
        assert fr == sorted(fr) and fr[-1] == pytest.approx(1.0)        # monotone across both passes
        assert {ph for _, ph in seen} >= {"measure", "render"}
        with pytest.raises(RenderCancelled):
            Renderer(calm_scene(), 0.5, destination="youtube", progress=lambda f, ph: False).to_array()

    @pytest.mark.parametrize("sr_out", [44100, 48000, 96000])
    def test_output_chain_true_peak_at_output_rate(self, sr_out):
        """The limiter runs after the resampler: hail-like transients (sharp, broadband) driven into
        a -1 dBTP ceiling stay under it at the delivered rate. With the limiter at 48 kHz ahead of
        the 44.1 kHz resampler, the band-limiting rang them up to -0.82 dBTP."""
        from raingen.master.loudness import true_peak_db
        from raingen.render import _OutputChain

        rng = np.random.default_rng(1)
        n = 2 * SAMPLE_RATE
        x = 0.05 * rng.standard_normal((2, n))
        for t in rng.integers(1000, n - 1000, 300):
            x[:, t:t + 3] += rng.choice([-1.0, 1.0]) * np.array([4.0, -3.0, 1.5])
        frames = int(n * sr_out / SAMPLE_RATE) - 2000
        out: list = []
        ch = _OutputChain("mix", "mix", 2, frames, sr_in=SAMPLE_RATE, sr_out=sr_out, gain_db=6.0,
                          ceiling_dbtp=-1.0, true_peak=False, sink=out.append)
        for b in range(0, n, BLOCK):
            ch.push(x[:, b:b + BLOCK])
        y = np.concatenate(out, axis=1)
        assert ch.done and y.shape[1] == frames
        assert ch.limiter.max_gain_reduction_db > 6.0                  # the limiter is working hard
        assert true_peak_db(y, sr_out) <= -1.0 + 1e-3

    def test_output_chain_alignment_with_limiter(self):
        """Limiter look-ahead is trimmed at the output rate: an impulse at input sample T lands at
        T * 44100 / 48000 in the delivered signal, with or without a limiter."""
        from raingen.render import _OutputChain

        T = 3 * BLOCK + 517
        x = np.zeros((1, SAMPLE_RATE // 2))
        x[0, T] = 0.1
        peaks = []
        for ceiling in (None, -1.0):
            out: list = []
            ch = _OutputChain("mix", "mix", 1, 20000, sr_in=SAMPLE_RATE, sr_out=44100, skip=0,
                              ceiling_dbtp=ceiling, layout="mono", meter=False, sink=out.append)
            for b in range(0, x.shape[1], BLOCK):
                ch.push(x[:, b:b + BLOCK])
            y = np.concatenate(out, axis=1)
            assert y.shape[1] == 20000
            peaks.append(int(np.argmax(np.abs(y[0]))))
        assert peaks[0] == peaks[1]
        assert abs(peaks[0] - T * 44100 / SAMPLE_RATE) <= 1.0

    @pytest.mark.slow                  # 2 s of real synthesis, two passes + calibration replays
    def test_real_hail_44k1_true_peak(self, tmp_out):
        """hailstorm-car (dense, peaky hail transients) to youtube at 44.1 kHz: the written file's
        true peak (BS.1770 8x detector) is at or below the -1 dBTP ceiling."""
        import soundfile as sf

        from raingen.master.loudness import true_peak_db
        from raingen.render import render
        from raingen.scene.presets import preset

        sc = preset("hailstorm-car")
        sc.render.quality = "mobile"
        rep = render(sc, 2.0, out=str(tmp_out / "hail.wav"), destination="youtube", sample_rate=44100)
        y, sr = sf.read(rep["path"], always_2d=True)
        assert sr == 44100 and y.shape[0] == 2 * 44100
        assert true_peak_db(y.T, sr) <= -1.0 + 0.1
        assert_clean(y.T)


class TestRenderOptions:
    @pytest.mark.parametrize("block", [256, 480, 1000])
    def test_any_block_length(self, synth, block):
        """The renderer (tape, latency trim, limiter, resampler) works for any block length: a
        deterministic sine gives the same output as with BLOCK = 1024 (float rounding only)."""
        from raingen.render import Renderer

        synth(lambda: [SineGen(440.0, 0.05), ImpulseGen([7777], amp=0.3)])
        kw = dict(layout="binaural", destination="youtube", bit_depth=32)
        ref, mref = Renderer(calm_scene(), 0.5, **kw).to_array()
        y, m = Renderer(calm_scene(), 0.5, block=block, **kw).to_array()
        assert y.shape == ref.shape
        # same programme, same measured gain; binaural FFT convolution differs by rounding (< -200 dB)
        assert np.max(np.abs(y - ref)) <= 1e-9 * np.max(np.abs(ref))
        assert [round(x.t * SAMPLE_RATE) for x in m] == [round(x.t * SAMPLE_RATE) for x in mref] == [7777]

    def test_mix_mutes_and_scales_stems(self, synth):
        from raingen.render import Renderer, stem_gains

        synth(lambda: [NoiseGen(0.01, "rain_near", "test.near", clicks_per_s=0.0),
                       NoiseGen(0.01, "thunder", "test.thunder", clicks_per_s=0.0)])
        kw = dict(layout="stereo", destination="raw", bit_depth=32)
        full, _ = Renderer(calm_scene(), 0.5, **kw).to_array()
        no_th, _ = Renderer(calm_scene(), 0.5, mix={"thunder": "off"}, **kw).to_array()
        quiet, _ = Renderer(calm_scene(), 0.5, mix={"thunder": "off", "rain": "-6 dB"}, **kw).to_array()
        # two equal, independent sources: muting one costs 10 log10 2 = 3.0 dB; -6 dB on the other 6.0 dB more
        assert lufs(full) - lufs(no_th) == pytest.approx(3.0, abs=0.5)
        assert lufs(no_th) - lufs(quiet) == pytest.approx(6.0, abs=0.05)
        assert stem_gains({"drips": -20})["aftermath"] == pytest.approx(0.1)
        with pytest.raises(ValueError):
            stem_gains({"kazoo": 0})

    def test_44k1_file(self, synth, tmp_out):
        import soundfile as sf

        from raingen.render import render

        synth(lambda: [NoiseGen(0.01, clicks_per_s=0.0)])
        rep = render(calm_scene(), 0.5, out=str(tmp_out / "x.wav"), destination="podcast", sample_rate=44100)
        info = sf.info(rep["path"])
        assert info.samplerate == 44100 and info.frames == 22050
        x, _ = sf.read(rep["path"], always_2d=True)
        assert lufs(x.T, sr=44100) == pytest.approx(-16.0, abs=0.5)        # Apple Podcasts -16 LKFS +-1

    def test_api_text_and_scene_methods(self, synth, tmp_out):
        import raingen as rg

        synth(lambda: [NoiseGen(0.01)])
        rep = rg.render("light rain on grass, wind 10 km/h", seconds=0.25, out=str(tmp_out / "t.wav"))
        assert rep["understood"] and os.path.isfile(rep["path"]) and rep["seconds"] == 0.25
        assert "Loudness" in str(rep)
        sc = calm_scene()
        rep2 = sc.render(seconds=0.25, out=str(tmp_out / "m.wav"))            # RenderSettings is callable
        assert rep2["frames"] == 12000
        assert sc.render.quality == "mobile"                                  # ... and still the settings
        y, _ = rg.render_array(sc, 0.25, destination="raw")
        assert y.shape == (2, 12000)
        with pytest.raises(rg.SceneParseError):
            rg.render("qwzx vbnm", seconds=0.25, out=str(tmp_out / "e.wav"), strict=True)

    def test_variations_are_deterministic(self):
        import raingen as rg

        base = light_scene()
        a, b = rg.variations(base, 4), rg.variations(base, 4)
        assert [v.seed for v in a] == [v.seed for v in b]
        assert len({v.seed for v in a} | {base.seed}) == 5
        assert all(v.weather.precipitation.type == "rain" and v.place.shelter == "open" for v in a)
        rates = [v.weather.precipitation.rate_mm_h for v in a]
        # +-20 % (log domain, 1 sd) around 4 mm/h: all within 4 x 2^(+-4 sd * 0.27)
        assert all(4.0 * 2 ** -1.1 < r < 4.0 * 2 ** 1.1 for r in rates)
        assert [v.seed for v in rg.variations(base, 4, jitter=0.0)] == [v.seed for v in a]
        assert all(v.weather.precipitation.rate_mm_h == 4.0 for v in rg.variations(base, 2, jitter=0.0))


class TestThreads:
    def test_cli_sets_single_thread_env_before_numpy(self):
        code = ("import os, sys\n"
                "for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.pop(v, None)\n"
                "from raingen import cli\n"
                "assert cli.limit_blas_threads(1) is True\n"
                "assert os.environ['OPENBLAS_NUM_THREADS'] == '1'\n"
                "os.environ['OMP_NUM_THREADS'] = '3'\n"
                "cli.limit_blas_threads(1)\n"
                "assert os.environ['OMP_NUM_THREADS'] == '3'   # the user's choice wins\n"
                "import numpy\n"
                "assert cli.limit_blas_threads(1) is False\n"
                "print('ok')\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr

    def test_limit_threads_is_safe_without_threadpoolctl(self):
        from raingen.render import limit_threads

        with limit_threads(1):
            assert float(np.ones((64, 64)) @ np.ones(64) @ np.ones(64)) == 64.0 * 64.0

    def test_limit_threads_caps_blas_pools(self):
        """With threadpoolctl installed, every BLAS/OpenMP pool runs one thread inside a render and
        gets its previous size back afterwards."""
        tp = pytest.importorskip("threadpoolctl")
        from raingen.render import limit_threads

        before = [i["num_threads"] for i in tp.threadpool_info()]
        with tp.threadpool_limits(limits=2):
            with limit_threads(1):
                assert all(i["num_threads"] == 1 for i in tp.threadpool_info())
        assert [i["num_threads"] for i in tp.threadpool_info()] == before


# ===========================================================================
# real physics scenes
# ===========================================================================
def _light(seconds, **kw):
    from raingen.render import render_array

    key = ("light", seconds, tuple(sorted(kw.items())))
    return cached_render_array(key, lambda: render_array(light_scene(kw.pop("seed", 0x5EED)), seconds, **kw))


class TestRealScenes:
    """Real generators at 'mobile' detail: 0.25-0.5 s renders take 0.5-3 s; the longer ones are slow."""

    @pytest.mark.parametrize("layout", list(LAYOUT_CHANNELS))
    def test_layouts(self, layout):
        y, _ = _light(0.25, layout=layout, destination="raw", bit_depth=32)
        assert y.shape == (LAYOUT_CHANNELS[layout], int(0.25 * SAMPLE_RATE))
        assert_clean(y)
        assert np.max(np.abs(y)) > 1e-6, "rain is not silent"
        if layout in ("foa", "hoa3"):
            w2 = np.mean(y[0] ** 2)
            xyz2 = np.mean(y[1:4] ** 2, axis=1)
            # SN3D: a plane wave has X^2 + Y^2 + Z^2 = W^2; any mixture of sources has <= (Zotter & Frank 2019)
            assert np.sum(xyz2) <= 1.05 * w2
        if layout == "5.1":
            # LFE (index 3, SMPTE L R C LFE Ls Rs) carries only low frequencies or nothing (ITU-R BS.775)
            lfe = y[3]
            assert np.sqrt(np.mean(lfe ** 2)) < 0.5 * np.sqrt(np.mean(y[0] ** 2)) or centroid_hz(lfe) < 200.0

    def test_layout_levels_consistent(self):
        """The same scene measures similar loudness in every speaker layout (+-4 LU; BS.1770 channel weights)."""
        ref = lufs(_light(0.25, layout="stereo", destination="raw", bit_depth=32)[0], "stereo")
        for layout in ("binaural", "5.1", "7.1", "mono"):
            got = lufs(_light(0.25, layout=layout, destination="raw", bit_depth=32)[0], layout)
            assert abs(got - ref) <= 4.0, (layout, got, ref)

    def test_rain_spectrum_plausible(self):
        """Light rain on grass and puddles: broadband hiss; centroid 1-10 kHz (Medwin et al. 1992,
        Nystuen 2001: rain noise peaks 1-5 kHz with energy to 20 kHz)."""
        y, _ = _light(0.25, layout="stereo", destination="raw", bit_depth=32)
        c = centroid_hz(y)
        assert 1000.0 < c < 10000.0, c

    def test_same_seed_bit_identical_and_seed_matters(self):
        from raingen.render import render_array

        y1, m1 = _light(0.25, layout="stereo", destination="raw", bit_depth=32)
        y2, m2 = render_array(light_scene(), 0.25, layout="stereo", destination="raw", bit_depth=32)
        assert np.array_equal(y1, y2)
        assert [(m.t, m.kind) for m in m1] == [(m.t, m.kind) for m in m2]
        y3, _ = render_array(light_scene(seed=0x5EEE), 0.25, layout="stereo", destination="raw", bit_depth=32)
        assert y3.shape == y1.shape and not np.array_equal(y1, y3)
        r = np.corrcoef(y1[0], y3[0])[0, 1]
        # different seeds: different drops -> waveforms essentially uncorrelated
        assert abs(r) < 0.3, r
        # ... but the same weather: loudness within 3 LU (stationary rain, 0.25 s)
        assert abs(lufs(y1) - lufs(y3)) < 3.0

    @pytest.mark.slow                  # 2 s + 1 s (3 s) of real synthesis
    def test_prefix_stability_and_no_clicks(self):
        """The first second of a 2 s render equals a 1 s render (raw: no normalisation gain), and the
        final mix shows no discontinuities at block boundaries (scene grid and latency-shifted grid)."""
        from raingen.render import Renderer

        r2 = Renderer(light_scene(), 2.0, layout="stereo", destination="raw")
        y2, _ = r2.to_array()
        r1 = Renderer(light_scene(), 1.0, layout="stereo", destination="raw")
        y1, _ = r1.to_array()
        n = y1.shape[1]
        # MASTER_PLAN 9.4: same build + block size + sequential render -> bit-identical
        assert np.array_equal(y1, y2[:, :n])
        assert_clean(y2)
        from raingen.render import BlockRenderer
        lat = BlockRenderer(light_scene(), "stereo").latency
        lim = r2._limiter_latency(2)
        assert_no_block_clicks(y2, offsets=sorted({0, (-lat) % BLOCK, (-lat - lim) % BLOCK}))

    @pytest.mark.parametrize("destination", ["raw", "youtube", "broadcast_eu", "game", "vr", "post_stems"])
    def test_destinations_to_file(self, destination, tmp_out):
        import soundfile as sf

        from raingen.master.destinations import get_destination
        from raingen.master.loudness import integrated_loudness
        from raingen.render import SAFETY_CEILING_DBTP, render

        d = get_destination(destination)
        rep = render(light_scene(), 0.5, out=str(tmp_out / f"{destination}.wav"), destination=destination)
        x, sr = sf.read(rep["path"], always_2d=True)
        assert sr == 48000 and x.shape[0] == int(0.5 * 48000)
        assert x.shape[1] == (4 if destination == "vr" else 2)            # vr -> AmbiX FOA (YouTube 360)
        assert_clean(x.T)
        tp = dbtp(x.T)
        ceiling = d["tp_dbtp"] if d["tp_dbtp"] is not None else SAFETY_CEILING_DBTP
        assert tp <= ceiling + 0.17                                      # 8x detector: <= 0.17 dB under-read
        if d["normalize"]:
            # short (0.5 s) programme: one or two gating blocks; +-1 LU
            assert rep["integrated_lufs"] == pytest.approx(d["target_lufs"], abs=1.0)
            assert not any("silent" in w for w in rep["warnings"])
        else:
            assert rep["gain_db"] == 0.0 and rep["target_lufs"] is None
        if x.shape[1] == 2:
            # the report is the meter of the written file (dither adds < 0.01 LU at these levels)
            assert integrated_loudness(x.T, sr) == pytest.approx(rep["integrated_lufs"], abs=0.1)

    def test_real_stems_sum_to_mix(self, tmp_out):
        """Real generators, raw float: every emission lands in one of the 7 stems and the stems sum
        to the mix (decoders are linear; shelter and room act per stem)."""
        import soundfile as sf

        from raingen.scene.presets import preset
        from raingen.render import render

        sc = preset("porch-summer-thunderstorm")      # rain near + far, wind, thunder, aftermath, porch shelter
        sc.render.quality = "mobile"
        rep = render(sc, 0.5, out=str(tmp_out) + os.sep, layout="stereo", destination="raw", bit_depth=32,
                     stems=True)
        mix, _ = sf.read(rep["path"], always_2d=True)
        total = sum(sf.read(p, always_2d=True)[0] for p in rep["stems"].values())
        assert np.max(np.abs(mix)) > 1e-5
        # float32 files: rounding of 8 files, < 1e-5 of full scale
        assert np.max(np.abs(total - mix)) <= 1e-5 * max(1.0, np.max(np.abs(mix))) + 1e-7
        assert not any("unknown stem" in w for w in rep["warnings"])

    @pytest.mark.slow                  # 3 s of real synthesis, two passes
    def test_real_normalised_render(self):
        """A real preset, 3 s, youtube: -14 LUFS +-0.5 LU, <= -1 dBTP (+0.17 dB 8x detector)."""
        from raingen.render import Renderer
        from raingen.scene.presets import preset

        r = Renderer(preset("steady-rain-leaves-night"), 3.0, layout="stereo", destination="youtube",
                     quality="mobile")
        y, _ = r.to_array()
        assert r.pass2 == "replay"
        assert lufs(y) == pytest.approx(-14.0, abs=0.5)
        assert dbtp(y) <= -1.0 + 0.17
        assert_clean(y)

    def test_louder_rain_is_louder(self):
        """2 mm/h vs 16 mm/h: impact rate ~ R, so ~ +9 dB (10 log10 8); accept +3..+15 LU
        (Marshall-Palmer DSD: larger drops at higher rates add level; Nystuen 2001)."""
        from raingen.render import render_array

        lo, _ = render_array(light_scene(rate=2.0), 0.5, layout="stereo", destination="raw", bit_depth=32)
        hi, _ = render_array(light_scene(rate=16.0), 0.5, layout="stereo", destination="raw", bit_depth=32)
        d = lufs(hi) - lufs(lo)
        assert 3.0 <= d <= 15.0, d

    def test_short_real_clip_normalised(self):
        from raingen.render import Renderer

        r = Renderer(light_scene(), 0.25, layout="stereo", destination="youtube")
        y, _ = r.to_array()
        assert math.isfinite(r._measured) and not any("silent" in w for w in r.warnings)
        assert lufs(y) == pytest.approx(-14.0, abs=1.0)
        assert dbtp(y) <= -1.0 + 0.17


# ===========================================================================
# real-time engine without a sound card
# ===========================================================================
def _engine(scene, **kw):
    from raingen.engine import Engine

    kw.setdefault("prime_s", 0.5)
    kw.setdefault("buffer_s", 0.3)
    kw.setdefault("target_lufs", None)
    kw.setdefault("quality", "mobile")
    return Engine(scene, layout="stereo", **kw)


def _read_all(eng, seconds: float, chunk: int = 2048) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    parts = []
    while n > 0:
        k = min(chunk, n)
        parts.append(eng.read(k, timeout=120.0))
        n -= k
    return np.concatenate(parts, axis=0).T


class TestEngine:
    def test_read_markers_and_time(self, synth):
        T = [int(0.2 * SAMPLE_RATE), int(0.9 * SAMPLE_RATE)]
        synth(lambda: [ImpulseGen(T), NoiseGen(0.005, clicks_per_s=0.0)])
        got = []
        eng = _engine(calm_scene(), on_marker=got.append).start(audio=False)
        try:
            y = _read_all(eng, 0.5)
            first = eng.poll()
            y = np.concatenate([y, _read_all(eng, 0.7)], axis=1)
            second = eng.poll()
        finally:
            eng.stop()
        assert y.shape[0] == 2 and np.all(np.isfinite(y))
        # poll() delivers a marker once its audio has been played, never before (engine.py contract)
        assert [round(m.t * SAMPLE_RATE) for m in first] == T[:1]
        assert [round(m.t * SAMPLE_RATE) for m in second] == T[1:]
        assert [m.t for m in got] == [m.t for m in first + second]
        assert eng.markers_delivered == 2
        # the engine does not trim latency: the impulse is heard chain_latency samples later
        lat = eng._chain_latency
        env = np.max(np.abs(y), axis=0)
        k = int(np.argmax(env[T[0] - 200 + lat: T[0] + 200 + lat])) + T[0] - 200 + lat
        assert abs(k - (T[0] + lat)) <= 1
        assert eng.time == pytest.approx(1.2 - lat / SAMPLE_RATE, abs=1e-6)

    def test_underrun_accounting(self, synth):
        synth(lambda: [NoiseGen(0.005)])
        eng = _engine(calm_scene()).start(audio=False)
        eng.stop()
        queued = eng._pushed - eng._consumed
        y = eng.read(queued + 4096)
        assert eng.underruns == 1
        assert not np.any(y[queued:]) and np.any(y[:queued])
        t = eng.time
        eng.read(1024)                                       # pure silence: the scene clock stands still
        assert eng.underruns == 2 and eng.time == t

    def test_volume_glide_is_smooth(self, synth):
        """vibe.volume_db glides per sample: no per-block steps (zipper)."""
        synth(lambda: [SineGen(997.0, 0.05)])
        ref_eng = _engine(calm_scene()).start(audio=False)
        try:
            ref = _read_all(ref_eng, 1.6)
        finally:
            ref_eng.stop()
        eng = _engine(calm_scene()).start(audio=False)
        try:
            y = _read_all(eng, 0.4)
            eng.set("vibe.volume_db", -12.0, ramp=0.5)
            y = np.concatenate([y, _read_all(eng, 1.2)], axis=1)
        finally:
            eng.stop()
        mask = np.abs(ref[0]) > 0.5 * np.max(np.abs(ref[0]))
        g = y[0][mask] / ref[0][mask]
        idx = np.nonzero(mask)[0]
        assert g[0] == pytest.approx(1.0, abs=1e-4) and g[-1] == pytest.approx(10 ** (-12 / 20), abs=1e-4)
        # gain may only move by ~ (total change) / (ramp samples) per sample; a block step is ~1000x that
        slope = np.abs(np.diff(g)) / np.diff(idx)
        assert np.max(slope) <= 3.0 * (1.0 - 10 ** (-12 / 20)) / (0.5 * SAMPLE_RATE)
        assert np.all(np.diff(g) <= 1e-6)                     # monotone fade (float32 output: eps 1.2e-7)

    def test_rain_rate_glide(self, synth):
        """set() glides the rain rate in log2 over >= the manifest smoothing time, never a step."""
        from raingen.scene import manifest as mf

        gens = {}

        def factory():
            g = NoiseGen(0.01, clicks_per_s=0.0, rate_gain=True)
            gens.setdefault("all", []).append(g)
            return [g]

        synth(factory)
        sc = calm_scene()
        sc.weather.precipitation.type = "rain"
        sc.weather.precipitation.rate_mm_h = 2.0
        eng = _engine(sc).start(audio=False)
        try:
            y = _read_all(eng, 0.3)
            eng.set("rain", 16.0, ramp=0.5)
            y = np.concatenate([y, _read_all(eng, 4.5)], axis=1)
        finally:
            eng.stop()
        rates = np.asarray(gens["all"][0].rates)
        spec = mf.find_param("weather.precipitation.rate_mm_h")
        lr = np.log2(np.maximum(rates, 1e-6))
        steps = np.abs(np.diff(lr))
        # a log2 glide of 3 octaves over >= smoothing_s: per block <= 3 / (smoothing_s * 46.875) (+ World modulation)
        per_block = 3.0 / (max(spec.smoothing_s, 0.5) * SAMPLE_RATE / BLOCK)
        assert np.max(steps) <= per_block + 0.15, (np.max(steps), per_block)
        assert rates[-1] / rates[0] == pytest.approx(8.0, rel=0.5)
        assert_clean(y)
        assert_no_block_clicks(y, offsets=(0, eng._chain_latency % BLOCK))

    @pytest.mark.parametrize("phase", [0.5 * math.pi, 0.0])
    def test_structural_change_crossfades(self, synth, phase):
        """Changing the shelter rebuilds the pipeline and crossfades: no click, no level bump.

        The rebuilt pipeline plays the same sine shifted by `phase`: pi/2 (uncorrelated sides; a
        hard switch would be a step of ~1.4x the amplitude) and 0 (identical sides, where a plain
        cos/sin crossfade would swell by +3 dB at its midpoint).
        """
        built = []

        def factory():
            built.append(1)
            return [SineGen(500.0, 0.05, phase=0.0 if len(built) == 1 else phase)]

        synth(factory)
        eng = _engine(calm_scene(), xfade_s=0.3).start(audio=False)
        try:
            y = _read_all(eng, 0.4)
            eng.set("place.shelter", "porch")
            y = np.concatenate([y, _read_all(eng, 1.6)], axis=1)
            cur = eng._current.place.shelter
        finally:
            eng.stop()
        assert cur == "porch" and len(built) == 2
        assert_clean(y)
        assert_no_block_clicks(y, offsets=(0, eng._chain_latency % BLOCK))
        d = np.abs(np.diff(y[0]))
        # a 500 Hz sine of amplitude A changes by at most 2 pi f A / sr per sample (+5 % for the fade)
        assert np.max(d[4096:]) <= 1.05 * 2 * math.pi * 500.0 * np.max(np.abs(y[0, 4096:])) / SAMPLE_RATE
        # level per 1024-sample block: the open-air and porch transfers differ a little; no swell or dip
        blocks = y[:, 4096: 4096 + ((y.shape[1] - 4096) // BLOCK) * BLOCK].reshape(2, -1, BLOCK)
        lev = 10 * np.log10(np.mean(blocks ** 2, axis=(0, 2)) + 1e-20)
        assert np.max(lev) - np.min(lev) < 1.5, lev

    def test_set_coerces_units_and_rejects_unknown(self, synth):
        synth(lambda: [NoiseGen(0.005)])
        eng = _engine(calm_scene())
        assert eng.set("wind", "36 km/h") == pytest.approx(10.0)              # 36 km/h = 10 m/s
        assert eng.set("rain", "8 mm/h") == pytest.approx(8.0)
        with pytest.raises(KeyError):
            eng.set("weather.no_such_param", 1.0)

    def test_mode_change_keeps_timeline(self, synth):
        """A wellness mode change swaps the governor at the same total latency (markers stay aligned)."""
        synth(lambda: [NoiseGen(0.005, clicks_per_s=0.0)])
        eng = _engine(calm_scene(), xfade_s=0.2).start(audio=False)
        try:
            lat0 = eng._chain_latency
            y = _read_all(eng, 0.3)
            eng.set("mode", "sleep")
            y = np.concatenate([y, _read_all(eng, 1.2)], axis=1)
            lat1 = eng._chain_latency
        finally:
            eng.stop()
        assert lat0 == lat1
        assert_clean(y)
        assert_no_block_clicks(y, offsets=(0, lat0 % BLOCK))

    @pytest.mark.slow
    def test_real_scene_engine(self):
        eng = _engine(light_scene(), target_lufs=-20.0).start(audio=False)
        try:
            y = _read_all(eng, 1.0)
            eng.set("wind", "force 4", ramp=0.5)
            y = np.concatenate([y, _read_all(eng, 1.0)], axis=1)
        finally:
            eng.stop()
        assert_clean(y)
        assert dbtp(y) <= -1.0 + 0.17
        # the engine calibrates its start (prime) to -20 LUFS: +-3 LU over the first second
        assert lufs(y[:, : SAMPLE_RATE]) == pytest.approx(-20.0, abs=3.0)
        assert_no_block_clicks(y, offsets=(0, eng._chain_latency % BLOCK))


# ===========================================================================
# command line
# ===========================================================================
def cli(*args):
    from raingen.cli import main
    return main([str(a) for a in args])


class TestCli:
    COMMANDS = ["render", "variations", "presets", "metar", "live", "play", "manifest", "lint-claims", "scene",
                "destinations"]

    @pytest.mark.parametrize("command", [None, *COMMANDS])
    def test_help(self, command, capsys):
        with pytest.raises(SystemExit) as e:
            cli(*([command] if command else []), "--help")
        assert e.value.code == 0
        assert "usage: raingen" in capsys.readouterr().out

    def test_version_and_module_entry(self):
        r = subprocess.run([sys.executable, "-m", "raingen", "--version"], cwd=REPO, capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0 and r.stdout.strip().startswith("raingen ")

    def test_presets_and_destinations(self, capsys):
        from raingen.scene.presets import PRESETS

        assert cli("presets") == 0
        out = capsys.readouterr().out
        assert all(name in out for name in PRESETS) and len(PRESETS) == 28
        assert cli("presets", "--json") == 0
        assert len(json.loads(capsys.readouterr().out)) == 28
        assert cli("presets", "--tag", "no-such-tag") == 1
        capsys.readouterr()
        assert cli("destinations") == 0
        out = capsys.readouterr().out
        assert "broadcast_eu" in out and "-23 LUFS" in out
        assert cli("destinations", "--json") == 0
        rows = {d["name"]: d for d in json.loads(capsys.readouterr().out)}
        assert rows["broadcast_us"]["target_lufs"] == -24.0 and rows["broadcast_us"]["tp_dbtp"] == -2.0

    @pytest.mark.parametrize("fmt", ["json-schema", "osc", "wwise", "fmod", "markdown", "json"])
    def test_manifest(self, fmt, tmp_out, capsys):
        out = tmp_out / f"manifest.{fmt}"
        assert cli("manifest", "--format", fmt, "--out", out) == 0
        text = out.read_text()
        assert len(text) > 200 and "rate_mm_h" in text
        if fmt in ("json-schema", "json"):
            json.loads(text)

    def test_scene_text_preset_metar(self, tmp_out, capsys):
        from raingen.scene import Scene

        p1 = tmp_out / "text.json"
        assert cli("scene", "heavy", "rain", "on", "a", "tin", "roof", "--out", p1, "--seed", "0x10") == 0
        s1 = Scene.from_json(p1.read_text())
        assert s1.seed == 16 and s1.weather.precipitation.rate_mm_h >= 4.0     # 'heavy' rain (WMO: > 4 mm/h)
        p2 = tmp_out / "preset.yaml"
        assert cli("scene", "tent-in-a-storm", "--out", p2) == 0
        assert p2.read_text().strip()
        p3 = tmp_out / "metar.json"
        assert cli("scene", "KJFK 121755Z 21016G24KT 1SM +TSRA BR BKN015CB 18/17 A2990", "--out", p3) == 0
        s3 = Scene.from_json(p3.read_text())
        # METAR: 16 kt = 8.2 m/s from 210 deg, +TSRA = heavy rain with thunder, 18 C
        assert s3.weather.wind.speed_mps == pytest.approx(8.23, abs=0.1)
        assert s3.weather.wind.from_deg == pytest.approx(210.0)
        assert s3.weather.thunder.enabled and s3.weather.temperature_c == pytest.approx(18.0)
        capsys.readouterr()
        assert cli("scene", "steady-rain-leaves-night", "--describe") == 0
        assert "Precipitation" in capsys.readouterr().out

    def test_metar_print(self, capsys):
        # unquoted METAR: '-RA' (light rain) must not be taken for an option
        assert cli("metar", "EGLL", "121750Z", "24012KT", "9999", "-RA", "BKN020", "12/09", "Q1008") == 0
        d = json.loads(capsys.readouterr().out)
        assert d["weather"]["precipitation"]["type"] in ("rain", "drizzle", "showers")
        assert d["weather"]["pressure_hpa"] == pytest.approx(1008.0)

    def test_lint_claims(self, tmp_out, capsys):
        good = tmp_out / "good.md"
        good.write_text("Rain sounds for relaxing evenings.\n")
        assert cli("lint-claims", good) == 0
        assert cli("lint-claims", tmp_out / "missing.md") == 1
        assert "not found" in capsys.readouterr().err

    def test_error_exit_codes(self, tmp_out, capsys):
        assert cli("render", "no_such_scene.json", "--seconds", "1") == 1          # FileNotFoundError
        assert "raingen: error:" in capsys.readouterr().err
        assert cli("render", "zzqx", "vvbn", "--strict", "--seconds", "1") == 1   # SceneParseError
        assert cli("render", "steady-rain-leaves-night", "--seconds", "-1") == 1  # ValueError
        assert cli("live") == 1                                                   # CLIError: no location
        with pytest.raises(SystemExit) as e:
            cli("render", "x", "--layout", "9.1")                                # argparse: exit 2
        assert e.value.code == 2
        with pytest.raises(SystemExit) as e:
            cli("render", "x", "--seed", "notanumber")
        assert e.value.code == 2
        with pytest.raises(SystemExit) as e:
            cli("render", "x", "--destination", "nowhere")
        assert e.value.code == 2
        from raingen.cli import build_parser
        a = build_parser().parse_args(["render", "x", "--destination", "EBU-R128"])
        assert a.destination == "broadcast_eu"                                   # aliases accepted

    @pytest.mark.parametrize("ext", ["wav", "flac"])
    def test_render_with_stems_and_captions(self, ext, synth, tmp_out, capsys):
        import soundfile as sf

        synth(lambda: [NoiseGen(0.02, stem="rain_near", clicks_per_s=0.0),
                       ImpulseGen([int(0.2 * SAMPLE_RATE)], amp=0.05, stem="thunder")])
        scene_path = tmp_out / "calm.json"
        calm_scene().save(str(scene_path))
        out = tmp_out / f"out.{ext}"
        code = cli("render", scene_path, "--seconds", "0.5", "--out", out, "--stems", "--captions", "--json",
                   "--destination", "youtube")
        assert code == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["path"] == str(out) and os.path.isfile(out)
        assert len(rep["stems"]) == 7
        assert all(p.endswith(f".{ext}") and os.path.isfile(p) for p in rep["stems"].values())
        assert set(rep["captions"]) == {"srt", "csv"}
        x, sr = sf.read(str(out), always_2d=True)
        assert x.shape == (24000, 2) and sr == 48000
        # 0.5 s = one gating block: BS.1770 integrated loudness of the file, +-0.5 LU of the youtube target
        assert lufs(x.T) == pytest.approx(-14.0, abs=0.5)
        assert dbtp(x.T) <= -1.0 + 0.17
        assert rep["markers"] == 1 and not rep["warnings"]
        # stems share the mix's gain (post_stems-style relative levels): their sum is the mix before limiting
        total = sum(sf.read(p, always_2d=True)[0] for p in rep["stems"].values())
        assert lufs(total.T) == pytest.approx(lufs(x.T), abs=0.2)


# ===========================================================================
# fixes routed from the other areas (spatial, master-io, scene-control, legacy-ci, site)
# ===========================================================================
class DiffuseImpulseGen:
    """One impulse of a fully diffuse (spread 1) source, `before_end` samples before a block boundary."""

    def __init__(self, at: int, amp: float = 0.5, azimuth: float = 0.7, elevation: float = 0.2):
        self.name = "test.diffuse"
        self.at, self.amp, self.azimuth, self.elevation = int(at), float(amp), float(azimuth), float(elevation)

    def render(self, ctx: BlockContext) -> list:
        s = np.zeros(ctx.n)
        start = int(round(ctx.t0 * ctx.sr))
        if start <= self.at < start + ctx.n:
            s[self.at - start] = self.amp
        return [Emission(s, azimuth=self.azimuth, elevation=self.elevation, spread=1.0, stem="rain_far",
                         source_id="diffuse")]


class TestRoutedFixes:
    def test_encoder_batches_per_stem_and_keeps_decorrelator_tails(self, synth):
        """BlockRenderer encodes with encode_into + finish_block: a diffuse impulse 5 samples before a
        block boundary keeps its 20 ms all-pass tail in the next block, with the energy the SN3D
        diffuse gains promise (an all-pass has unit energy: Parseval)."""
        from raingen.render import render_array
        from raingen.spatial.encoder import spread_gains

        T = 4 * BLOCK - 5
        synth(lambda: [DiffuseImpulseGen(T)])
        y, _ = render_array(calm_scene(), 0.25, layout="hoa3", destination="raw", bit_depth=32)
        g_coh, g_dif = spread_gains(3, 0.7, 0.2, 1.0)
        hi = y[1:]                                            # ACN 1..15: diffuse part only at spread 1
        expect = 0.5 ** 2 * float(np.sum(g_dif[1:] ** 2) + np.sum(g_coh[1:] ** 2))
        total = float(np.sum(hi ** 2))
        after = float(np.sum(hi[:, 4 * BLOCK:] ** 2))
        # all-pass FIRs are flat to ~0.05 dB (encoder.design_allpass_firs): energy within 5 %
        assert total == pytest.approx(expect, rel=0.05)
        # the FIR envelope decays over ~5 ms: nearly all of it lies past the boundary, in the next block
        assert after >= 0.9 * total
        # W carries the dry impulse at its sample (decoder passthrough, no latency)
        assert int(np.argmax(np.abs(y[0]))) == T

    def test_short_clip_warns_provisional(self, synth):
        """< 400 ms: normalised on the meter's provisional reading, and the report says so (BS.1770-5)."""
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(0.01, clicks_per_s=0.0)])
        r = Renderer(calm_scene(), 0.3, layout="stereo", destination="broadcast_eu")
        y, _ = r.to_array()
        assert any("provisional" in w for w in r.summary["warnings"])
        assert r.summary["loudness_provisional"] is True
        assert r.summary["integrated_lufs"] == pytest.approx(-23.0, abs=0.5)

    @pytest.mark.parametrize("destination", ["youtube", "sleep"])
    def test_programme_below_absolute_gate_is_normalised(self, synth, destination):
        """Rain through a closed window measured ~20 dB SPL (about -75 LUFS at 1.0 = 1 Pa): no 400 ms
        block passes the -70 LUFS absolute gate, so BS.1770 gives no integrated value. It must still
        be brought to the target (on its ungated loudness), not delivered inaudible as 'silent'."""
        from raingen.master.destinations import get_destination
        from raingen.render import Renderer

        synth(lambda: [NoiseGen(4e-5, clicks_per_s=0.0)])
        r = Renderer(calm_scene(), 1.0, layout="stereo", destination=destination)
        y, _ = r.to_array()
        assert r._measured < -70.0 and r._measure_note == "below_gate"
        assert not any("silent" in w for w in r.warnings)
        assert any("absolute gate" in w for w in r.warnings)
        # delivered programme, gated BS.1770-5 measurement: +-0.5 LU of the destination target
        assert lufs(y) == pytest.approx(get_destination(destination)["target_lufs"], abs=0.5)

    def test_unknown_destination_is_an_error(self):
        from raingen.render import Renderer

        with pytest.raises(ValueError, match="unknown destination"):
            Renderer(calm_scene(), 1.0, destination="youtbe")
        # aliases resolve (raingen.master.destinations.ALIASES)
        assert Renderer(calm_scene(), 1.0, destination="EBU-R128").dest["name"] == "broadcast_eu"
        assert Renderer(calm_scene(), 1.0, destination="ott").dest["name"] == "online_streaming"

    def test_scene_parse_error_is_text_parse_error(self):
        import raingen as rg
        from raingen.api import _resolve
        from raingen.scene.textparse import TextParseError

        assert issubclass(rg.SceneParseError, TextParseError) and issubclass(rg.SceneParseError, ValueError)
        with pytest.raises(rg.SceneParseError, match="nothing recognised"):
            _resolve("qwzx vbnm", strict=True)                  # textparse's strict check, re-raised
        sc, understood = _resolve("heavy rain, wind force 6", strict=True)
        assert understood and sc.weather.precipitation.rate_mm_h >= 4.0     # WMO 'heavy': > 4 mm/h

    def test_drop_frame_timecode_in_bext_and_ixml(self, synth, tmp_out):
        """29.97 DF '01:00:00;00' = 107892 frames = 3599.9964 s (SMPTE ST 12-1) -> bext TimeReference
        172 799 827 samples at 48 kHz (EBU Tech 3285), iXML TIMECODE_FLAG DF / rate 30000/1001."""
        import struct

        from raingen.render import render

        synth(lambda: [NoiseGen(0.005)])
        rep = render(calm_scene(), 0.25, out=str(tmp_out / "df.wav"), destination="raw", start_tc="01:00:00;00",
                     fps=29.97)
        raw = open(rep["path"], "rb").read()
        k = raw.index(b"bext") + 8
        tref = struct.unpack("<Q", raw[k + 338: k + 346])[0]
        assert tref == round(107892 * 1001 / 30000 * 48000)
        assert b"<TIMECODE_FLAG>DF</TIMECODE_FLAG>" in raw
        assert b"<TIMECODE_RATE>30000/1001</TIMECODE_RATE>" in raw
        assert b"<FILE_SAMPLE_RATE>48000</FILE_SAMPLE_RATE>" in raw

    def test_locked_destination_keeps_finished_render(self, synth, tmp_out, monkeypatch):
        """If the finished file cannot replace the destination (Windows: open in a player), the
        render is kept in its .part file and the error says where (nothing is deleted)."""
        from raingen.render import render

        synth(lambda: [NoiseGen(0.005)])
        target = tmp_out / "locked.wav"
        real_replace = os.replace

        def replace(src, dst, *a, **kw):
            if os.fspath(dst) == str(target):
                raise PermissionError(13, "The process cannot access the file", str(dst))
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(os, "replace", replace)
        with pytest.raises(OSError) as e:
            render(calm_scene(), 0.25, out=str(target), destination="raw", stems=True)
        part = str(target) + ".part"
        assert part in str(e.value) and os.path.isfile(part) and os.path.getsize(part) > 12000 * 2 * 3
        assert not target.exists()
        # the stems were not blocked and were put in place
        assert len([p for p in os.listdir(tmp_out) if p.endswith(".wav")]) == 7

    def test_one_shots_ucs_names(self, tmp_out, monkeypatch):
        import soundfile as sf

        import raingen.api as api
        from raingen.io.ucs import UCS_CATIDS

        monkeypatch.setattr(api, "thunder_one_shots",
                            lambda count, **kw: [np.full((2, 4800), 0.1 * (i + 1)) for i in range(count)])
        paths = api.one_shots("thunder", count=2, out_dir=str(tmp_out), distance_km=2.5)
        names = [os.path.basename(p) for p in paths]
        # UCS v8.2.1 CatID THUN = WEATHER / THUNDER; CatID_FXName_CreatorID_SourceID_UserData
        assert all(re.fullmatch(r"THUN_Thunder 2\.5 km one-shot_RGN_raingen_0[12]\.wav", n) for n in names), names
        assert UCS_CATIDS["THUN"] == ("WEATHER", "THUNDER")
        assert sf.info(paths[0]).frames == 4800
        with pytest.raises(ValueError):
            api.one_shots("hail", count=1)

    @pytest.mark.slow                  # four thunder-only renders (strike, roll and echoes)
    def test_thunder_one_shots(self):
        """CG strikes at 3 km: one clip per strike, starting 50 ms before the first arrival
        (flash-to-bang removed), low-frequency rumble, peak 0.9, deterministic per seed."""
        from raingen.render import thunder_one_shots

        a = thunder_one_shots(2, distance_km=3.0, max_s=2.0, seed=11)
        assert len(a) == 2
        for x in a:
            assert x.shape[0] == 2 and 0.5 * SAMPLE_RATE <= x.shape[1] <= 2.0 * SAMPLE_RATE + 1
            assert np.all(np.isfinite(x)) and np.max(np.abs(x)) == pytest.approx(0.9)
            env = np.max(np.abs(x), axis=0)
            onset = int(np.argmax(env > 0.01 * env.max()))
            # the cut is 50 ms before the first arrival (one block of rounding)
            assert abs(onset - int(0.05 * SAMPLE_RATE)) <= BLOCK, onset
            # distant thunder is a low rumble: most energy below 200 Hz (Holmes et al. 1971: spectral
            # peak 50-100 Hz at several km; ISO 9613-1 absorption removes the crack at 3 km)
            assert centroid_hz(x) < 400.0
            assert np.all(np.abs(x[:, -10:]) < 0.01)                    # faded out
        b = thunder_one_shots(2, distance_km=3.0, max_s=2.0, seed=11)
        assert all(np.array_equal(p, q) for p, q in zip(a, b))
        c = thunder_one_shots(1, distance_km=3.0, max_s=2.0, seed=12)
        assert not np.array_equal(a[0][:, : c[0].shape[1]], c[0][:, : a[0].shape[1]])
        # legacy caller (rain_generator.audio): 44.1 kHz mixer rate
        d = thunder_one_shots(1, distance_km=2.5, sample_rate=44100, max_s=1.0, seed=11)
        assert len(d) == 1 and d[0].shape[1] <= 44100 + 1


class TestCliRouted:
    def test_destination_choices_follow_the_table(self, capsys):
        from raingen.cli import build_parser
        from raingen.master.destinations import DESTINATIONS

        p = build_parser()
        for name in DESTINATIONS:
            assert p.parse_args(["render", "x", "--destination", name]).destination == name
        assert p.parse_args(["render", "x", "-d", "ott"]).destination == "online_streaming"
        with pytest.raises(SystemExit):
            p.parse_args(["render", "x", "--help"])
        out = capsys.readouterr().out
        assert "online_streaming" in out

    @pytest.mark.slow                  # real synthesis in a fresh interpreter
    def test_real_render_through_module_entry(self, tmp_out):
        """`python -m raingen render <preset> ...` end to end: real generators, flac + stems + captions,
        BLAS limited to one thread by the entry point (cli.limit_blas_threads)."""
        import soundfile as sf

        out = tmp_out / "real.flac"
        env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        r = subprocess.run([sys.executable, "-m", "raingen", "render", "steady-rain-leaves-night", "--seconds", "0.5",
                            "--quality", "mobile", "--out", str(out), "--stems", "--captions", "srt", "vtt",
                            "--destination", "youtube", "--json", "--quiet"],
                           cwd=REPO, capture_output=True, text=True, timeout=600, env=env)
        assert r.returncode == 0, r.stderr
        rep = json.loads(r.stdout)
        assert rep["path"] == str(out) and set(rep["captions"]) == {"srt", "vtt"} and len(rep["stems"]) == 7
        x, sr = sf.read(str(out), always_2d=True)
        assert (sr, x.shape) == (48000, (24000, 2))
        assert_clean(x.T)
        # 0.5 s: one 400 ms gating block (BS.1770-5); +-0.5 LU of YouTube's -14 LUFS; <= -1 dBTP (+0.17 dB)
        assert lufs(x.T) == pytest.approx(-14.0, abs=0.5)
        assert dbtp(x.T) <= -1.0 + 0.17

    def test_unencodable_labels_do_not_crash(self, monkeypatch, tmp_out):
        """A cp1252 pipe (Windows) cannot encode every character: main() switches stdout/stderr to
        errors='backslashreplace' instead of raising UnicodeEncodeError."""
        import io

        from raingen.cli import main

        buf = io.BytesIO()
        out = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
        monkeypatch.setattr(sys, "stdout", out)
        assert main(["scene", "steady-rain-leaves-night", "--describe"]) == 0
        out.flush()
        assert b"Precipitation" in buf.getvalue()
        print("rain → roof ☔")
        out.flush()
        assert b"\\u2192" in buf.getvalue()
