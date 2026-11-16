"""Legacy audio: the live procedural synthesizer, SoundEngine thunder, and the file renderers
that delegate to raingen (with the old synthesizer as fallback).

Audio is checked with numbers: block-boundary jump statistics (no flutter), Poisson drop
statistics, Butterworth responses, BS.1770 loudness / true peak of the delegated renders, and
the fallback paths with raingen made to fail.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import logging
import math
import sys

import numpy as np
import pygame
import pytest
from scipy.signal import sosfilt

import rain_generator.audio as audio_mod
from rain_generator.audio import (
    lowpass_filter,
    bandpass_filter,
    drop_waveform,
    mixer_pcm,
    ProceduralRainSynthesizer,
    SoundEngine,
)

SR = 44100


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def boundary_jump_ratio(x: np.ndarray, block: int) -> float:
    """Mean |x[n] - x[n-1]| across block boundaries over the mean everywhere else.

    About 1 for a signal that runs on seamlessly from block to block; a filter restarted from
    rest (or a tail cut off) at every block makes the boundary jumps several times larger.
    """
    d = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    idx = np.arange(block, x.size, block) - 1           # d[k*block - 1] = x[k*block] - x[k*block - 1]
    rest = np.ones(d.size, dtype=bool)
    rest[idx] = False
    return float(d[idx].mean() / d[rest].mean())


def impulse_gain_db(filter_fn, f_hz: float, fs: int, n: int = 1 << 16) -> float:
    """Magnitude response of a (linear) filter function at f_hz from its impulse response."""
    imp = np.zeros(n)
    imp[0] = 1.0
    h = filter_fn(imp)
    H = np.fft.rfft(h)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return float(20.0 * np.log10(np.abs(np.interp(f_hz, f, np.abs(H)))))


def short_programme_lufs(x: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 loudness of a whole clip without gating: K-weighting, channel mean squares
    summed with weight 1 (L/R), -0.691 + 10 log10. Used for clips shorter than one 400 ms
    gating block, where integrated loudness is undefined."""
    from raingen.master.loudness import k_weighting_sos

    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = sosfilt(k_weighting_sos(sr), x, axis=-1)
    z = float(np.sum(np.mean(y * y, axis=1)))
    return -0.691 + 10.0 * math.log10(z) if z > 0.0 else float("-inf")


def int16_to_float(pcm: np.ndarray) -> np.ndarray:
    """(n, C) int16 -> (C, n) float in [-1, 1)."""
    return (np.asarray(pcm, dtype=np.float64) / 32768.0).T


def _fresh_synth(rate: float, length: int, rng: np.random.Generator) -> np.ndarray:
    s = ProceduralRainSynthesizer(SR)
    s._rng = rng                                       # share one stream across fresh synths
    return s.generate_drop_impacts(length, rate)


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
def test_lowpass_filter():
    fs = 44100
    data = np.random.default_rng(0).normal(0, 1.0, 1000)
    filtered = lowpass_filter(data, cutoff=2000, fs=fs, order=4)
    assert filtered.shape == data.shape
    assert np.all(np.isfinite(filtered))


@pytest.mark.parametrize("order, cutoff", [(4, 2000.0), (8, 1800.0), (8, 200.0)])
def test_lowpass_filter_is_butterworth(order, cutoff):
    fs = 44100
    fn = lambda x: lowpass_filter(x, cutoff=cutoff, fs=fs, order=order)  # noqa: E731
    # Butterworth definition: |H| = 1 at DC and 1/sqrt(2) (-3.01 dB) at the cutoff, for any order
    assert impulse_gain_db(fn, 1.0, fs) == pytest.approx(0.0, abs=0.05)
    assert impulse_gain_db(fn, cutoff, fs) == pytest.approx(-3.01, abs=0.1)
    # roll-off 6*order dB/octave: one octave above the cutoff the analog response is
    # -10 log10(1 + 4**order) dB; the bilinear transform only steepens it below Nyquist
    assert impulse_gain_db(fn, 2.0 * cutoff, fs) <= -10.0 * math.log10(1.0 + 4.0 ** order) + 0.5


def test_bandpass_filter():
    fs = 44100
    data = np.random.default_rng(1).normal(0, 1.0, 1000)
    filtered = bandpass_filter(data, lowcut=200, highcut=600, fs=fs, order=2)
    assert filtered.shape == data.shape
    assert np.all(np.isfinite(filtered))
    fn = lambda x: bandpass_filter(x, lowcut=200, highcut=600, fs=fs, order=2)  # noqa: E731
    # Butterworth band-pass: unity at the geometric centre, -3.01 dB at both band edges
    assert impulse_gain_db(fn, math.sqrt(200 * 600), fs) == pytest.approx(0.0, abs=0.1)
    assert impulse_gain_db(fn, 200.0, fs) == pytest.approx(-3.01, abs=0.15)
    assert impulse_gain_db(fn, 600.0, fs) == pytest.approx(-3.01, abs=0.15)


def test_legacy_portuguese_lowpass_alias_is_well_conditioned():
    import thunderstorm_simulator as ts

    fs = 44100
    # the alias defaults to an 8th-order filter; in (b, a) form it read -3.26 dB at 200 Hz
    fn = lambda x: ts.filtro_passa_baixa(x, 200, fs)  # noqa: E731
    assert impulse_gain_db(fn, 200.0, fs) == pytest.approx(-3.01, abs=0.1)  # Butterworth -3.01 dB point
    rain = ts.gerar_ruido_chuva(4096, fs, 2.0)
    assert rain.shape == (4096,)
    assert np.all(np.isfinite(rain)) and np.max(np.abs(rain)) <= 1.0


# ---------------------------------------------------------------------------
# procedural synthesizer
# ---------------------------------------------------------------------------
def test_procedural_synthesizer():
    synth = ProceduralRainSynthesizer(sample_rate=44100)

    noise = synth.generate_rain_noise(2048, intensity=1.5)
    assert len(noise) == 2048
    assert np.all(np.isfinite(noise))
    assert np.max(np.abs(noise)) < 2.0

    drops = synth.generate_drop_impacts(2048, rate_multiplier=2.0)
    assert len(drops) == 2048
    assert np.all(np.isfinite(drops))

    wind = synth.generate_wind_noise(2048, wind_strength=0.8)
    assert len(wind) == 2048
    assert np.all(np.isfinite(wind))

    # Zero wind produces silence
    calm = synth.generate_wind_noise(1024, wind_strength=0.0)
    assert np.all(calm == 0.0)


def _blocks(kind: str, synth: ProceduralRainSynthesizer, block: int, count: int, restart: bool = False):
    parts = []
    for _ in range(count):
        if kind == "rain":
            if restart:                                 # the pre-v3 behaviour: filter from rest per block
                synth._rain_zi, synth._rain_key = None, None
            parts.append(synth.generate_rain_noise(block, 1.2))
        elif kind == "drops":
            if restart:                                 # drop tails cut at the block end
                synth._drop_tail = np.zeros(0, dtype=np.float32)
            parts.append(synth.generate_drop_impacts(block, 20.0))
        else:
            if restart:
                synth._wind_zi, synth._wind_key, synth._wind_phase = None, None, 0.0
            parts.append(synth.generate_wind_noise(block, 0.8))
    return np.concatenate(parts)


@pytest.mark.parametrize("kind", ["rain", "drops", "wind"])
@pytest.mark.parametrize("block", [512, 2048, 777])
def test_synth_has_no_flutter_at_block_boundaries(kind, block):
    count = max(200, 150_000 // block)
    x = _blocks(kind, ProceduralRainSynthesizer(SR, seed=11), block, count)
    assert np.all(np.isfinite(x))
    # seamless continuation: boundary jumps statistically like any other sample step
    # (>= 200 boundaries -> standard error of the ratio about 5 %, so +-20 % is 4 SE)
    assert 0.8 < boundary_jump_ratio(x, block) < 1.2
    # control: restarting the state every block is what the statistic is built to catch
    y = _blocks(kind, ProceduralRainSynthesizer(SR, seed=11), block, count, restart=True)
    assert boundary_jump_ratio(y, block) > 2.0


def test_synth_handles_any_block_length():
    synth = ProceduralRainSynthesizer(SR, seed=8)
    lengths = [0, 1, 3, 17, 0, 499, 1, 4096] * 40
    parts = []
    for n in lengths:
        r = synth.generate_rain_noise(n, 1.0)
        d = synth.generate_drop_impacts(n, 30.0)
        w = synth.generate_wind_noise(n, 0.6)
        assert r.shape == d.shape == w.shape == (n,)
        parts.append(r + d + w)
    x = np.concatenate(parts)
    assert np.all(np.isfinite(x))
    # a drop tail longer than the following blocks is spread over them, never dropped:
    # the stream stays seamless at the (many, irregular) boundaries
    edges = np.cumsum([n for n in lengths if n])[:-1]
    d = np.abs(np.diff(x))
    rest = np.ones(d.size, dtype=bool)
    rest[edges - 1] = False
    assert 0.75 < d[edges - 1].mean() / d[rest].mean() < 1.33        # ~280 boundaries, +-4 SE


def test_engine_callback_is_continuous_and_bounded():
    engine = SoundEngine()
    engine.synthesizer = ProceduralRainSynthesizer(engine.sample_rate, seed=5)
    engine.update_parameters(master_vol=0.9, rain_vol=0.8, drops_vol=0.7, wind_vol=0.6, thunder_vol=0.5,
                             is_muted=False, rain_intensity=1.9, drop_rate=1.7, wind_speed=1.1)
    frames = 512
    out = []
    for _ in range(300):
        buf = np.zeros((frames, 2), dtype=np.float32)
        engine._audio_callback(buf, frames, None, None)
        out.append(buf.copy())
    y = np.concatenate(out)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0          # tanh soft clip
    assert np.array_equal(y[:, 0], y[:, 1])                             # legacy live output is dual mono
    assert 0.8 < boundary_jump_ratio(y[:, 0], frames) < 1.2             # no flutter at the callback rate
    rms = float(np.sqrt(np.mean(y[:, 0] ** 2)))
    assert rms > 1e-3
    # no DC: |mean| within 4 standard errors of zero for noise band-limited to ~1.9 kHz
    # (about 2 * 1900 * T independent samples)
    n_eff = 2.0 * 1900.0 * y.shape[0] / SR
    assert abs(float(np.mean(y[:, 0]))) < 4.0 * rms / math.sqrt(n_eff)


@pytest.mark.parametrize("rate", [0.1, 0.5])
def test_drop_count_is_poisson_with_fractional_rate(rate):
    """16 drops/s x rate: a 2048-sample block expects 0.074 drops at rate 0.1. The count is
    Poisson, so such blocks still get a drop with probability 1 - exp(-lambda) (truncating
    lambda to an integer would never give one)."""
    rng = np.random.default_rng(2024)
    trials, length = 3000, 2048
    hits = sum(bool(np.any(_fresh_synth(rate, length, rng) != 0.0)) for _ in range(trials))
    lam = 16.0 * rate * length / SR
    p = 1.0 - math.exp(-lam)                                            # Poisson P(N >= 1)
    se = math.sqrt(p * (1.0 - p) / trials)                              # binomial standard error
    assert hits > 0
    assert abs(hits / trials - p) < 4.0 * se


def test_zero_drop_rate_gives_no_drops():
    synth = ProceduralRainSynthesizer(SR, seed=3)
    assert not np.any(synth.generate_drop_impacts(8192, 0.0))


def test_drop_waveform_ends_without_a_click():
    for duration, freq, decay in [(180, 1600.0, 100.0), (499, 3800.0, 40.0), (300, 2500.0, 70.0)]:
        w = drop_waveform(duration, freq, decay, 0.1, SR).astype(np.float64)
        assert w.shape == (duration,)
        assert w[0] == 0.0                                              # zero phase onset
        assert np.max(np.abs(w)) <= 0.1 + 1e-7                         # envelope <= amplitude
        assert abs(w[-1]) <= 1e-3 * np.max(np.abs(w))                   # faded to zero, not truncated
        # before the fade it is exactly the damped oscillator A sin(2 pi f t) exp(-decay t)
        body = duration - max(16, duration // 4)
        t = np.arange(body) / SR
        ref = 0.1 * np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)
        assert np.max(np.abs(w[:body] - ref)) < 1e-6                     # float32 rounding only
        # and the fade only ever attenuates it
        t = np.arange(duration) / SR
        full = np.abs(0.1 * np.sin(2 * np.pi * freq * t) * np.exp(-decay * t))
        assert np.all(np.abs(w) <= full + 1e-6)


def test_drop_impacts_have_no_dc():
    x = _blocks("drops", ProceduralRainSynthesizer(SR, seed=4), 512, 300)
    rms = float(np.sqrt(np.mean(x ** 2)))
    # random polarity: same-sign zero-phase drops gave |mean| = 3.2 % of RMS
    assert abs(float(np.mean(x))) < 0.01 * rms


def test_synthesizer_seed_is_deterministic():
    def run(seed):
        s = ProceduralRainSynthesizer(SR, seed=seed)
        return np.concatenate([s.generate_rain_noise(1024, 1.0) + s.generate_drop_impacts(1024, 3.0)
                               + s.generate_wind_noise(1024, 0.7) for _ in range(8)])

    assert np.array_equal(run(9), run(9))
    assert not np.array_equal(run(9), run(10))


def test_synth_levels_follow_controls():
    def rms_db(x):
        return 10.0 * math.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)))

    n = 1 << 17
    # rain noise amplitude is proportional to intensity: x2 -> +6.02 dB
    a = ProceduralRainSynthesizer(SR, seed=1).generate_rain_noise(n, 1.0)
    b = ProceduralRainSynthesizer(SR, seed=1).generate_rain_noise(n, 2.0)
    assert rms_db(b) - rms_db(a) == pytest.approx(6.02, abs=0.05)
    # Poisson superposition of independent drops: energy proportional to rate, x4 -> +6.02 dB
    a = _blocks_rate(ProceduralRainSynthesizer(SR, seed=2), 4.0)
    b = _blocks_rate(ProceduralRainSynthesizer(SR, seed=3), 16.0)
    assert rms_db(b) - rms_db(a) == pytest.approx(6.02, abs=1.0)
    # stronger wind is louder (gain min(1.5, strength) on a band of equal noise power)
    a = ProceduralRainSynthesizer(SR, seed=1).generate_wind_noise(n, 0.3)
    b = ProceduralRainSynthesizer(SR, seed=1).generate_wind_noise(n, 0.9)
    assert rms_db(b) > rms_db(a) + 6.0


def _blocks_rate(synth: ProceduralRainSynthesizer, rate: float) -> np.ndarray:
    return np.concatenate([synth.generate_drop_impacts(2048, rate) for _ in range(400)])


def test_rain_and_wind_spectra():
    n = 1 << 17
    f = np.fft.rfftfreq(n, 1.0 / SR)
    rain = ProceduralRainSynthesizer(SR, seed=6).generate_rain_noise(n, 1.0)
    P = np.abs(np.fft.rfft(rain)) ** 2
    # 4th-order Butterworth at 1.9 kHz: two octaves up (7.6 kHz) the response is below -45 dB
    low = P[(f > 100) & (f < 1500)].mean()
    high = P[(f > 7600) & (f < 9000)].mean()
    assert 10.0 * math.log10(high / low) < -45.0
    wind = ProceduralRainSynthesizer(SR, seed=6).generate_wind_noise(n, 0.5)
    W = np.abs(np.fft.rfft(wind)) ** 2
    centre = 280.0 + 0.5 * 220.0                                        # the wind band's centre
    lo, hi = max(60.0, centre - 140.0), min(1200.0, centre + 180.0)
    band = W[(f >= lo / 2) & (f <= hi * 2)].sum()
    assert band / W.sum() > 0.9                                         # wind energy sits in its band


# ---------------------------------------------------------------------------
# SoundEngine
# ---------------------------------------------------------------------------
def test_sound_engine_volume_updates():
    engine = SoundEngine()
    engine.update_parameters(
        master_vol=0.9,
        rain_vol=0.5,
        drops_vol=0.4,
        wind_vol=0.3,
        thunder_vol=0.8,
        is_muted=False,
        rain_intensity=1.2,
        drop_rate=1.0,
        wind_speed=0.5
    )
    assert engine.master_volume == 0.9
    assert engine.rain_volume == 0.5
    assert not engine.is_muted

    # Audio callback simulation
    outdata = np.zeros((512, 2), dtype=np.float32)
    engine._audio_callback(outdata, 512, None, None)
    assert outdata.shape == (512, 2)
    assert np.all(np.isfinite(outdata))
    assert np.max(np.abs(outdata)) <= 1.0


def test_sound_engine_mute():
    engine = SoundEngine()
    engine.update_parameters(
        master_vol=1.0,
        rain_vol=1.0,
        drops_vol=1.0,
        wind_vol=1.0,
        thunder_vol=1.0,
        is_muted=True,
        rain_intensity=1.0,
        drop_rate=1.0,
        wind_speed=0.5
    )
    outdata = np.ones((256, 2), dtype=np.float32)
    engine._audio_callback(outdata, 256, None, None)
    assert np.all(outdata == 0.0)


def test_mixer_pcm_formats():
    t = np.arange(441) / SR
    clip = np.vstack([0.5 * np.sin(2 * np.pi * 100 * t), -0.5 * np.sin(2 * np.pi * 100 * t)])
    pcm = mixer_pcm(clip, -16, 2)
    assert pcm.dtype == np.int16 and pcm.shape == (441, 2) and pcm.flags["C_CONTIGUOUS"]
    assert np.max(pcm[:, 0]) == round(0.5 * 32767 * np.max(np.sin(2 * np.pi * 100 * t)))
    assert np.array_equal(pcm[:, 0], -pcm[:, 1])
    assert mixer_pcm(clip, 32, 2).dtype == np.float32
    assert mixer_pcm(clip, 8, 2).dtype == np.uint8 and mixer_pcm(np.zeros((2, 4)), 8, 2)[0, 0] == 128
    assert mixer_pcm(clip, 16, 2).dtype == np.uint16
    assert mixer_pcm(clip, -8, 2).dtype == np.int8
    mono = mixer_pcm(clip, -16, 1)
    assert mono.shape == (441,) and np.all(mono == 0)                  # L + R cancel
    six = mixer_pcm(clip[:1], -16, 6)
    assert six.shape == (441, 6) and np.array_equal(six[:, 0], six[:, 5])
    hot = mixer_pcm(np.full((2, 4), 3.0), -16, 2)                       # clipped, not wrapped
    assert np.all(hot == 32767)


# ---------------------------------------------------------------------------
# thunder clips: WAV files, resampling, raingen-synthesized fallback
# ---------------------------------------------------------------------------
def _write_wav(path, sr, data):
    import scipy.io.wavfile as wavfile

    wavfile.write(str(path), sr, data)


def test_find_thunder_clips(tmp_path, monkeypatch):
    repo_clips = audio_mod.find_thunder_clips()
    assert len(repo_clips) == 10                                        # sounds/thunder_1..10.wav
    tone = (np.sin(np.arange(2400) * 0.05) * 8000).astype(np.int16)
    _write_wav(tmp_path / "thunder_2.wav", 24000, tone)
    _write_wav(tmp_path / "thunder_5.wav", 24000, tone)
    assert [os.path.basename(p) for p in audio_mod.find_thunder_clips(str(tmp_path))] == \
        ["thunder_2.wav", "thunder_5.wav"]
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(audio_mod, "_sound_dir_candidates", lambda d=None: [str(empty)])
    assert audio_mod.find_thunder_clips() == []


def test_load_thunder_clip_resamples_to_the_output_rate(tmp_path):
    sr_in, f0 = 24000, 100.0
    t = np.arange(sr_in // 2) / sr_in
    _write_wav(tmp_path / "t.wav", sr_in, (0.5 * np.sin(2 * np.pi * f0 * t) * 32767).astype(np.int16))
    x = audio_mod.load_thunder_clip(str(tmp_path / "t.wav"), SR)
    assert x.shape == (SR // 2, 2) and x.dtype == np.float32           # 0.5 s at 44.1 kHz, mono -> stereo
    X = np.abs(np.fft.rfft(x[:, 0]))
    f_peak = np.fft.rfftfreq(x.shape[0], 1.0 / SR)[np.argmax(X)]
    assert f_peak == pytest.approx(f0, abs=2.0)                        # pitch kept (no 24k-as-44.1k speed-up)
    mid = x[SR // 8: 3 * SR // 8, 0]
    assert np.max(np.abs(mid)) == pytest.approx(0.5, rel=0.01)          # level kept by the polyphase filter


class _ThunderStub:
    """Stands in for raingen.render.thunder_one_shots; holds until `release` is set."""

    def __init__(self, clips=None, exc=None):
        import threading

        self.calls = []
        self.clips = clips
        self.exc = exc
        self.release = threading.Event()

    def __call__(self, **kw):
        self.calls.append(kw)
        self.release.wait(timeout=30.0)
        if self.exc is not None:
            raise self.exc
        if self.clips and isinstance(self.clips[0], list):              # a sequence of results, one per call
            return self.clips[min(len(self.calls), len(self.clips)) - 1]
        return self.clips


def _no_wav_engine(monkeypatch, tmp_path, stub):
    import raingen.render

    monkeypatch.setattr(audio_mod, "_sound_dir_candidates", lambda d=None: [str(tmp_path)])
    monkeypatch.setattr(raingen.render, "thunder_one_shots", stub)
    engine = SoundEngine()
    assert engine.thunder_sounds == []
    return engine


def test_synthesized_thunder_when_no_wav_clips(monkeypatch, tmp_path):
    stub = _ThunderStub()
    engine = _no_wav_engine(monkeypatch, tmp_path, stub)
    freq, fmt, channels = pygame.mixer.get_init()
    n = int(0.75 * freq)
    t = np.arange(n) / freq
    roll = 0.9 * np.sin(2 * np.pi * 60.0 * t) * np.exp(-3.0 * t)
    stub.clips = [np.vstack([roll, roll]), np.vstack([roll, -roll])]

    engine.play_thunder(delay=0.0, volume_scale=1.0)                   # starts synthesis, queues the request
    assert engine._synth_thunder_state == "running"
    assert len(engine._pending_synth) == 1
    stub.release.set()
    engine._synth_thread.join(timeout=30.0)
    assert engine._synth_thunder_state == "ready"
    assert stub.calls == [dict(count=audio_mod.SYNTH_THUNDER_COUNT, distance_km=audio_mod.SYNTH_THUNDER_DISTANCE_KM,
                               sample_rate=int(freq), max_s=audio_mod.SYNTH_THUNDER_MAX_S, layout="stereo")]
    assert len(engine.thunder_sounds) == 2
    for snd in engine.thunder_sounds:
        assert snd.get_length() == pytest.approx(n / freq, abs=2e-3)
        back = pygame.sndarray.array(snd)
        assert back.shape[0] == n
        if fmt == -16 and channels == 2:
            assert np.max(np.abs(back.astype(np.int32))) == np.max(np.abs(np.round(roll * 32767.0)))

    assert len(engine._pending_synth) == 1
    engine.process_delayed_thunders()                                  # the queued request is played now
    assert engine._pending_synth == []
    engine.play_thunder(delay=0.0)                                     # later requests use the clips directly
    assert len(stub.calls) == 1


def test_synthesized_thunder_retries_an_empty_storm(monkeypatch, tmp_path):
    stub = _ThunderStub()
    engine = _no_wav_engine(monkeypatch, tmp_path, stub)
    freq, _fmt, _ch = pygame.mixer.get_init()
    roll = 0.5 * np.sin(2 * np.pi * 80.0 * np.arange(int(0.6 * freq)) / freq)
    stub.clips = [[], [np.vstack([roll, roll])]]                       # first storm: no usable arrival
    stub.release.set()
    engine.play_thunder(delay=0.0)
    engine._synth_thread.join(timeout=30.0)
    assert engine._synth_thunder_state == "ready"
    assert len(stub.calls) == 2
    assert "seed" not in stub.calls[0] and stub.calls[1]["seed"] == audio_mod.SYNTH_THUNDER_RETRY_SEED
    assert len(engine.thunder_sounds) == 1


def test_synthesized_thunder_failure_is_contained(monkeypatch, tmp_path, caplog):
    stub = _ThunderStub(exc=RuntimeError("boom"))
    engine = _no_wav_engine(monkeypatch, tmp_path, stub)
    with caplog.at_level(logging.WARNING, logger="rain_generator.audio"):
        engine.play_thunder(delay=0.5)
        assert len(engine._pending_synth) == 1
        stub.release.set()
        engine._synth_thread.join(timeout=30.0)
    assert engine._synth_thunder_state == "failed"
    assert engine._pending_synth == []
    assert "Synthesized thunder unavailable" in caplog.text
    engine.play_thunder(delay=0.0)                                     # no retry storm, no exception
    engine.process_delayed_thunders()
    assert len(stub.calls) == 1


@pytest.mark.slow
def test_synthesized_thunder_from_raingen(monkeypatch, tmp_path):
    """The real raingen one-shots at the app's distance and clip length, one clip instead of three."""
    monkeypatch.setattr(audio_mod, "_sound_dir_candidates", lambda d=None: [str(tmp_path)])
    monkeypatch.setattr(audio_mod, "SYNTH_THUNDER_COUNT", 1)
    engine = SoundEngine()
    assert engine.thunder_sounds == []
    engine.play_thunder(delay=0.0)
    engine._synth_thread.join(timeout=900.0)
    assert engine._synth_thunder_state == "ready"
    assert len(engine.thunder_sounds) >= 1
    freq, fmt, channels = pygame.mixer.get_init()
    pcm = pygame.sndarray.array(engine.thunder_sounds[0]).astype(np.float64)
    x = pcm.reshape(pcm.shape[0], -1).mean(axis=1) / 32768.0
    assert np.all(np.isfinite(x)) and x.size >= int(0.5 * freq)
    assert np.max(np.abs(pcm)) / 32768.0 == pytest.approx(0.9, abs=0.01)  # thunder_one_shots: peak 0.9
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / freq)
    # thunder power peaks below ~200 Hz (Holmes, Brook & Krehbiel 1971) and air absorbs
    # tens of dB per km above 4 kHz (ISO 9613-1): the rumble band must dominate the hiss band
    assert P[(f > 20) & (f < 500)].sum() > P[f > 4000].sum()


# ---------------------------------------------------------------------------
# file renderers delegated to raingen
# ---------------------------------------------------------------------------
def test_legacy_presets_map_onto_raingen_presets():
    from raingen.scene.presets import PRESETS as RAINGEN_PRESETS
    from rain_generator.presets import PRESETS

    assert set(audio_mod.LEGACY_TO_RAINGEN) == set(PRESETS)
    for legacy, key in audio_mod.LEGACY_TO_RAINGEN.items():
        assert key in RAINGEN_PRESETS
        assert audio_mod.raingen_preset_key(legacy) == key
    assert audio_mod.raingen_preset_key("tin_roof_cabin_storm") == "tin-roof-cabin-storm"  # raingen names pass
    assert audio_mod.raingen_preset_key("no-such-preset") == audio_mod.LEGACY_TO_RAINGEN["steady"]


def test_legacy_mix_mapping():
    from raingen.render import stem_gains

    default = audio_mod.legacy_mix(None)
    assert all(v == pytest.approx(0.0) for v in default.values())   # default volumes = unity stems
    assert set(stem_gains(default)) >= set(default)                   # every key is a raingen stem
    mix = audio_mod.legacy_mix({"rain": 0.0, "drops": 0.225, "wind": 0.7}, include_thunder=False)
    assert mix["rain_far"] is None and mix["ambience"] is None        # muted
    for stem in ("rain_near", "aftermath", "interior"):
        assert mix[stem] == pytest.approx(-6.0206, abs=1e-3)          # half the volume = 20 log10(0.5) dB
    assert mix["wind"] == pytest.approx(6.0206, abs=1e-3)
    assert mix["thunder"] is None
    g = stem_gains(mix)
    assert g["rain_far"] == 0.0 and g["rain_near"] == pytest.approx(0.5)


class _CountingBlocks:
    """Counts BlockRenderer.render_block calls (synthesis work) during a delegated render."""

    def __init__(self, monkeypatch):
        import raingen.render as rr

        self.calls = 0
        orig = rr.BlockRenderer.render_block

        def counted(br, *a, **kw):
            self.calls += 1
            return orig(br, *a, **kw)

        monkeypatch.setattr(rr.BlockRenderer, "render_block", counted)


def _assert_no_fallback(caplog):
    assert "legacy synthesizer" not in caplog.text, caplog.text


def test_generate_storm_audio(monkeypatch, caplog):
    from raingen.core.constants import BLOCK, SAMPLE_RATE
    from raingen.master.loudness import true_peak_db
    from rain_generator.audio import generate_storm_audio

    sr, dur = 22050, 0.25
    counter = _CountingBlocks(monkeypatch)
    with caplog.at_level(logging.WARNING):
        data = generate_storm_audio(duration=dur, preset_name="drizzle", sample_rate=sr)
    _assert_no_fallback(caplog)
    assert data.shape == (int(dur * sr), 2)
    assert data.dtype == np.int16
    assert np.any(data != 0)
    x = int16_to_float(data)
    # one synthesis pass (raingen replays its pass-1 tape for normalisation): the blocks of
    # the programme plus the decoder/limiter latency, not twice that
    assert counter.calls <= math.ceil(dur * SAMPLE_RATE / BLOCK) + 8
    # loudness contract: -18 LUFS at the default master (EBU R128 / BS.1770 measurement;
    # whole-clip K-weighted loudness because 0.25 s is shorter than one 400 ms gating block)
    assert short_programme_lufs(x, sr) == pytest.approx(audio_mod.LEGACY_TARGET_LUFS, abs=1.0)
    # true-peak ceiling -1 dBTP (BS.1770-4 Annex 2 true peak), plus 0.1 dB for 16-bit rounding
    assert true_peak_db(x, sr) <= audio_mod.LEGACY_CEILING_DBTP + 0.1
    assert abs(float(np.mean(x))) < 0.01 * float(np.sqrt(np.mean(x ** 2)))  # no DC offset


def test_delegated_render_settings():
    """The legacy wrapper's raingen render, checked without rendering: the quality tier that
    keeps CI fast, a single synthesis (pass-1 tape replayed for normalisation), and the
    loudness contract scaled by the master volume."""
    from raingen.scene.presets import preset as raingen_preset

    r = audio_mod._raingen_renderer(0.25, "cozy", 22050, None, True, 16)
    assert r.scene.render.quality == audio_mod.LEGACY_QUALITY == "mobile"
    assert r.use_tape and r.two_pass                                   # measured once, replayed from the tape
    assert (r.layout, r.dest["name"], r.sr_out, r.bit_depth) == ("stereo", "raw", 22050, 16)
    assert r.target_lufs == pytest.approx(audio_mod.LEGACY_TARGET_LUFS)
    assert r.ceiling_dbtp == pytest.approx(audio_mod.LEGACY_CEILING_DBTP)
    assert r.scene.seed == raingen_preset("window-rain-bedroom").seed   # seeded preset: reproducible files
    assert r.frames_out == int(round(0.25 * 22050))
    half = audio_mod._raingen_renderer(0.25, "cozy", 22050, {"master": 0.4}, True, 16)
    # loudness follows amplitude: half the master volume is 20 log10(0.5) = -6.02 dB
    assert r.target_lufs - half.target_lufs == pytest.approx(6.0206, abs=1e-3)
    assert audio_mod._raingen_renderer(0.25, "cozy", 22050, {"master": 0.0}, True, 16) is None
    muted = audio_mod._raingen_renderer(0.25, "steady", 22050, {"thunder": 0.0, "wind": 0.35}, False, 16)
    assert muted.mix["thunder"] is None and muted.mix["wind"] == pytest.approx(0.0)


def test_export_audio_file(tmp_path, monkeypatch, caplog):
    import soundfile as sf
    from raingen.core.constants import BLOCK, SAMPLE_RATE
    from raingen.master.loudness import true_peak_db
    from rain_generator.audio import export_audio_file

    sr, dur = 22050, 0.25
    out_file = str(tmp_path / "test_rain.wav")
    counter = _CountingBlocks(monkeypatch)
    with caplog.at_level(logging.WARNING):
        result = export_audio_file(out_file, duration=dur, preset_name="drizzle", sample_rate=sr)
    _assert_no_fallback(caplog)
    # streamed to disk in one synthesis pass (the old export rendered the programme twice)
    assert counter.calls <= math.ceil(dur * SAMPLE_RATE / BLOCK) + 8
    assert os.path.exists(result)
    assert os.path.samefile(result, out_file)                          # (8.3 short names on Windows)
    assert not os.path.exists(out_file + ".part")
    info = sf.info(result)
    assert (info.samplerate, info.channels, info.subtype) == (sr, 2, "PCM_16")
    assert info.frames == int(round(dur * sr))
    with open(result, "rb") as fh:
        head = fh.read(1 << 16)
    assert b"bext" in head and b"raingen" in head                     # BWF metadata embedded
    x, _ = sf.read(result, dtype="float64", always_2d=True)
    x = x.T
    assert short_programme_lufs(x, sr) == pytest.approx(audio_mod.LEGACY_TARGET_LUFS, abs=1.0)  # BS.1770
    assert true_peak_db(x, sr) <= audio_mod.LEGACY_CEILING_DBTP + 0.1                         # Annex 2


def test_muted_master_renders_silence_without_raingen(monkeypatch):
    import raingen.render as rr

    def forbidden(*a, **kw):
        raise AssertionError("a muted master must not start a render")

    monkeypatch.setattr(rr, "Renderer", forbidden)
    data = audio_mod.generate_storm_audio(0.5, "steady", sample_rate=8000, volumes={"master": 0.0})
    assert data.shape == (4000, 2) and data.dtype == np.int16 and not np.any(data)
    assert audio_mod.generate_storm_audio(0.0, "steady").shape == (0, 2)


# -- fallback to the legacy synthesizer --------------------------------------
class _FailingRenderer:
    def __init__(self, *a, **kw):
        raise RuntimeError("raingen exploded")


class _SilentRenderer:
    """Stands in for a raingen render that measured silence (normalisation skipped)."""

    def __init__(self, scene, seconds, *a, sample_rate=48000, target_lufs=None, **kw):
        self.n = int(round(seconds * sample_rate))
        self.target_lufs = target_lufs

    def to_array(self):
        return np.full((2, self.n), 10.0 ** (-68.0 / 20.0)), []        # -68 dBFS, like window-rain-bedroom

    def run(self, out=None, captions=False):
        path = os.path.abspath(os.fspath(out))
        with open(path, "wb") as fh:
            fh.write(b"not really a wav")
        return {"path": path, "true_peak_dbtp": -68.0, "integrated_lufs": float("-inf")}


def _assert_legacy_audio(data, n):
    assert data.shape == (n, 2) and data.dtype == np.int16
    x = int16_to_float(data)
    assert np.any(data != 0) and np.all(np.isfinite(x))
    assert np.array_equal(data[:, 0], data[:, 1])                    # the v2 synth is dual mono


@pytest.mark.parametrize("failure", ["raises", "missing", "silent"])
def test_generate_falls_back_to_legacy_synth(monkeypatch, caplog, failure):
    import raingen.render as rr

    if failure == "raises":
        monkeypatch.setattr(rr, "Renderer", _FailingRenderer)
    elif failure == "missing":
        monkeypatch.setitem(sys.modules, "raingen.render", None)      # import raingen.render -> ImportError
    else:
        monkeypatch.setattr(rr, "Renderer", _SilentRenderer)
    with caplog.at_level(logging.WARNING, logger="rain_generator.audio"):
        data = audio_mod.generate_storm_audio(0.5, "thunderstorm", sample_rate=22050)
    assert "using the legacy synthesizer" in caplog.text
    _assert_legacy_audio(data, 11025)


@pytest.mark.parametrize("failure", ["raises", "silent"])
def test_export_falls_back_to_legacy_synth(monkeypatch, tmp_path, caplog, failure):
    import raingen.render as rr
    import scipy.io.wavfile as wavfile

    monkeypatch.setattr(rr, "Renderer", _FailingRenderer if failure == "raises" else _SilentRenderer)
    out_file = str(tmp_path / "fallback.wav")
    with caplog.at_level(logging.WARNING, logger="rain_generator.audio"):
        result = audio_mod.export_audio_file(out_file, duration=0.5, preset_name="cozy", sample_rate=22050)
    assert "using the legacy synthesizer" in caplog.text
    assert os.path.samefile(result, out_file)
    rate, data = wavfile.read(result)                                  # a real WAV replaced the silent one
    assert rate == 22050
    _assert_legacy_audio(data, 11025)


def test_silence_guard_is_relative_to_the_target():
    # a normalised programme peaks above its loudness target; raw silence sits far below it
    assert audio_mod._is_silent(-68.0, -18.0)
    assert audio_mod._is_silent(float("-inf"), -18.0)
    assert not audio_mod._is_silent(-12.0, -18.0)
    assert not audio_mod._is_silent(-70.0, -76.0)                     # master at 0.001: quiet, not silent
    assert not audio_mod._is_silent(None, -18.0)


# -- slow: level, determinism and every legacy preset through raingen ---------
@pytest.mark.slow
def test_generate_storm_audio_master_volume_and_determinism(caplog):
    from raingen.master.loudness import integrated_loudness

    sr, dur = 22050, 0.6
    with caplog.at_level(logging.WARNING):
        a = audio_mod.generate_storm_audio(dur, "drizzle", sample_rate=sr)
        b = audio_mod.generate_storm_audio(dur, "drizzle", sample_rate=sr, volumes={"master": 0.4})
        c = audio_mod.generate_storm_audio(dur, "drizzle", sample_rate=sr)
    _assert_no_fallback(caplog)
    assert np.array_equal(a, c)                                         # seeded raingen preset: bit-identical
    la = integrated_loudness(int16_to_float(a), sr)
    lb = integrated_loudness(int16_to_float(b), sr)
    assert la == pytest.approx(audio_mod.LEGACY_TARGET_LUFS, abs=1.0)    # BS.1770 gated, >= 400 ms
    assert la - lb == pytest.approx(20.0 * math.log10(0.8 / 0.4), abs=0.5)  # half the master volume = -6.02 dB


@pytest.mark.slow
@pytest.mark.parametrize("preset", [
    "drizzle",
    "steady",
    "thunderstorm",
    "gale",
    pytest.param("cozy", marks=pytest.mark.xfail(
        reason="raingen window-rain-bedroom renders at about -67 dBTP (interior scenes nearly silent); "
               "the legacy wrapper falls back to the old synthesizer", strict=False)),
])
def test_every_legacy_preset_renders_through_raingen(preset, caplog):
    from raingen.master.loudness import integrated_loudness, true_peak_db

    sr = 22050
    with caplog.at_level(logging.WARNING):
        data = audio_mod.generate_storm_audio(0.5, preset, sample_rate=sr)
    x = int16_to_float(data)
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) > 10.0 ** (-40.0 / 20.0)  # audible either way
    _assert_no_fallback(caplog)
    assert integrated_loudness(x, sr) == pytest.approx(audio_mod.LEGACY_TARGET_LUFS, abs=1.0)  # BS.1770
    assert true_peak_db(x, sr) <= audio_mod.LEGACY_CEILING_DBTP + 0.1                         # Annex 2
