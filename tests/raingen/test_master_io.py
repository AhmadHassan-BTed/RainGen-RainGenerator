"""Mastering and file output: loudness, true peak, limiter, dither, destinations, WAV/RF64/BWF,
FLAC, UCS naming and captions, checked against the published standards they implement.

References used for the numbers below:
  BS.1770-4/5  ITU-R BS.1770 (K-weighting table, gating, channel weights, Annex 2 true peak)
  Tech 3341    EBU loudness-meter conformance signals (tolerances +-0.1 LU, TP +0.2/-0.4 dB)
  Tech 3342    EBU loudness range (LRA) conformance signals (+-1 LU)
  Tech 3285    EBU BWF bext chunk v2 (field sizes and offsets)
  Tech 3306    EBU RF64 (ds64 chunk)
  LWV 1992     Lipshitz, Wannamaker & Vanderkooy, "Quantization and dither: a theoretical survey"
  UCS 8.2.1    Universal Category System full list (CatID / Category / SubCategory)
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import re
import struct
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import ellip, firwin, resample_poly, sosfiltfilt, sosfreqz

from raingen.core.rng import RngFactory
from raingen.core.types import EventMarker
from raingen.io import captions as cap
from raingen.io.ucs import UCS_CATIDS, ucs_catid, ucs_filename, ucs_fxname, ucs_metadata
from raingen.io.wav import (
    FlacWriter,
    StreamingWavWriter,
    build_bext,
    build_ixml,
    open_writer,
    timecode_to_frames,
    timecode_to_seconds,
)
from raingen.master.destinations import (
    DESTINATIONS,
    get_destination,
    normalization_gain_db,
)
from raingen.master.dither import Ditherer, lsb, quantize, tpdf_dither
from raingen.master.limiter import TruePeakLimiter, limit
from raingen.master.loudness import (
    LoudnessMeter,
    PolyphaseOversampler,
    channel_weights,
    integrated_loudness,
    k_weighting_sos,
    true_peak_db,
    weight_for_position,
)

SR = 48_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sine(dbfs: float, secs: float, f: float = 1000.0, ch: int = 2, phase: float = 0.0, sr: int = SR,
         fade_s: float = 0.0) -> np.ndarray:
    """Sine of peak level `dbfs` (Tech 3341 convention: 0 dBFS = full-scale sine), (ch, n)."""
    n = round(secs * sr)
    t = np.arange(n) / sr
    x = 10.0 ** (dbfs / 20.0) * np.sin(2.0 * np.pi * f * t + phase)
    if fade_s > 0.0:
        r = round(fade_s * sr)
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
        x[:r] *= ramp
        x[-r:] *= ramp[::-1]
    return np.tile(x, (ch, 1))


def seq(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts, axis=1)


def meter_of(x: np.ndarray, layout: str = "stereo", sr: int = SR, block: int = 48_000,
             true_peak: bool = False) -> LoudnessMeter:
    x = np.atleast_2d(x)
    m = LoudnessMeter(sr, x.shape[0], layout, true_peak=true_peak)
    for s in range(0, x.shape[1], block):
        m.process(x[:, s:s + block])
    return m


def db_power_mean(*levels_and_secs) -> float:
    """Loudness of concatenated segments (level, seconds) assuming each reads its own level."""
    e = sum(10.0 ** (lv / 10.0) * s for lv, s in levels_and_secs)
    return 10.0 * math.log10(e / sum(s for _, s in levels_and_secs))


def reference_tp_db(x: np.ndarray) -> float:
    """Independent near-ideal true-peak reference: 32x polyphase, 64 taps per phase, Kaiser beta 12."""
    h = firwin(32 * 64 + 1, 1.0 / 32.0, window=("kaiser", 12.0))
    up = resample_poly(np.atleast_2d(x), 32, 1, axis=1, window=h)
    return 20.0 * math.log10(max(np.max(np.abs(up)), np.max(np.abs(x))))


def band_limit_20k(x: np.ndarray) -> np.ndarray:
    """Zero-phase 12th-order elliptic low-pass at 20 kHz (what every DAC/codec reconstructs alike)."""
    return sosfiltfilt(ellip(12, 0.01, 90, 20_000, fs=SR, output="sos"), x, axis=-1)


# ---------------------------------------------------------------------------
# K-weighting (BS.1770-4 Annex 1, Tables 1 and 2)
# ---------------------------------------------------------------------------
BS1770_STAGE1 = [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585]
BS1770_STAGE2 = [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]


def test_k_weighting_48k_matches_bs1770_table():
    sos = k_weighting_sos(48_000)
    # BS.1770-4 Annex 1 Tables 1 and 2 publish 14 decimals
    np.testing.assert_allclose(sos[0], BS1770_STAGE1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sos[1], BS1770_STAGE2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fs", [44_100, 96_000, 192_000])
def test_k_weighting_generalises_to_other_rates(fs):
    f = np.array([25.0, 50.0, 100.0, 500.0, 997.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0])
    _, h48 = sosfreqz(k_weighting_sos(48_000), worN=f, fs=48_000)
    _, h = sosfreqz(k_weighting_sos(fs), worN=f, fs=fs)
    d = 20 * np.log10(np.abs(h)) - 20 * np.log10(np.abs(h48))
    # same analog prototype; the 48 kHz bilinear design itself is warped by <= ~0.03 dB below 16 kHz
    assert np.max(np.abs(d)) < 0.05
    g997 = 20 * np.log10(abs(sosfreqz(k_weighting_sos(fs), worN=[997.0], fs=fs)[1][0]))
    # the -0.691 dB constant in BS.1770 cancels the K-filter gain at 997 Hz
    assert abs(g997 - 0.691) < 0.03
    # pole radii < 1: both stages stable at every rate
    for sec in k_weighting_sos(fs):
        assert np.all(np.abs(np.roots(sec[3:])) < 1.0)


@pytest.mark.parametrize("fs", [44_100, 48_000, 96_000])
def test_1k_sine_at_minus23_reads_minus23_at_any_rate(fs):
    x = sine(-23.0, 10.0, sr=fs)
    # EBU Tech 3341 case 1: stereo 1 kHz at -23 dBFS reads -23.0 +-0.1 LUFS
    assert integrated_loudness(x, fs) == pytest.approx(-23.0, abs=0.1)


# ---------------------------------------------------------------------------
# EBU Tech 3341 conformance cases (integrated, momentary, short-term)
# ---------------------------------------------------------------------------
def test_tech3341_case1_and_2_steady_tones():
    m = meter_of(sine(-23.0, 20.0))
    # Tech 3341 case 1: M, S and I all -23.0 +-0.1
    assert m.integrated() == pytest.approx(-23.0, abs=0.1)
    assert m.momentary() == pytest.approx(-23.0, abs=0.1)
    assert m.short_term() == pytest.approx(-23.0, abs=0.1)
    assert not m.integrated_is_provisional
    # Tech 3341 case 2: -33 dBFS -> -33.0 +-0.1
    assert integrated_loudness(sine(-33.0, 20.0), SR) == pytest.approx(-33.0, abs=0.1)


def test_tech3341_case3_and_4_gating():
    x3 = seq(sine(-36, 10), sine(-23, 60), sine(-36, 10))
    # Tech 3341 case 3: relative gate removes the -36 parts -> -23.0 +-0.1
    assert integrated_loudness(x3, SR) == pytest.approx(-23.0, abs=0.1)
    x4 = seq(sine(-72, 10), sine(-36, 10), sine(-23, 60), sine(-36, 10), sine(-72, 10))
    # Tech 3341 case 4: absolute + relative gates -> -23.0 +-0.1
    assert integrated_loudness(x4, SR) == pytest.approx(-23.0, abs=0.1)


def test_tech3341_case5_relative_gate_keeps_both_levels():
    x = seq(sine(-26, 20), sine(-20, 20.1), sine(-26, 20))
    # Tech 3341 case 5: -26/-20/-26 dBFS -> -23.0 +-0.1
    assert integrated_loudness(x, SR) == pytest.approx(-23.0, abs=0.1)


def test_tech3341_case6_surround_weights_and_lfe_excluded():
    # 5.1 in WAV order L R C LFE Ls Rs; Tech 3341 case 6 (5.0): L,R -28, C -24, Ls,Rs -30 dBFS
    levels = [-28, -28, -24, None, -30, -30]
    x = np.concatenate([sine(lv, 20, ch=1) if lv is not None else np.zeros((1, 20 * SR)) for lv in levels])
    assert integrated_loudness(x, SR, "5.1") == pytest.approx(-23.0, abs=0.1)     # Tech 3341 case 6
    x[3] = sine(0.0, 20, f=50.0, ch=1)[0]                                       # full-scale LFE
    assert integrated_loudness(x, SR, "5.1") == pytest.approx(-23.0, abs=0.1)     # BS.1770: LFE not measured


def test_surround_weight_is_plus_1p5_db():
    tone = sine(-20, 5, ch=1)[0]
    front = np.zeros((6, tone.size))
    front[0] = tone
    surround = np.zeros((6, tone.size))
    surround[4] = tone
    d = integrated_loudness(surround, SR, "5.1") - integrated_loudness(front, SR, "5.1")
    # BS.1770-4 Table 3: G = 1.41 (+1.5 dB) for 60 <= |az| <= 120, |el| < 30
    assert d == pytest.approx(10 * math.log10(1.41), abs=0.01)


def test_channel_weights_match_speaker_positions():
    from raingen.spatial.decoders import SPEAKER_LAYOUTS

    # BS.1770-5 Table 3 position rule, applied to the decoder's own loudspeaker azimuths
    for lay, spk in SPEAKER_LAYOUTS.items():
        expect = [weight_for_position(az, el) for _, az, el in spk]
        np.testing.assert_allclose(channel_weights(lay, len(spk)), expect, err_msg=lay)
    assert weight_for_position(110.0) == 1.41
    assert weight_for_position(-90.0) == 1.41
    assert weight_for_position(135.0) == 1.0             # rear surround: outside 60..120
    assert weight_for_position(90.0, 45.0) == 1.0        # height: |el| >= 30
    assert weight_for_position(None) == 0.0              # LFE
    np.testing.assert_allclose(channel_weights("foa", 4), [1, 0, 0, 0])  # W only for AmbiX


def test_absolute_gate_minus70():
    # BS.1770-4 sec. 5.1: blocks below -70 LKFS are discarded -> nothing left -> -inf
    assert integrated_loudness(sine(-75, 5), SR) == float("-inf")
    assert integrated_loudness(sine(-65, 5), SR) == pytest.approx(-65.0, abs=0.1)


def test_relative_gate_minus10_lu():
    # -20 and -29: both within 10 LU of their energy mean -> power mean of both (BS.1770-4 sec. 5.1)
    both = integrated_loudness(seq(sine(-20, 20), sine(-29, 20)), SR)
    assert both == pytest.approx(db_power_mean((-20, 20), (-29, 20)), abs=0.1)
    # -20 and -35: the -35 part is > 10 LU below the gated mean and is dropped
    gated = integrated_loudness(seq(sine(-20, 20), sine(-35, 20)), SR)
    assert gated == pytest.approx(-20.0, abs=0.1)


def test_momentary_and_short_term_windows():
    m = LoudnessMeter(SR, 2, true_peak=False)
    m.process(sine(-40, 5))
    m.process(sine(-20, 0.2))
    # momentary = mean square of the last 400 ms (BS.1770 / Tech 3341 sec. 2.3)
    assert m.momentary() == pytest.approx(db_power_mean((-20, 0.2), (-40, 0.2)), abs=0.15)
    m.process(sine(-20, 0.2))
    assert m.momentary() == pytest.approx(-20.0, abs=0.1)
    m.process(sine(-20, 1.1))
    # short-term = last 3 s: 1.5 s at -20, 1.5 s at -40
    assert m.short_term() == pytest.approx(db_power_mean((-20, 1.5), (-40, 1.5)), abs=0.15)
    m.process(sine(-20, 1.5))
    assert m.short_term() == pytest.approx(-20.0, abs=0.1)
    assert m.max_momentary() == pytest.approx(-20.0, abs=0.1)
    assert m.max_short_term() == pytest.approx(-20.0, abs=0.1)


def test_momentary_weights_partial_subblock_by_length():
    m = LoudnessMeter(SR, 2, true_peak=False)
    m.process(sine(-20, 0.1))
    m.process(sine(-40, 0.01))
    # 100 ms at -20 plus 10 ms at -40: energy mean over 110 ms, not a 50/50 average
    assert m.momentary() == pytest.approx(db_power_mean((-20, 0.1), (-40, 0.01)), abs=0.15)


def test_meter_is_block_size_independent():
    r = np.random.default_rng(5)
    x = r.standard_normal((2, 4 * SR)) * 0.1 * np.linspace(0.2, 1.0, 4 * SR)
    ref = meter_of(x, block=4 * SR, true_peak=True)
    for blk in (1, 333, 4800):
        if blk == 1:
            m = meter_of(x[:, :2400], block=1, true_peak=True)
            r1 = meter_of(x[:, :2400], block=SR, true_peak=True)
            assert m.integrated() == pytest.approx(r1.integrated(), abs=1e-9)
            assert m.true_peak_db() == pytest.approx(r1.true_peak_db(), abs=1e-9)
            continue
        m = meter_of(x, block=blk, true_peak=True)
        assert m.integrated() == pytest.approx(ref.integrated(), abs=1e-9)   # streaming == offline
        assert m.lra() == pytest.approx(ref.lra(), abs=1e-9)
        assert m.true_peak_db() == pytest.approx(ref.true_peak_db(), abs=1e-9)


# ---------------------------------------------------------------------------
# short programmes (< one 400 ms gating block)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("secs", [0.05, 0.1, 0.25, 0.39])
def test_short_clip_reads_finite_provisional_loudness(secs):
    m = meter_of(sine(-23, secs))
    assert m.integrated_is_provisional
    assert m.integrated(provisional=False) == float("-inf")        # strict BS.1770: no gating block yet
    # ungated mean square of a -23 dBFS stereo 1 kHz tone is -23 LUFS; the K-filter start-up
    # transient and partial cycles cost at most ~0.3 dB on 50 ms
    assert m.integrated() == pytest.approx(-23.0, abs=0.3)
    assert integrated_loudness(sine(-23, secs), SR) == pytest.approx(-23.0, abs=0.3)
    assert m.summary()["integrated_provisional"] is True


def test_short_clip_reading_is_continuous_across_400ms():
    a = integrated_loudness(sine(-30, 0.39), SR)
    b = integrated_loudness(sine(-30, 0.41), SR)
    assert not meter_of(sine(-30, 0.41)).integrated_is_provisional
    assert abs(a - b) < 0.2          # provisional and gated readings agree on a steady tone


def test_short_silence_stays_silent():
    assert integrated_loudness(np.zeros((2, 4800)), SR) == float("-inf")
    assert integrated_loudness(sine(-90, 0.2), SR) == float("-inf")   # below the -70 LUFS absolute gate
    m = LoudnessMeter(SR, 2)
    assert m.integrated() == float("-inf") and not m.integrated_is_provisional


# ---------------------------------------------------------------------------
# loudness range (EBU Tech 3342)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("levels,expected", [
    ((-20, -30), 10.0),                     # Tech 3342 case 1
    ((-20, -15), 5.0),                      # Tech 3342 case 2
    ((-40, -20), 20.0),                     # Tech 3342 case 3
    ((-50, -35, -20, -35, -50), 15.0),      # Tech 3342 case 4
])
def test_tech3342_lra(levels, expected):
    m = meter_of(seq(*[sine(lv, 20) for lv in levels]))
    assert m.lra() == pytest.approx(expected, abs=1.0)     # Tech 3342 tolerance +-1 LU


def test_lra_of_steady_tone_is_zero():
    assert meter_of(sine(-23, 10)).lra() == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# true peak (BS.1770 Annex 2; Tech 3341 cases 15-19)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f_div,phase_deg,sample_peak_db", [
    (4, 0.0, 0.0),          # samples hit the crests
    (4, 45.0, -3.01),       # Tech 3341-style: fs/4 at 45 deg, samples at 0.707 of the crest
    (6, 60.0, -1.25),       # fs/6 at 60 deg: samples at sin 60 = 0.866
    (8, 67.5, -0.69),       # fs/8 at 67.5 deg: samples at sin 67.5 = 0.924
])
def test_true_peak_of_offset_sines(f_div, phase_deg, sample_peak_db):
    x = sine(-6.0, 1.0, f=SR / f_div, ch=1, phase=math.radians(phase_deg), fade_s=0.01)
    sp = 20 * math.log10(np.max(np.abs(x)))
    assert sp == pytest.approx(-6.0 + sample_peak_db, abs=0.02)       # geometry of the test signal
    tp = true_peak_db(x, SR)
    # Tech 3341 true-peak tolerance: +0.2 / -0.4 dB around the -6.0 dBTP crest
    assert -6.4 <= tp <= -5.8
    assert tp == pytest.approx(-6.0, abs=0.05)                          # 8x detector, measured accuracy
    m = meter_of(x, layout="mono", true_peak=True)
    assert m.true_peak_db() == pytest.approx(tp, abs=0.01)


def test_fs4_45deg_reads_3db_over_sample_peak():
    x = sine(3.01, 1.0, f=SR / 4, ch=2, phase=math.pi / 4, fade_s=0.01)
    m = meter_of(x, true_peak=True)
    # Tech 3341 case 19 idea: 0 dBFS sample peak, +3 dBTP true peak (+0.2/-0.4)
    assert m.sample_peak_db() == pytest.approx(0.0, abs=0.02)
    assert m.true_peak_db() == pytest.approx(3.01, abs=0.05)
    assert m.true_peak_db() - m.sample_peak_db() == pytest.approx(3.01, abs=0.1)


def test_true_peak_accuracy_up_to_20khz():
    n = 4800
    def hann(k):
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
    k = np.arange(n)
    fine = np.arange(n // 2 - 200, n // 2 + 200, 1.0 / 256.0)              # crest region, 256x grid
    rng = np.random.default_rng(2)
    for f in (1000.0, 5000.0, 10000.0, 15000.0, 18000.0, 20000.0):
        for ph in rng.uniform(0, np.pi, 6):
            x = np.sin(2 * np.pi * f * k / SR + ph) * hann(k)
            # analytic crest of the (practically band-limited) Hann-windowed burst
            peak_true = 20 * math.log10(np.max(np.abs(np.sin(2 * np.pi * f * fine / SR + ph) * hann(fine))))
            tp = true_peak_db(x, SR)
            # plan sec. 10: an 8x detector under-reads by at most ~0.17 dB; it must never over-read
            assert peak_true - 0.17 <= tp <= peak_true + 0.01, (f, ph, tp - peak_true)


def test_true_peak_includes_interpolator_tail():
    # a crest right at the end of the programme: the meter must not miss the last samples
    x = sine(-1.0, 0.05, f=SR / 4, ch=1, phase=math.pi / 4)
    m = meter_of(x, layout="mono", true_peak=True)
    assert m.true_peak_db() == pytest.approx(true_peak_db(x, SR), abs=1e-9)
    before = m.true_peak_db()
    assert m.true_peak_db() == before                                       # reading has no side effects


def _rain_like(n: int, seed: int, level: float = 0.02, clicks: int = 60, loud_from: int = -1,
               channels: int = 2) -> np.ndarray:
    """Sparse decaying clicks on a quiet noise floor, optionally a dense loud stretch: (channels, n)."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((channels, n)) * level * 0.05
    for i in r.integers(0, n - 64, clicks):
        x[:, i:i + 48] += r.standard_normal((channels, 48)) * np.exp(-np.arange(48) / 6.0) * level * r.uniform(1, 30)
    if loud_from >= 0:
        x[:, loud_from:] += r.standard_normal((channels, n - loud_from)) * 0.4
    return x


@pytest.mark.parametrize("factor,tpp", [(8, None), (4, None), (8, 11)])
def test_sparse_peak_readers_match_full_interpolation(factor, tpp):
    x_all = _rain_like(6000, 1, loud_from=4500)
    for block in (1, 7, 333, 1024, 3000):                 # 3000: block-Toeplitz path, not a multiple of 16
        # tiny blocks on an excerpt that still crosses into the dense loud stretch at 4500
        x = x_all[:, 4200:4800] if block < 100 else x_all
        for floor in (0.0, 0.05, 0.3, 2.0):
            full = PolyphaseOversampler(factor, 2, tpp)
            env = PolyphaseOversampler(factor, 2, tpp)
            chp = PolyphaseOversampler(factor, 2, tpp)
            for s in range(0, x.shape[1], block):
                blk = x[:, s:s + block]
                up = np.abs(full.process(blk)).reshape(2, blk.shape[1], factor)
                ref_env = up.max(axis=(0, 2))
                got = env.peak_envelope(blk, floor)
                over = ref_env > floor
                # exact wherever the peak can exceed the floor (BLAS row subsets may differ by an ulp)
                np.testing.assert_allclose(got[over], ref_env[over], rtol=1e-13, atol=0)
                assert np.all(got[~over] <= floor)
                ref_ch = up.max(axis=(1, 2))
                got_ch = chp.channel_peaks(blk, floor)
                np.testing.assert_allclose(got_ch[ref_ch > floor], ref_ch[ref_ch > floor], rtol=1e-13, atol=0)
                assert np.all(got_ch[ref_ch <= floor] <= floor)
            # all three advanced the stream identically: the next outputs agree
            tail = full.flush()
            np.testing.assert_array_equal(env.flush(), tail)
            np.testing.assert_array_equal(chp.flush(), tail)


@pytest.mark.parametrize("factor,tpp", [(8, None), (4, None), (8, 11)])
def test_block_toeplitz_interpolation_equals_direct(factor, tpp):
    x = np.random.default_rng(6).standard_normal((3, 9001))
    direct = PolyphaseOversampler(factor, 3, tpp)
    ref = np.concatenate([direct.process(x[:, s:s + 1000]) for s in range(0, 9001, 1000)], axis=1)
    toep = PolyphaseOversampler(factor, 3, tpp)
    assert 9001 >= toep._TOEPLITZ_MIN_N
    got = np.concatenate([toep.process(x[:, :5003]), toep.process(x[:, 5003:])], axis=1)
    # same FIR, summed in a different order by BLAS: equal to rounding
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(toep.flush(), direct.flush())          # same stream state after


def test_l1_bound_is_rigorous_and_tight():
    os_ = PolyphaseOversampler(8, 1)
    # Kaiser-sinc 8x/48: the half-sample phases have the largest L1 norm, ~2.55 (+8.1 dB)
    assert os_.l1 == pytest.approx(2.5467, abs=1e-3)
    p = int(np.argmax(np.sum(np.abs(os_.phases), axis=1)))
    worst = np.sign(os_.phases[p][::-1])[None, :] * 0.5                 # sign pattern of that phase
    y = np.abs(os_.process(np.concatenate([worst, np.zeros((1, 64))], axis=1)))
    # the bound is attained by the matching sign pattern (so it cannot be tightened) ...
    assert np.max(y) == pytest.approx(0.5 * os_.l1, rel=1e-6)
    r = np.random.default_rng(4)
    for _ in range(20):
        x = r.standard_normal((1, 2000)) * r.uniform(0.01, 3.0)
        o = PolyphaseOversampler(8, 1)
        up = np.abs(o.process(x)).reshape(2000, 8).max(axis=1)
        wm = o._window_max(np.concatenate([np.zeros((1, 47)), x], axis=1), 2000)[0]
        assert np.all(up <= wm * o.l1)                                   # ... and never exceeded


@pytest.mark.parametrize("name", ["isp_worst", "rain_clicks", "silence_then_burst", "noise_plus20"])
def test_limiter_sparse_detection_changes_nothing(name):
    x = _worst_cases()[name] * 0.6
    y, _ = _run_limiter(x, -1.0, 1024)
    lim = TruePeakLimiter(SR, 2, -1.0)
    lim._os.l1 = 1e12                                   # every non-silent sample becomes a candidate
    parts = [lim.process(x[:, s:s + 1024]) for s in range(0, x.shape[1], 1024)]
    ref = np.concatenate([*parts, lim.flush()], axis=1)
    # same gain decisions; only BLAS row-subset rounding (~1e-16) may differ
    np.testing.assert_allclose(y, ref, rtol=0, atol=1e-12)


def test_limiter_returns_to_bit_exact_after_release():
    quiet = _rain_like(3 * SR, 2, level=0.01)
    x = quiet.copy()
    x[:, 4800:9600] += sine(6.0, 0.1)                       # +6 dBTP burst at 0.1-0.2 s
    y, lim = _run_limiter(x, -1.0, 1024)
    y = y[:, lim.latency:lim.latency + x.shape[1]]
    assert lim.max_gain_reduction_db > 6.0
    # 80 ms release decays below 1e-9 Np (~1.7 s): from 2.2 s on the limiter is idle again and
    # passes the input bit-exactly; just after the burst it is still reducing gain
    np.testing.assert_array_equal(y[:, int(2.2 * SR):], x[:, int(2.2 * SR):])
    assert np.max(np.abs(y[:, 10_000:11_000] - x[:, 10_000:11_000])) > 0.0


def test_peak_detection_is_faster_on_sparse_material():
    # quiet crackly 7.1.4 rain bed (~-30 dBTP): the 8x interpolation dominates the full path here
    x = _rain_like(SR, 3, level=0.01, clicks=60, channels=12)

    def run(l1):
        best = float("inf")
        for _ in range(3):
            lim = TruePeakLimiter(SR, 12, -1.0)
            m = LoudnessMeter(SR, 12, "7.1.4")
            if l1 is not None:
                lim._os.l1 = m._tp.l1 = l1
            t0 = time.perf_counter()
            for s in range(0, x.shape[1], 1024):
                m.process(lim.process(x[:, s:s + 1024]))
            best = min(best, time.perf_counter() - t0)
        return best, m.true_peak_db()

    fast, tp_fast = run(None)
    full, tp_full = run(1e12)
    assert tp_fast == pytest.approx(tp_full, abs=1e-9)                 # same reading
    # limiter + true-peak meter: measured ~4.4x (690 -> 156 ms per s of audio); 2x leaves room for load
    assert full / fast > 2.0, (full, fast)


# ---------------------------------------------------------------------------
# true-peak limiter
# ---------------------------------------------------------------------------
def _worst_cases() -> dict:
    n = SR // 2
    t = np.arange(n) / SR
    r = np.random.default_rng(3)
    cases = {
        "sine_fs4_45": np.tile(np.sin(2 * np.pi * SR / 4 * t + np.pi / 4) * 2.0, (2, 1)),
        "noise_plus20": r.standard_normal((2, n)) * 3.0,
        "impulses": np.zeros((2, n)),
        "square_fs4": np.tile(np.where((np.arange(n) // 2) % 2 == 0, 1.5, -1.5), (2, 1)),
        "nyquist": np.tile(np.where(np.arange(n) % 2 == 0, 2.0, -2.0), (2, 1)),
    }
    cases["impulses"][:, ::4800] = 4.0
    k = np.arange(-16, 16)
    isp = np.sign(np.sinc(k + 0.5))                    # sign pattern maximising the inter-sample peak
    w = np.zeros(n)
    for s in range(1000, n - 100, 3000):
        w[s:s + 32] = isp
    cases["isp_worst"] = np.tile(w, (2, 1))
    burst = np.zeros((2, n))
    burst[:, 12000:16800] = r.standard_normal((2, 4800)) * 4.0
    cases["silence_then_burst"] = burst
    clicks = np.zeros((2, n))
    for i in r.integers(0, n - 50, 200):
        clicks[:, i:i + 20] += r.standard_normal(20) * np.exp(-np.arange(20) / 3.0) * 3.0
    cases["rain_clicks"] = clicks
    return cases


def _run_limiter(x: np.ndarray, ceiling: float, block: int) -> tuple[np.ndarray, TruePeakLimiter]:
    lim = TruePeakLimiter(SR, x.shape[0], ceiling)
    parts = [lim.process(x[:, s:s + block]) for s in range(0, x.shape[1], block)]
    parts.append(lim.flush())
    return np.concatenate(parts, axis=1), lim


@pytest.mark.parametrize("ceiling", [-1.0, pytest.param(-2.0, marks=pytest.mark.slow)])
def test_limiter_never_exceeds_ceiling_on_worst_cases(ceiling):
    for name, x in _worst_cases().items():
        for block in (1024, 333):
            y, _ = _run_limiter(x, ceiling, block)
            assert np.all(np.isfinite(y))
            tp = true_peak_db(y, SR)
            # BS.1770 Annex 2 true peak on our 8x meter: never above the ceiling (the limiter
            # detects with the same interpolator); 0.05 dB safety below it
            assert tp <= ceiling + 1e-6, (name, block, tp)
            assert tp >= ceiling - 0.06, (name, block, tp)                   # and not over-limited
        # full-band synthetic worst cases (energy right up to Nyquist): their ideal reconstruction
        # depends on the interpolator's length; a near-ideal 32x reference stays within 0.5 dB
        assert reference_tp_db(y) <= ceiling + 0.5, name


@pytest.mark.parametrize("name", ["noise_plus20", "silence_then_burst", "rain_clicks", "sine_fs4_45"])
def test_limiter_holds_ceiling_on_near_ideal_reconstruction(name):
    x = band_limit_20k(_worst_cases()[name])
    t = np.arange(SR // 2) / SR
    y, _ = _run_limiter(x, -1.0, 1024)
    # content below 20 kHz (all real programme): a near-ideal 32x reconstruction agrees with the meter
    assert reference_tp_db(y) <= -1.0 + 0.05, name
    tone = np.tile(np.sin(2 * np.pi * 19_000 * t + 0.3) * 3.0, (2, 1))      # 19 kHz, +9.5 dBTP in
    y, _ = _run_limiter(tone, -1.0, 1024)
    assert reference_tp_db(y) <= -1.0 + 0.05


def test_limiter_latency_and_transparency_below_ceiling():
    r = np.random.default_rng(0)
    x = r.standard_normal((2, 10_000)) * 0.05                  # ~-17 dBTP: limiter idle
    y, lim = _run_limiter(x, -1.0, 700)
    assert lim.latency == lim._L + math.ceil(lim._os.delay)
    assert lim.latency == round(0.005 * SR) + 24                # 5 ms look-ahead + 8x/48-tap interpolator delay
    np.testing.assert_array_equal(y[:, :lim.latency], 0.0)
    np.testing.assert_array_equal(y[:, lim.latency:lim.latency + x.shape[1]], x)   # a pure delay, bit-exact
    np.testing.assert_array_equal(limit(x, SR), x)               # offline helper is latency-compensated
    assert lim.max_gain_reduction_db == 0.0


def test_limiter_block_size_independent_and_any_block_length():
    x = _worst_cases()["noise_plus20"][:, :3000]
    ref, _ = _run_limiter(x, -1.0, 3000)
    for block in (1, 7, 256, 1000):
        y, _ = _run_limiter(x, -1.0, block)
        np.testing.assert_allclose(y, ref, rtol=0, atol=1e-12)


def test_limiter_gain_reduction_and_release():
    x = sine(6.0, 0.5, ch=2, fade_s=0.0)
    x = seq(x, sine(-20.0, 1.0))                                # loud tone, then quiet tone
    y, lim = _run_limiter(x, -1.0, 1024)
    # required reduction = 6.0 dBTP in - (-1 - 0.05 safety) = 7.05 dB
    assert lim.max_gain_reduction_db == pytest.approx(7.05, abs=0.1)
    y = y[:, lim.latency:lim.latency + x.shape[1]]
    quiet_in = x[:, int(1.0 * SR):]
    quiet_out = y[:, int(1.0 * SR):]
    # 80 ms exponential release: 0.5 s after the loud part the gain is back within 0.05 dB of unity
    g = 20 * math.log10(np.sqrt(np.mean(quiet_out ** 2)) / np.sqrt(np.mean(quiet_in ** 2)))
    assert g == pytest.approx(0.0, abs=0.05)


def test_limiter_is_clean_on_steady_tone():
    x = sine(4.0, 1.0, f=997.0, ch=1, fade_s=0.0)
    y = limit(x, SR, -1.0)
    seg = y[0, SR // 4:]
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    f = np.fft.rfftfreq(seg.size, 1 / SR)
    fund = spec[np.argmin(np.abs(f - 997.0))]
    harmonics = [spec[np.argmin(np.abs(f - k * 997.0))] for k in range(2, 10)]
    thd = math.sqrt(sum(h * h for h in harmonics)) / fund
    # steady gain on a steady tone: harmonic distortion far below audibility (< -60 dB)
    assert 20 * math.log10(thd) < -60.0
    assert true_peak_db(y, SR) <= -1.0 + 1e-6


@pytest.mark.parametrize("fs", [44_100, 96_000])
def test_limiter_meter_and_helper_agree_at_other_rates(fs):
    for name in ("noise_plus20", "rain_clicks", "sine_fs4_45"):
        x = resample_poly(_worst_cases()[name], fs // 300, 160, axis=1)      # same content at fs
        lim = TruePeakLimiter(fs, 2, -1.0)
        parts = [lim.process(x[:, s:s + 1024]) for s in range(0, x.shape[1], 1024)]
        y = np.concatenate([*parts, lim.flush()], axis=1)
        m = meter_of(y, sr=fs, true_peak=True)
        # one interpolator (8x, 48 taps) for limiter, meter and helper at every rate (plan sec. 10)
        assert true_peak_db(y, fs) == pytest.approx(m.true_peak_db(), abs=1e-9), name
        assert m.true_peak_db() <= -1.0 + 1e-6, name                      # ceiling holds at fs too
        assert m.true_peak_db() >= -1.06, name                             # 0.05 dB safety, not more


def test_limiter_survives_nan_and_mono():
    x = np.sin(np.arange(4800) * 0.3) * 2.0
    x[100] = np.nan
    x[200] = np.inf
    lim = TruePeakLimiter(SR, 1, -1.0)
    y = np.concatenate([lim.process(x), lim.flush()[0]])
    assert y.ndim == 1 and np.all(np.isfinite(y))
    assert true_peak_db(y, SR) <= -1.0 + 1e-6


# ---------------------------------------------------------------------------
# TPDF dither (Lipshitz, Wannamaker & Vanderkooy 1992)
# ---------------------------------------------------------------------------
def test_tpdf_noise_floor_and_bounds():
    q = lsb(16)
    r = np.random.default_rng(11)
    x = np.sin(np.arange(400_000) * 0.01) * 0.3 + r.uniform(-1, 1, 400_000) * 0.01
    y = tpdf_dither(x, 16, np.random.default_rng(12))
    e = y - x
    # output on the 16-bit grid, total error within +-1.5 LSB (1 LSB dither + 0.5 LSB rounding)
    assert np.allclose(np.round(y / q), y / q, atol=1e-9)
    assert np.max(np.abs(e)) <= 1.5 * q + 1e-15
    # LWV 1992: TPDF +-1 LSB makes the total error variance q^2/12 + q^2/6 = q^2/4
    assert np.var(e) / (q * q / 4) == pytest.approx(1.0, abs=0.03)
    assert abs(np.mean(e)) < 0.01 * q
    # noise floor: RMS q/2 -> 20 log10(2^-15 / 2) = -96.33 dBFS
    assert 20 * math.log10(np.std(e)) == pytest.approx(-96.33, abs=0.15)


def test_tpdf_error_moments_independent_of_signal():
    q = lsb(16)
    n = 100_000
    dither_means, dither_vars, plain_means = [], [], []
    for k in range(8):                                        # DC inputs across one LSB
        x = np.full(n, (k / 8.0 + 0.03) * q + 100 * q)
        e = tpdf_dither(x, 16, np.random.default_rng(100 + k)) - x
        dither_means.append(np.mean(e) / q)
        dither_vars.append(np.var(e) / (q * q / 4))
        plain_means.append(np.mean(quantize(x, 16) - x) / q)
    # LWV 1992: with TPDF the first two error moments do not depend on the input
    assert np.max(np.abs(dither_means)) < 0.01
    np.testing.assert_allclose(dither_vars, 1.0, atol=0.03)
    # without dither the error is a deterministic function of the input (sensitivity check)
    assert np.max(np.abs(plain_means)) > 0.3


def test_dither_linearises_sub_lsb_sine():
    q = lsb(16)
    n = SR
    t = np.arange(n) / SR
    x = 0.25 * q * np.sin(2 * np.pi * 1000.0 * t)              # a quarter of an LSB
    assert np.all(quantize(x, 16) == 0.0)                     # undithered: the tone vanishes
    y = tpdf_dither(x, 16, np.random.default_rng(3))
    amp = 2.0 / n * abs(np.sum(y * np.exp(-2j * np.pi * 1000.0 * t)))
    # dithered quantiser is linear on average: the 1 kHz component survives at its true level
    assert amp / (0.25 * q) == pytest.approx(1.0, abs=0.1)
    e = y - x
    assert abs(np.corrcoef(e, x)[0, 1]) < 0.01                 # error uncorrelated with the signal


def test_no_dither_at_32bit_float():
    x = np.random.default_rng(1).standard_normal((2, 1000)) * 0.1
    np.testing.assert_array_equal(tpdf_dither(x, 32, np.random.default_rng(0)), x)
    np.testing.assert_array_equal(Ditherer(32, RngFactory(1)).process(x, 0), x)


def test_ditherer_is_deterministic_per_seed_and_block():
    x = np.zeros((2, 1024))
    a = Ditherer(16, RngFactory(7)).process(x, 3)
    b = Ditherer(16, RngFactory(7)).process(x, 3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, Ditherer(16, RngFactory(7)).process(x, 4))
    assert not np.array_equal(a, Ditherer(16, RngFactory(8)).process(x, 3))
    assert not np.array_equal(a[0], a[1])                      # channels get independent dither


def test_highpass_tpdf_shape():
    q = lsb(16)
    d = Ditherer(16, RngFactory(5), shape="hp", quantize_output=False)
    x = np.zeros((1, 1 << 16))
    noise = np.concatenate([d.process(x[:, i:i + 4096], i // 4096) for i in range(0, x.shape[1], 4096)], axis=1)[0]
    # difference of two uniform +-0.5 LSB draws: triangular, variance q^2/6 (LWV 1992)
    assert np.var(noise) / (q * q / 6) == pytest.approx(1.0, abs=0.03)
    assert np.max(np.abs(noise)) <= q
    p = np.abs(np.fft.rfft(noise)) ** 2
    lo = np.mean(p[1:len(p) // 4])
    hi = np.mean(p[3 * len(p) // 4:])
    # (1 - z^-1) shaping: |H|^2 = 2 - 2 cos w, ~0.15 in the bottom quarter vs ~3.7 in the top quarter
    assert hi / lo > 10.0


# ---------------------------------------------------------------------------
# destinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,lufs,tp", [
    ("youtube", -14.0, -1.0),           # YouTube playback reference (unofficial), -1 dBTP
    ("spotify", -14.0, -1.0),           # Spotify loudness normalisation 'normal'
    ("apple_music", -16.0, -1.0),       # Apple Sound Check (unofficial)
    ("podcast", -16.0, -1.0),           # Apple Podcasts / AES TD1004 stereo
    ("broadcast_eu", -23.0, -1.0),      # EBU R128
    ("broadcast_us", -24.0, -2.0),      # ATSC A/85
])
def test_destination_targets_match_published_specs(name, lufs, tp):
    d = get_destination(name)
    assert d["target_lufs"] == lufs and d["tp_dbtp"] == tp and d["normalize"]
    assert d["sample_rate"] >= 44_100 and d["bit_depth"] in (16, 24)


def test_destinations_follow_the_plan():
    sb = get_destination("streaming_broadcaster")
    assert -20.0 <= sb["target_lufs"] <= -16.0 and sb["tp_dbtp"] == -1.0       # R128 s2 interim range
    sl = get_destination("sleep")
    assert -30.0 <= sl["target_lufs"] <= -24.0 and sl["lra_max"] <= 4.0 and sl["tp_dbtp"] == -1.0
    os_ = get_destination("online_streaming")
    assert -27.0 <= os_["target_lufs"] <= -23.0                                 # A/85 Annex L / AES71
    for name in ("post_stems", "game", "raw", "vr"):
        assert not get_destination(name)["normalize"]                          # calibrated, not normalised
    assert get_destination("post_stems")["tp_dbtp"] == -2.0
    assert get_destination("vr")["layout_hint"] == "foa"
    for d in DESTINATIONS.values():
        if d["format"] == "flac":
            assert d["layout_hint"] in ("stereo", "mono")                     # FLAC holds <= 8 channels
        if d["normalize"]:
            assert d["target_lufs"] is not None and d["tp_dbtp"] is not None


def test_destination_lookup_aliases_and_copies():
    assert get_destination("Apple-Music")["name"] == "apple_music"
    assert get_destination("EBU R128")["name"] == "broadcast_eu"
    assert get_destination("atsc")["name"] == "broadcast_us"
    assert get_destination("nonsense")["name"] == "raw"
    assert get_destination(None)["name"] == "raw"
    d = get_destination("youtube")
    d["target_lufs"] = 0.0
    assert DESTINATIONS["youtube"]["target_lufs"] == -14.0
    assert normalization_gain_db(-30.0, "youtube") == pytest.approx(16.0)
    assert normalization_gain_db(float("-inf"), "youtube") == 0.0
    assert normalization_gain_db(-30.0, "post_stems") == 0.0


# ---------------------------------------------------------------------------
# WAV / BWF / RF64
# ---------------------------------------------------------------------------
def read_bytes(path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def parse_chunks(path) -> tuple[bytes, list[tuple[bytes, int, int]]]:
    """(magic, [(id, size, payload offset)])."""
    data = read_bytes(path)
    magic = data[:4]
    assert data[8:12] == b"WAVE"
    out = []
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        out.append((cid, size, pos + 8))
        if size == 0xFFFFFFFF:
            break
        pos += 8 + size + (size & 1)
    return magic, out


def chunk(path, cid: bytes) -> bytes:
    data = read_bytes(path)
    for c, size, off in parse_chunks(path)[1]:
        if c == cid:
            return data[off:off + size]
    raise KeyError(cid)


@pytest.mark.parametrize("bits,channels", [(16, 1), (16, 2), (24, 2), (24, 6), (24, 12), (32, 2), (32, 16)])
def test_wav_round_trip_bit_exact(tmp_out, bits, channels):
    r = np.random.default_rng(bits * 100 + channels)
    n = 3001                                                       # odd: exercises the pad byte
    path = tmp_out / f"rt_{bits}_{channels}.wav"
    if bits == 32:
        x = r.uniform(-1, 1, (channels, n)).astype(np.float32).astype(np.float64)
    else:
        full = 1 << (bits - 1)
        codes = r.integers(-full, full, (channels, n))
        codes[:, 0] = -full
        codes[:, 1] = full - 1
        x = codes / full
    with StreamingWavWriter(str(path), SR, channels, bits) as w:
        for s in range(0, n, 1000):
            w.write(x[:, s:s + 1000])
    info = sf.info(str(path))
    assert (info.samplerate, info.channels, info.frames) == (SR, channels, n)
    if bits == 16:
        got = sf.read(str(path), dtype="int16", always_2d=True)[0].T
        np.testing.assert_array_equal(got, codes)
    elif bits == 24:
        got = sf.read(str(path), dtype="int32", always_2d=True)[0].T
        np.testing.assert_array_equal(got, codes << 8)                  # libsndfile left-justifies 24-bit
    else:
        got = sf.read(str(path), dtype="float32", always_2d=True)[0].T
        np.testing.assert_array_equal(got, x.astype(np.float32))
    magic, chunks = parse_chunks(path)
    assert magic == b"RIFF"
    ids = [c for c, _, _ in chunks]
    assert ids[0] == b"JUNK" and chunks[0][1] == 28                    # Tech 3306: space for ds64 first
    assert ids[1] == b"fmt " and ids[-1] == b"data"
    assert (b"fact" in ids) == (bits == 32)                              # non-PCM needs fact
    size = (tmp_out / f"rt_{bits}_{channels}.wav").stat().st_size
    assert struct.unpack("<I", read_bytes(path)[4:8])[0] == size - 8
    assert chunks[-1][1] == n * channels * bits // 8


def test_wav_clips_overs_and_cleans_nan(tmp_out):
    p = tmp_out / "clip.wav"
    with StreamingWavWriter(str(p), SR, 1, 16) as w:
        w.write(np.array([[2.0, -2.0, np.nan, np.inf, -np.inf]]))
    got = sf.read(str(p), dtype="int16")[0]
    np.testing.assert_array_equal(got, [32767, -32768, 0, 32767, -32768])


def test_fmt_extensible_masks_and_guid(tmp_out):
    p = tmp_out / "fmt.wav"
    with StreamingWavWriter(str(p), SR, 12, 24, channel_mask=0x2D63F) as w:
        w.write(np.zeros((12, 10)))
    fmt = chunk(p, b"fmt ")
    tag, ch, sr, byte_rate, align, bits, cb, valid, mask = struct.unpack("<HHIIHHHHI", fmt[:24])
    assert (tag, ch, sr, align, bits, cb, valid, mask) == (0xFFFE, 12, SR, 36, 24, 22, 24, 0x2D63F)
    assert byte_rate == SR * 36
    # KSDATAFORMAT_SUBTYPE_PCM = 00000001-0000-0010-8000-00AA00389B71 (mmreg.h)
    assert fmt[24:40] == bytes.fromhex("0100000000001000800000aa00389b71")
    pf = tmp_out / "float.wav"
    with StreamingWavWriter(str(pf), SR, 2, 32) as w:
        w.write(np.zeros((2, 10)))
    # KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = 00000003-0000-0010-8000-00AA00389B71
    assert chunk(pf, b"fmt ")[24:40] == bytes.fromhex("0300000000001000800000aa00389b71")
    assert struct.unpack("<I", chunk(pf, b"fact"))[0] == 10


@pytest.mark.parametrize("bw64", [False, True])
def test_rf64_ds64_header(tmp_out, bw64):
    p = tmp_out / "rf64.wav"
    n, C = 4801, 2
    x = np.random.default_rng(4).integers(-2 ** 23, 2 ** 23, (C, n)) / 2 ** 23
    with StreamingWavWriter(str(p), SR, C, 24, rf64_threshold=0, bw64=bw64,
                            bext={"description": "rf64 test"}) as w:
        w.write(x)
    raw = read_bytes(p)
    # EBU Tech 3306: 'RF64' (BS.2088: 'BW64'), 32-bit sizes = 0xFFFFFFFF, ds64 first, 28 bytes
    assert raw[:4] == (b"BW64" if bw64 else b"RF64")
    assert struct.unpack("<I", raw[4:8])[0] == 0xFFFFFFFF
    assert raw[12:16] == b"ds64" and struct.unpack("<I", raw[16:20])[0] == 28
    riff, data_size, frames, table = struct.unpack("<QQQI", raw[20:48])
    assert riff == len(raw) - 8
    assert data_size == n * C * 3 and frames == n and table == 0
    di = raw.index(b"data")
    assert struct.unpack("<I", raw[di + 4:di + 8])[0] == 0xFFFFFFFF
    if not bw64:                                       # libsndfile reads RF64 back
        got = sf.read(str(p), dtype="int32", always_2d=True)[0].T
        np.testing.assert_array_equal(got, (x * 2 ** 23).astype(np.int64) << 8)


def test_open_writer_can_force_rf64(tmp_out):
    p = tmp_out / "forced.wav"
    with open_writer(str(p), SR, 2, 24, {"rf64": True}) as w:
        w.write(np.zeros((2, 100)))
    assert read_bytes(p)[:4] == b"RF64"


def test_bext_layout_per_tech3285(tmp_out):
    payload = build_bext({"description": "D" * 300, "originator": "raingen", "originator_reference": "RGN0001",
                          "origination_date": "2026-11-16", "origination_time": "09:30:00",
                          "time_reference": 5_000_000_000, "umid": "ab" * 64, "loudness_value": -23.0,
                          "loudness_range": 5.5, "max_true_peak_level": -1.0,
                          "max_momentary_loudness": float("-inf"), "max_short_term_loudness": None},
                         SR, 2, 24)
    # Tech 3285 v2 offsets: Description 0/256, Originator 256/32, OriginatorReference 288/32,
    # OriginationDate 320/10, OriginationTime 330/8, TimeReference 338/8, Version 346/2,
    # UMID 348/64, loudness 412..421 (5 x int16), Reserved 422/180, CodingHistory 602..
    assert payload[0:256] == b"D" * 256                       # truncated to the field
    assert payload[256:288].rstrip(b"\0") == b"raingen"
    assert payload[288:320].rstrip(b"\0") == b"RGN0001"
    assert payload[320:330] == b"2026-11-16"
    assert payload[330:338] == b"09:30:00"
    lo, hi = struct.unpack("<II", payload[338:346])
    assert (hi << 32) | lo == 5_000_000_000
    assert struct.unpack("<H", payload[346:348])[0] == 2
    assert payload[348:412] == bytes.fromhex("ab" * 64)
    lv = struct.unpack("<5h", payload[412:422])
    # loudness in 0.01 units; 0x7FFF = not set (used for -inf and None)
    assert lv == (-2300, 550, -100, 0x7FFF, 0x7FFF)
    assert payload[422:602] == b"\0" * 180
    hist = payload[602:].decode("ascii")
    assert hist.startswith("A=PCM,F=48000,W=24,M=stereo,T=") and hist.endswith("\r\n")   # EBU R98


def test_bext_time_reference_from_timecode_and_loudness_patch(tmp_out):
    p = tmp_out / "bwf.wav"
    md = {"ucs": {"FXName": "Rain", "Description": "Rain test"}, "start_tc": "01:00:00:00", "fps": 25}
    with open_writer(str(p), SR, 2, 24, md) as w:
        w.write(np.zeros((2, 4800)))
        w.set_loudness(-23.04, 4.2, -1.27, -20.0, -21.5)
    b = chunk(p, b"bext")
    lo, hi = struct.unpack("<II", b[338:346])
    assert (hi << 32) | lo == 3600 * SR                         # samples since midnight at 01:00:00:00
    assert struct.unpack("<5h", b[412:422]) == (-2304, 420, -127, -2000, -2150)
    assert sf.info(str(p)).frames == 4800


def test_timecode_conversions():
    assert timecode_to_seconds("01:00:00:00", 25) == 3600.0
    assert timecode_to_seconds("01:00:00:00", 24) == 3600.0
    # NDF 29.97 counts 30 frames per timecode second at 30000/1001 fps: 108000 frames = 3603.6 s
    assert timecode_to_seconds("01:00:00:00", 29.97) == pytest.approx(3603.6, abs=1e-9)
    # SMPTE ST 12-1 drop-frame: 01:00:00;00 = 107892 frames = 3599.9964 s
    assert timecode_to_frames("01:00:00;00", 29.97) == 107_892
    assert timecode_to_seconds("01:00:00;00", 29.97) == pytest.approx(3599.9964, abs=1e-6)
    assert timecode_to_frames("00:01:00;02", 29.97) == 1800              # ;00 and ;01 do not exist
    assert timecode_to_frames("00:10:00;00", 29.97) == 17_982
    with pytest.raises(ValueError):
        timecode_to_frames("01:00:00;00", 25)


def test_ixml_is_well_formed_and_carries_the_scene(scene):
    from raingen.scene import Scene

    scene.name = "Storm & <rain> \x01 test"
    ucs = ucs_metadata(scene)
    doc = build_ixml(ucs, scene, ["L", "R"], extra={"1bad key": "a\x02b", "RAINGEN_LAYOUT": "stereo",
                                                     "xmlish": {"k": [1, 2]}}, fps=29.97, sr=SR)
    root = ET.fromstring(doc.encode("utf-8"))                 # well-formed XML 1.0
    assert root.tag == "BWFXML"
    assert root.findtext("SPEED/TIMECODE_RATE") == "30000/1001"
    assert root.findtext("SPEED/FILE_SAMPLE_RATE") == str(SR)
    assert root.findtext("TRACK_LIST/TRACK_COUNT") == "2"
    user = root.find("USER")
    assert user.findtext("CATID") == ucs["CatID"]
    assert user.findtext("F_1BADKEY") == "a b"
    assert json.loads(user.findtext("F_XMLISH")) == {"k": [1, 2]}
    back = Scene.from_dict(json.loads(user.findtext("RAINGEN_SCENE")))
    assert back.to_dict() == scene.to_dict()


def test_ixml_chunk_in_file_parses(tmp_out, scene):
    p = tmp_out / "ixml.wav"
    with open_writer(str(p), SR, 2, 24, {"ucs": ucs_metadata(scene), "scene": scene}) as w:
        w.write(np.zeros((2, 100)))
    root = ET.fromstring(chunk(p, b"iXML"))
    assert root.find("USER").findtext("RAINGEN_SEED") == f"0x{scene.seed:X}"


def test_flac_vorbis_comment_carries_scene(tmp_out, scene):
    from raingen.scene import Scene

    p = tmp_out / "scene.flac.part"                                 # the renderer's temp name
    w = open_writer(str(p), SR, 2, 24, {"scene": scene, "ucs": ucs_metadata(scene), "comment": "hello"})
    assert isinstance(w, FlacWriter)
    w.write(np.zeros((2, 1000)))
    w.close()
    f = sf.SoundFile(str(p))
    assert f.format == "FLAC" and f.subtype == "PCM_24"
    comment = f.comment
    assert comment.startswith("hello\nRAINGEN_SCENE=")
    back = Scene.from_dict(json.loads(comment.split("RAINGEN_SCENE=", 1)[1]))
    assert back.to_dict() == scene.to_dict()
    with pytest.raises(ValueError):
        FlacWriter(str(tmp_out / "x.flac"), SR, 12)                  # FLAC: at most 8 channels


def test_renderer_publishes_with_os_replace():
    import raingen.render

    src = inspect.getsource(raingen.render)
    # atomic on POSIX and Windows: os.replace overwrites an existing target, os.rename does not on Windows
    assert '+ ".part"' in src and "os.replace(" in src and "os.rename(" not in src


# ---------------------------------------------------------------------------
# UCS naming
# ---------------------------------------------------------------------------
# A reference subset of the UCS v8.2.1 Full List (CatID -> Category, SubCategory)
UCS_821_REFERENCE = {
    "RAIN": ("RAIN", "GENERAL"), "RAINClth": ("RAIN", "CLOTH"), "RAINConc": ("RAIN", "CONCRETE"),
    "RAINGlas": ("RAIN", "GLASS"), "RAINInt": ("RAIN", "INTERIOR"), "RAINMetl": ("RAIN", "METAL"),
    "RAINPlas": ("RAIN", "PLASTIC"), "RAINVege": ("RAIN", "VEGETATION"), "RAINWatr": ("RAIN", "WATER"),
    "RAINWood": ("RAIN", "WOOD"), "WIND": ("WIND", "GENERAL"), "WINDGust": ("WIND", "GUST"),
    "WINDInt": ("WIND", "INTERIOR"), "WINDTurb": ("WIND", "TURBULENT"), "WINDVege": ("WIND", "VEGETATION"),
    "WINDTonl": ("WIND", "TONAL"), "WINDDsgn": ("WIND", "DESIGNED"),
    "HAIL": ("WEATHER", "HAIL"), "STORM": ("WEATHER", "STORM"), "THUN": ("WEATHER", "THUNDER"),
    "WTHR": ("WEATHER", "MISC"), "WATRDrip": ("WATER", "DRIP"), "WATRFlow": ("WATER", "FLOW"),
    "WATRSplsh": ("WATER", "SPLASH"), "WATRMisc": ("WATER", "MISC"), "AMBForst": ("AMBIENCE", "FOREST"),
    "AMBUrbn": ("AMBIENCE", "URBAN"), "AMBSubn": ("AMBIENCE", "SUBURBAN"), "AMBRurl": ("AMBIENCE", "RURAL"),
    "AMBSea": ("AMBIENCE", "SEASIDE"), "AMBTrop": ("AMBIENCE", "TROPICAL"), "AMBLake": ("AMBIENCE", "LAKESIDE"),
    "AMBGras": ("AMBIENCE", "GRASSLAND"), "AMBInsc": ("AMBIENCE", "INSECT"), "AMBMisc": ("AMBIENCE", "MISC"),
}
UCS_NAME = re.compile(r"^(?P<cat>[A-Za-z]+)_(?P<fx>[^_]+)_(?P<creator>[A-Za-z0-9-]+)_(?P<source>[A-Za-z0-9-]+)"
                      r"(?:_(?P<user>[A-Za-z0-9-]+))?\.wav$")


def test_ucs_table_matches_v821():
    for catid, (cat, sub) in UCS_CATIDS.items():
        if catid in UCS_821_REFERENCE:
            assert UCS_821_REFERENCE[catid] == (cat, sub), catid
    for bad in ("WTHRThun", "WTHRStrm", "WTHRHail", "WTHRMisc", "AMBInsct", "AMB", "WATR"):
        assert bad not in UCS_CATIDS                          # not CatIDs in UCS v8.2.1


def test_ucs_filenames_for_all_presets():
    from raingen.scene.presets import list_presets, preset

    stems = (None, "rain_near", "rain_far", "wind", "thunder", "aftermath", "ambience", "interior")
    for entry in list_presets():
        name = entry if isinstance(entry, str) else entry.get("name", entry.get("id"))
        sc = preset(name)
        for stem in stems:
            fn = ucs_filename(sc, stem, ext="wav")
            m = UCS_NAME.match(fn)
            assert m, fn                                        # UCS: CatID_FXName_CreatorID_SourceID
            assert m["cat"] in UCS_CATIDS, fn
            assert len(m["fx"]) <= 100 and not re.search(r'[\\/:*?"<>|]', m["fx"])
            md = ucs_metadata(sc, stem)
            assert (md["Category"], md["SubCategory"]) == UCS_CATIDS[md["CatID"]]
            assert md["Filename"] + ".wav" == fn


def test_ucs_categories_follow_the_scene(scene):
    scene.weather.precipitation.type = "rain"
    scene.weather.precipitation.rate_mm_h = 6.0
    scene.place.shelter = "attic"
    assert ucs_catid(scene) == "RAINInt"                      # UCS: rain heard from inside a building
    scene.place.shelter = "tent"
    assert ucs_catid(scene) == "RAINClth"
    assert ucs_catid(scene, "thunder") == "THUN"
    scene.weather.precipitation.type = "hail"
    assert ucs_catid(scene) == "HAIL"
    scene.weather.precipitation.type = "snow"
    assert ucs_catid(scene) == "WTHR"
    scene.weather.precipitation.type = "rain"
    scene.weather.precipitation.rate_mm_h = 30.0
    scene.place.shelter = "open"
    scene.weather.thunder.enabled = True
    scene.weather.thunder.distance_km = 2.0
    assert ucs_catid(scene) == "STORM"
    scene.weather.wind.speed_mps = 25.0
    assert ucs_catid(scene, "wind") == "WINDTurb"
    assert "Heavy rain" in ucs_fxname(scene)                   # WMO: heavy 10-50 mm/h


# ---------------------------------------------------------------------------
# captions and cue export
# ---------------------------------------------------------------------------
SRT_BLOCK = re.compile(r"^(\d+)\n(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})\n((?:.+\n){1,2})$")
VTT_TIME = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3}) --> (\d\d:\d\d:\d\d\.\d{3})$")


def _secs(ts: str) -> float:
    h, m, rest = ts.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _markers() -> list[EventMarker]:
    return [
        EventMarker(0.5, "lightning", {"distance_km": 12.0, "bearing_deg": 270.0}),
        EventMarker(36.0, "thunder", {"distance_km": 12.0, "bearing_deg": 270.0, "flash_t": 0.5}),
        EventMarker(2.0, "gust", {"beaufort": 8.0, "from_deg": 225.0}),
        EventMarker(2.4, "drip", {"target": "bucket"}),
        EventMarker(60.0, "thunder", {"distance_km": 0.8, "azimuth": math.pi / 2, "peak_db_spl": 118.0}),
        EventMarker(63.0, "thunder", {"distance_km": 4.0, "azimuth": 0.0, "peak_db_spl": 96.0}),
        EventMarker(66.0, "thunder", {"distance_km": 2.0, "azimuth": 0.0, "peak_db_spl": 104.0}),
        EventMarker(90.0, "thunder", {"distance_km": 6.0, "azimuth": math.pi}),
        EventMarker(3725.25, "hail", {"type": "hail"}),
    ]


def test_caption_wording_is_accessible():
    m = _markers()
    # plan sec. 2.9 style example, verbatim
    assert cap.caption_for(m[1]) == "[Distant thunder rumbles to the west]"
    assert cap.caption_for(m[4]) == "[Thunder cracks to the left]"
    assert cap.caption_for(m[7]) == "[Thunder rumbles behind]"
    assert cap.caption_for(m[2]) == "[A strong gust of wind from the south-west]"
    assert cap.caption_for(m[3]) == "[Water drips into the bucket]"
    assert cap.caption_for(EventMarker(0, "thunder", {"distance_km": 0.1})) == "[Thunder cracks overhead]"
    for mk in m:
        text = cap.caption_for(mk)
        assert text.startswith("[") and text.endswith("]") and text[1].isupper()
        assert not re.search(r"\d", text)                      # no numbers or units in captions (data -> CSV)


def test_srt_format_and_cue_rules(tmp_out):
    p = tmp_out / "c.srt"
    n = cap.write_srt(_markers(), str(p))
    text = p.read_text(encoding="utf-8")
    blocks = text.split("\n\n")[:-1]
    assert len(blocks) == n
    prev_end = -1.0
    for i, b in enumerate(blocks, start=1):
        mt = SRT_BLOCK.match(b + "\n")
        assert mt, b
        assert int(mt[1]) == i
        s, e = _secs(mt[2]), _secs(mt[3])
        assert s >= prev_end - 1e-9                              # no overlapping cues
        assert e - s >= 1.5 - 1e-9                               # plan sec. 2.9: each shows >= 1.5 s
        lines = mt[4].rstrip("\n").split("\n")
        assert len(lines) <= 2 and all(len(line) <= 42 for line in lines)
        prev_end = e
    assert "Lightning" not in text                              # visual-only events stay out of captions
    # the three thunder events at 60/63/66 s merge into one cue named after the closest one
    assert text.count("Thunder cracks to the left") == 1
    assert "Thunder rumbles\n" not in text and "Loud thunder crashes" not in text
    assert "01:02:05,250 --> " in text                           # hail at 3725.25 s
    assert "00:00:36,000 --> " in text                           # thunder captioned at its arrival


def test_vtt_format(tmp_out):
    p = tmp_out / "c.vtt"
    n = cap.write_vtt(_markers(), str(p))
    lines = p.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "WEBVTT" and lines[1] == ""
    times = [ln for ln in lines if "-->" in ln]
    assert len(times) == n
    for ln in times:
        assert VTT_TIME.match(ln), ln


def test_merge_window_is_ten_seconds():
    near = [EventMarker(t, "gust", {"beaufort": 5.0}) for t in (0.0, 4.0, 9.0)]
    assert len(cap._cues(near)) == 1                            # within 10 s: one cue
    far = [EventMarker(t, "gust", {"beaufort": 5.0}) for t in (0.0, 12.0)]
    assert len(cap._cues(far)) == 2


def test_csv_cue_sheet_is_well_formed(tmp_out):
    p = tmp_out / "c.csv"
    n = cap.write_csv(_markers(), str(p), fps=25, start_tc="01:00:00:00")
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert len(body) == n == len(_markers())                     # lightning included in the cue sheet
    assert all(len(r) == len(header) for r in body)
    for r in body:
        rec = dict(zip(header, r, strict=True))
        assert re.match(r"^\d\d:\d\d:\d\d:\d\d$", rec["timecode"])
        json.loads(rec["data_json"])
        float(rec["time_s"])
    first = dict(zip(header, body[0], strict=True))
    assert first["kind"] == "lightning" and first["timecode"] == "01:00:00:12"   # 0.5 s at 25 fps
    assert first["distance_km"] == "12.000" and first["direction"] == "west"


def test_edl_is_cmx3600(tmp_out):
    p = tmp_out / "c.edl"
    many = _markers() + [EventMarker(100.0 + i, "drip", {}) for i in range(1000)]
    n = cap.write_edl(many, str(p), fps=25, start_tc="01:00:00:00")
    raw = read_bytes(p)
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")      # CRLF throughout
    lines = raw.decode("ascii").split("\r\n")
    assert lines[0].startswith("TITLE: ") and lines[1] == "FCM: NON-DROP FRAME"
    ev = re.compile(r"^(\d{3})  AX       AA    C        (\S{11}) (\S{11}) (\S{11}) (\S{11})$")
    events = [ev.match(ln) for ln in lines if ev.match(ln)]
    assert len(events) == n
    nums = [int(e[1]) for e in events]
    assert nums[:3] == [1, 2, 3] and 0 not in nums and max(nums) == 999
    assert events[0][4] == "01:00:00:12"                         # record in = start_tc + 0.5 s
    for e in events:
        assert timecode_to_frames(e[5], 25) > timecode_to_frames(e[4], 25)   # record out after record in
        assert timecode_to_frames(e[3], 25) - timecode_to_frames(e[2], 25) == \
            timecode_to_frames(e[5], 25) - timecode_to_frames(e[4], 25)      # source and record durations match
    assert sum(ln.startswith("* LOC: ") for ln in lines) == n


def test_seconds_to_timecode():
    assert cap.seconds_to_timecode(1.5, 24, "01:00:00:00") == "01:00:01:12"
    assert cap.seconds_to_timecode(0.0, 25) == "00:00:00:00"
    assert cap.seconds_to_timecode(86400.0, 25) == "00:00:00:00"            # wraps at 24 h
    # drop frame: 60 s at 30000/1001 = 1798.2 frames -> 00:00:59;28; frame 1800 -> 00:01:00;02
    assert cap.seconds_to_timecode(60.0, 29.97, "00:00:00;00") == "00:00:59;28"
    assert cap.seconds_to_timecode(1800 * 1001 / 30000, 29.97, "00:00:00;00") == "00:01:00;02"
    r = np.random.default_rng(9)
    for f in r.integers(0, 2_589_407, 200):                                 # DF label <-> frame round trip
        tc = cap.seconds_to_timecode((int(f) + 0.5) * 1001 / 30000, 29.97, "00:00:00;00")
        assert timecode_to_frames(tc, 29.97) == f


# ---------------------------------------------------------------------------
# end to end through the renderer (short clips)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_short_clip_is_normalised_end_to_end(scene):
    from raingen.render import Renderer

    scene.weather.precipitation.rate_mm_h = 2.0
    r = Renderer(scene, 0.25, destination="youtube")
    y, _ = r.to_array()
    assert "programme is silent; loudness normalisation skipped" not in r.warnings
    assert np.all(np.isfinite(y)) and y.shape == (2, 12_000)
    # 250 ms is shorter than one gating block: the provisional reading drives normalisation
    assert integrated_loudness(y, SR) == pytest.approx(-14.0, abs=1.0)       # YouTube target
    assert true_peak_db(y, SR) <= -1.0 + 1e-6                                 # YouTube ceiling
    assert abs(float(np.mean(y))) < 1e-2                                       # no DC


# ---------------------------------------------------------------------------
# no denormal stalls on digital silence
# ---------------------------------------------------------------------------
def _has_subnormals(a: np.ndarray) -> bool:
    a = np.abs(np.asarray(a, dtype=np.float64))
    return bool(np.any((a > 0.0) & (a < np.finfo(np.float64).tiny)))


def test_meter_and_limiter_state_never_goes_subnormal():
    x = np.random.default_rng(0).standard_normal((2, SR)) * 0.5
    m = LoudnessMeter(SR, 2)
    lim = TruePeakLimiter(SR, 2, -1.0)
    for s in range(0, SR, 1024):
        m.process(x[:, s:s + 1024])
        lim.process(x[:, s:s + 1024] * 4.0)
    z = np.zeros((2, 1024))
    for _ in range(400):                                   # ~8.5 s of digital silence
        m.process(z)
        y = lim.process(z)
    # K-weighting recursion would otherwise sit in a subnormal limit cycle (~16x slower blocks)
    assert not _has_subnormals(m._zi)
    assert not _has_subnormals([lim._a_rel]) and not _has_subnormals(lim._b1) and not _has_subnormals(lim._b2)
    np.testing.assert_array_equal(y, 0.0)                    # silence in, exact silence out


@pytest.mark.slow
def test_render_to_bwf_file_end_to_end(tmp_out, scene):
    from raingen.render import Renderer

    scene.weather.precipitation.rate_mm_h = 4.0
    path = tmp_out / "mix.wav"
    path.write_bytes(b"stale file from an earlier run")         # must be replaced atomically
    rep = Renderer(scene, 0.5, destination="broadcast_eu", start_tc="01:00:00:00", fps=25).run(
        str(path), captions=("srt", "vtt", "csv", "edl"))
    assert not list(tmp_out.glob("*.part"))                       # temp file renamed away
    info = sf.info(str(path))
    assert (info.samplerate, info.channels, info.frames, info.subtype) == (SR, 2, SR // 2, "PCM_24")
    y = sf.read(str(path), always_2d=True)[0].T
    assert np.all(np.isfinite(y))
    b = chunk(path, b"bext")
    lo, hi = struct.unpack("<II", b[338:346])
    assert (hi << 32) | lo == 3600 * SR                           # TimeReference from start_tc
    lv = struct.unpack("<5h", b[412:422])
    # bext loudness = the meter's readings x 100 (Tech 3285 v2), and EBU R128: -23 LUFS / -1 dBTP
    assert lv[0] == round(rep["integrated_lufs"] * 100)
    assert lv[2] == round(rep["true_peak_dbtp"] * 100)
    assert rep["integrated_lufs"] == pytest.approx(-23.0, abs=0.5)
    assert rep["true_peak_dbtp"] <= -1.0 + 1e-6
    assert integrated_loudness(y, SR) == pytest.approx(rep["integrated_lufs"], abs=0.05)   # 24-bit file reads back
    root = ET.fromstring(chunk(path, b"iXML"))
    assert root.find("USER").findtext("RAINGEN_DESTINATION") == "broadcast_eu"
    for fmt in ("srt", "vtt", "csv", "edl"):
        assert (tmp_out / f"mix.{fmt}").exists()


def test_captions_never_exceed_two_lines():
    long = EventMarker(1.0, "custom", {"caption": "[" + "very " * 30 + "long description]"})
    burst = [EventMarker(5.0 + 0.1 * i, kind, {}) for i, kind in enumerate(("hail", "drip", "gust", "rain_start"))]
    for s, e, text, _ in cap._cues([long, *burst]):
        assert len(text.split("\n")) <= 2                       # plan sec. 2.9: <= 2 lines
        assert e - s >= 1.5 - 1e-9
