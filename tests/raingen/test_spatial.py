"""Spatial audio: AmbiX spherical harmonics, encoder, loudspeaker / binaural decoders, HRTFs.

Conventions under test (AmbiX, Nachbar, Zotter, Deleflie & Sontacchi 2011): ACN channel order,
SN3D normalisation, no Condon-Shortley phase, azimuth counter-clockwise with 0 = front and
+pi/2 = left, elevation up.
"""

import math
import sys
import warnings

import numpy as np
import pytest

from raingen.core.rng import RngFactory
from raingen.core.types import Emission
from raingen.spatial.decoders import (
    CHANNEL_MASKS,
    SPEAKER_LAYOUTS,
    allrad_matrix,
    channel_mask_for,
    make_decoder,
    normalize_layout,
)
from raingen.spatial.encoder import (
    DecorrelatorBank,
    Encoder,
    design_allpass_firs,
    spread_gains,
)
from raingen.spatial.hrtf import (
    HEAD_RADIUS_M,
    SPEED_OF_SOUND,
    BuiltinHRTF,
    get_hrtf,
    load_sofa,
    spherical_head_response,
)
from raingen.spatial.sh import (
    acn_nm,
    acn_order_index,
    cap_weights,
    fibonacci_sphere,
    lebedev26,
    legendre,
    maxre_energy_compensation,
    maxre_weights,
    n_channels,
    order_of,
    rotation_matrix,
    sh_from_xyz,
    sh_matrix,
    sh_real,
    unit_vectors,
    yaw_rotation_matrix,
)

ORDER = 3
N = n_channels(ORDER)
SR = 48_000
BLOCK = 1024
DEG = math.pi / 180.0

# WAVE_FORMAT_EXTENSIBLE dwChannelMask speaker bits (Microsoft, "Multiple channel audio data and
# WAVE files", ksmedia.h). Channels in a WAV file must appear in increasing bit order.
WAVE_BITS = {
    "FL": 0x1, "FR": 0x2, "FC": 0x4, "LFE": 0x8, "BL": 0x10, "BR": 0x20, "SL": 0x200, "SR": 0x400,
    "TFL": 0x1000, "TFR": 0x4000, "TBL": 0x8000, "TBR": 0x20000,
}
# our channel names -> WAVE speaker positions (5.1 surrounds at +-110 deg use the side bits, as in
# the 0x60F "5.1 (side)" mask; 7.1 rear surrounds are BL/BR, side surrounds SL/SR)
NAME_TO_BIT = {
    "5.1": {"L": "FL", "R": "FR", "C": "FC", "LFE": "LFE", "Ls": "SL", "Rs": "SR"},
    "7.1": {"L": "FL", "R": "FR", "C": "FC", "LFE": "LFE", "Lrs": "BL", "Rrs": "BR", "Lss": "SL", "Rss": "SR"},
    "7.1.4": {"L": "FL", "R": "FR", "C": "FC", "LFE": "LFE", "Lrs": "BL", "Rrs": "BR", "Lss": "SL", "Rss": "SR",
              "Ltf": "TFL", "Rtf": "TFR", "Ltr": "TBL", "Rtr": "TBR"},
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sphere_quadrature(n_theta=24, n_phi=48):
    """Gauss-Legendre in sin(el) x uniform azimuth: exact for SH products up to degree 2*n_theta - 1."""
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    el = np.arcsin(x)
    az = np.arange(n_phi) * 2.0 * math.pi / n_phi
    A, E = np.meshgrid(az, el)
    W = np.outer(wx, np.full(n_phi, 2.0 * math.pi / n_phi))
    return A.ravel(), E.ravel(), W.ravel()


def _band_db(y, sr, lo, hi, nfft=None):
    """Mean power (dB) of each row of y in [lo, hi) Hz."""
    nfft = nfft or y.shape[-1]
    Y = np.fft.rfft(y, nfft, axis=-1)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    k = (f >= lo) & (f < hi)
    return 10.0 * np.log10(np.mean(np.abs(Y[..., k]) ** 2, axis=-1))


def _stream(encoder, decoder, signal_fn, n_blocks, n=BLOCK, **em_kw):
    """Run blocks through encoder (+ decoder); em_kw values may be callables of the block index."""
    out = []
    for b in range(n_blocks):
        kw = {k: (v(b) if callable(v) else v) for k, v in em_kw.items()}
        bus = encoder.encode(Emission(signal_fn(b), **kw), n)
        out.append(decoder.process(bus) if decoder is not None else bus)
    return np.concatenate(out, axis=1)


def _boundary_jump_ratio(y, block, skip_blocks=2):
    """max |sample-to-sample jump| at block boundaries / 99.9th percentile of jumps elsewhere.

    d[i] = y[i + 1] - y[i]; the jumps across a boundary are d[k*block - 1] (and its neighbours).
    A click (discontinuity) adds to the signal's own slope and stands out; returns the worst channel.
    """
    d = np.abs(np.diff(np.atleast_2d(y), axis=-1))
    idx = np.arange(d.shape[-1]) + 1                 # d[i] ends on sample i + 1
    at = np.zeros(d.shape[-1], dtype=bool)
    for off in (-1, 0, 1):
        at |= ((idx + off) % block == 0)
    valid = idx > skip_blocks * block
    rest = ~at & valid
    at &= valid
    ratios = []
    for ch in d:
        if not np.any(ch[rest]):
            continue
        ratios.append(np.max(ch[at]) / max(np.percentile(ch[rest], 99.9), 1e-30))
    return max(ratios)


# ---------------------------------------------------------------------------
# spherical harmonics (sh.py)
# ---------------------------------------------------------------------------
def test_acn_indexing_and_channel_counts():
    # ACN = n^2 + n + m (Nachbar et al. 2011, eq. 1)
    n, m = acn_nm(3)
    assert list(n) == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    assert list(m) == [0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3]
    assert [n_channels(k) for k in range(5)] == [1, 4, 9, 16, 25]
    assert order_of(16) == 3 and order_of(4) == 1
    with pytest.raises(ValueError):
        order_of(5)


def test_first_order_sn3d_values(rng):
    az = rng.uniform(-math.pi, math.pi, 64)
    el = rng.uniform(-math.pi / 2, math.pi / 2, 64)
    Y = sh_real(1, az, el)
    # AmbiX first order (Nachbar et al. 2011, Table 1): W = 1, Y = sin az cos el, Z = sin el, X = cos az cos el
    np.testing.assert_allclose(Y[0], 1.0, atol=1e-14)
    np.testing.assert_allclose(Y[1], np.sin(az) * np.cos(el), atol=1e-14)
    np.testing.assert_allclose(Y[2], np.sin(el), atol=1e-14)
    np.testing.assert_allclose(Y[3], np.cos(az) * np.cos(el), atol=1e-14)
    # scalar path agrees with the vector path
    for a, e in zip(az[:5], el[:5]):
        np.testing.assert_allclose(sh_real(3, a, e), sh_real(3, np.array([a]), np.array([e]))[:, 0], atol=1e-14)
    # +pi/2 is left: Y (ACN 1) = +1 for a source at the left, X (ACN 3) = +1 in front
    np.testing.assert_allclose(sh_real(1, math.pi / 2, 0.0), [1.0, 1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(sh_real(1, 0.0, 0.0), [1.0, 0.0, 0.0, 1.0], atol=1e-15)


def test_orders_2_and_3_match_ambix_sn3d_table(rng):
    az = rng.uniform(-math.pi, math.pi, 50)
    el = rng.uniform(-math.pi / 2, math.pi / 2, 50)
    s, c = np.sin(el), np.cos(el)
    r3 = math.sqrt(3.0)
    # Real SN3D spherical harmonics, ACN 4..15 (Nachbar et al. 2011 / ambiX table; Zotter & Frank 2019, Table A.1)
    ref = {
        4: r3 / 2 * np.sin(2 * az) * c ** 2,
        5: r3 / 2 * np.sin(az) * np.sin(2 * el),
        6: 0.5 * (3 * s ** 2 - 1),
        7: r3 / 2 * np.cos(az) * np.sin(2 * el),
        8: r3 / 2 * np.cos(2 * az) * c ** 2,
        9: math.sqrt(5 / 8) * np.sin(3 * az) * c ** 3,
        10: math.sqrt(15) / 2 * np.sin(2 * az) * s * c ** 2,
        11: math.sqrt(3 / 8) * np.sin(az) * c * (5 * s ** 2 - 1),
        12: 0.5 * s * (5 * s ** 2 - 3),
        13: math.sqrt(3 / 8) * np.cos(az) * c * (5 * s ** 2 - 1),
        14: math.sqrt(15) / 2 * np.cos(2 * az) * s * c ** 2,
        15: math.sqrt(5 / 8) * np.cos(3 * az) * c ** 3,
    }
    Y = sh_real(3, az, el)
    for k, v in ref.items():
        np.testing.assert_allclose(Y[k], v, atol=1e-13, err_msg=f"ACN {k}")


def test_sn3d_orthogonality_on_the_sphere():
    az, el, w = _sphere_quadrature()
    Y = sh_real(ORDER, az, el)
    G = (Y * w) @ Y.T
    n = acn_order_index(ORDER)
    # SN3D: integral of Y_nm^2 over the sphere = 4 pi / (2n + 1), cross terms vanish (Zotter & Frank 2019, eq. 4.33)
    np.testing.assert_allclose(np.diag(G), 4 * math.pi / (2 * n + 1), rtol=1e-12)
    np.testing.assert_allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-12)


def test_addition_theorem_and_diffuse_variance(rng):
    u = rng.normal(size=(20, 3))
    v = rng.normal(size=(20, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    Yu, Yv = sh_from_xyz(ORDER, u), sh_from_xyz(ORDER, v)
    P = legendre(ORDER, np.sum(u * v, axis=1))
    n = acn_order_index(ORDER)
    for deg in range(ORDER + 1):
        # SN3D addition theorem: sum_m Y_nm(u) Y_nm(v) = P_n(u.v)  (Zotter & Frank 2019, eq. 4.34)
        np.testing.assert_allclose(np.sum(Yu[n == deg] * Yv[n == deg], axis=0), P[deg], atol=1e-12)


def test_lebedev26_and_dense_grid_quadrature():
    az, el, w = lebedev26()
    assert az.shape == (26,) and abs(np.sum(w) - 1.0) < 1e-14
    Y = sh_real(ORDER, az, el)
    G = (Y * w) @ Y.T
    n = acn_order_index(ORDER)
    # degree-7 Lebedev rule integrates degree-6 products exactly (Lebedev 1976)
    np.testing.assert_allclose(G, np.diag(1.0 / (2 * n + 1)), atol=1e-13)


def test_maxre_weights_match_zotter_frank():
    # Zotter & Frank 2012, JAES 60(10), eq. 10: a_n = P_n(r_E), r_E the largest root of P_{N+1}
    # (1/sqrt(3) at first order; 0.8611 at third order, Abramowitz & Stegun Table 25.4)
    np.testing.assert_allclose(maxre_weights(1), [1.0, 1.0 / math.sqrt(3.0)], atol=1e-12)
    np.testing.assert_allclose(maxre_weights(3), [1.0, 0.861136, 0.612333, 0.304747], atol=2e-6)
    np.testing.assert_allclose(maxre_weights(0), [1.0])
    # their closed-form approximation cos(137.9 deg / (N + 1.51)) agrees within 0.005
    for order in (1, 2, 3, 5):
        x = math.cos(math.radians(137.9) / (order + 1.51))
        np.testing.assert_allclose(maxre_weights(order), legendre(order, x), atol=5e-3)
    # energy compensation restores sum (2n+1) a_n^2 to sum (2n+1)
    g = maxre_weights(3) * maxre_energy_compensation(3)
    k = 2 * np.arange(4) + 1
    assert abs(np.sum(k * g * g) - np.sum(k)) < 1e-12


def test_cap_weights_limits():
    np.testing.assert_allclose(cap_weights(3, 0.0), 1.0)
    # whole sphere: only the mean survives
    np.testing.assert_allclose(cap_weights(3, math.pi), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    # hemisphere: first-order coefficient = mean of cos(theta) over the cap = 1/2; P_2 term vanishes
    np.testing.assert_allclose(cap_weights(3, math.pi / 2)[:3], [1.0, 0.5, 0.0], atol=1e-12)
    # small caps: mean of P_n(cos t) ~ 1 - n(n+1) t^2 / 4 over the cap, <t^2> = a^2 / 2 -> 1 - n(n+1) a^2 / 8
    a = 0.05
    np.testing.assert_allclose(cap_weights(3, a), [1 - k * (k + 1) * a * a / 8 for k in range(4)], atol=2e-5)  # O(a^4)


def test_rotations_move_sources_where_expected(rng):
    az, el = rng.uniform(-math.pi, math.pi, 10), rng.uniform(-1.2, 1.2, 10)
    Y = sh_real(ORDER, az, el)
    # yaw +90 deg turns the field CCW: a source in front ends up on the left
    R = yaw_rotation_matrix(ORDER, math.pi / 2)
    np.testing.assert_allclose(R @ Y, sh_real(ORDER, az + math.pi / 2, el), atol=1e-12)
    np.testing.assert_allclose(R @ sh_real(ORDER, 0.0, 0.0), sh_real(ORDER, math.pi / 2, 0.0), atol=1e-12)
    # general rotation: exact on SH of the rotated directions (an invariant subspace) and orthogonal in N3D
    Rg = rotation_matrix(ORDER, 0.3, 0.4, -0.2)
    u = unit_vectors(az, el)
    cy, sy, cp, sp, cr, sr_ = math.cos(.3), math.sin(.3), math.cos(.4), math.sin(.4), math.cos(-.2), math.sin(-.2)
    R3 = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
          @ np.array([[1, 0, 0], [0, cr, -sr_], [0, sr_, cr]]))
    np.testing.assert_allclose(Rg @ Y, sh_from_xyz(ORDER, u @ R3.T), atol=1e-9)
    s = np.sqrt(2.0 * acn_order_index(ORDER) + 1.0)
    Rn = (Rg * s[:, None]) / s[None, :]
    np.testing.assert_allclose(Rn @ Rn.T, np.eye(N), atol=1e-9)


def test_sh_matrix_shape():
    M = sh_matrix(ORDER, [0.0, 1.0, 2.0], [0.0, 0.1, -0.1])
    assert M.shape == (3, N)
    np.testing.assert_allclose(M[1], sh_real(ORDER, 1.0, 0.1))


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
def test_point_source_is_plane_wave_encoded(rng):
    x = rng.standard_normal(BLOCK)
    enc = Encoder(ORDER, SR)
    out = enc.encode(Emission(x, azimuth=0.7, elevation=0.3, spread=0.0, source_id="p"), BLOCK)
    np.testing.assert_allclose(out, np.outer(sh_real(ORDER, 0.7, 0.3), x), atol=1e-14)
    # SN3D: no channel of a plane wave exceeds W
    assert np.all(np.abs(sh_real(ORDER, 0.7, 0.3)) <= 1.0 + 1e-12)


@pytest.fixture(scope="module")
def decorrelators():
    return DecorrelatorBank(N - 1, SR, 20.0)


def test_decorrelators_are_true_allpass(decorrelators):
    firs = decorrelators.firs
    assert firs.shape == (N - 1, 960)
    # |H| on a 64x zero-padded grid, i.e. the response of the linear convolution between DFT bins
    mag_db = 20 * np.log10(np.abs(np.fft.rfft(firs, 65536, axis=1)))
    # requirement: flat magnitude within 0.1 dB at every frequency (all-pass; Kendall 1995)
    assert np.max(np.abs(mag_db)) < 0.1, np.max(np.abs(mag_db))
    np.testing.assert_allclose(np.sum(firs ** 2, axis=1), 1.0, rtol=1e-12)
    # short smear: 90 % of the energy within 12 ms (keeps drop transients compact)
    e = np.cumsum(firs ** 2, axis=1)
    assert np.max(np.argmax(e >= 0.9, axis=1)) < 0.012 * SR
    # mutually decorrelated: exactly orthogonal at zero lag (so a diffuse source gets the diagonal SN3D
    # covariance), and the max normalised cross-correlation over all lags stays at the few-tenths
    # level of independent random-phase filters (~ 3/sqrt(L_eff); Kendall 1995)
    X = firs @ firs.T
    np.testing.assert_allclose(X, np.eye(N - 1), atol=2e-3)
    assert np.max(np.abs(firs[:, 0])) < 2e-3          # and with the dry W path (zero-lag tap)
    F = np.fft.rfft(firs, 2048, axis=1)
    xc = np.abs(np.fft.irfft(F[:, None, :] * np.conj(F[None, :, :]), 2048, axis=-1)).max(axis=-1)
    assert np.max(xc[~np.eye(N - 1, dtype=bool)]) < 0.3


def test_allpass_design_beats_circular_normalisation():
    # the old circular design (|H| = 1 on the L-point grid only) had -48 dB notches in between;
    # a few L-BFGS iterations already flatten the true response by an order of magnitude
    noise = np.random.default_rng(5).standard_normal((2, 256))
    h = design_allpass_firs(noise, iters=150)
    mag_db = 20 * np.log10(np.abs(np.fft.rfft(h, 16384, axis=1)))
    assert np.max(np.abs(mag_db)) < 0.1


def test_spread_one_gives_diffuse_field_covariance(rng):
    # fully diffuse SN3D field: channel variances 1/(2n+1) relative to W, channels uncorrelated
    # (Zotter & Frank 2019, sec. 4.7 / the "diffuse-field covariance" of Gerzon's energy vector)
    enc = Encoder(ORDER, SR)
    blocks = 48
    sig = rng.standard_normal((blocks, BLOCK))
    y = _stream(enc, None, lambda b: sig[b], blocks, azimuth=0.4, elevation=0.2, spread=1.0, source_id="d")
    y = y[:, 2 * BLOCK:]
    C = (y @ y.T) / y.shape[1]
    n = acn_order_index(ORDER)
    var = np.diag(C) / C[0, 0]
    # tolerance: 46 blocks of white noise -> relative std of a variance estimate ~ sqrt(2/47000) ~ 0.7 %
    np.testing.assert_allclose(var, 1.0 / (2 * n + 1), rtol=0.06)
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    off = corr - np.eye(N)
    # decorrelated copies of one source: zero-lag correlation between channels only at the estimation
    # noise level of ~46k samples (std ~ 1/sqrt(n_eff) ~ 0.01 per pair, max over 120 pairs ~ 0.04)
    assert np.max(np.abs(off)) < 0.06
    # W carries the source unchanged whatever the spread
    np.testing.assert_allclose(y[0], sig.reshape(-1)[2 * BLOCK:], atol=1e-12)


def test_spread_gains_energy_and_limits():
    n = acn_order_index(ORDER)
    g0, d0 = spread_gains(ORDER, 0.3, 0.1, 0.0)
    np.testing.assert_allclose(g0, sh_real(ORDER, 0.3, 0.1))
    assert not d0.any()
    g1, d1 = spread_gains(ORDER, 0.3, 0.1, 1.0)
    np.testing.assert_allclose(g1[1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(d1[1:] ** 2, 1.0 / (2 * n[1:] + 1), rtol=1e-12)
    # W is never changed by spread: pressure (and loudness) independent of spread
    for s in np.linspace(0, 1, 11):
        g, d = spread_gains(ORDER, 1.0, -0.2, s)
        assert g[0] == 1.0 and d[0] == 0.0


def test_point_vs_diffuse_after_stereo_and_binaural_decoding(rng):
    blocks = 40
    sig = rng.standard_normal((blocks, BLOCK))

    def ic(y):
        return float(np.sum(y[0] * y[1]) / np.sqrt(np.sum(y[0] ** 2) * np.sum(y[1] ** 2)))

    st = {}
    for sp in (0.0, 1.0):
        y = _stream(Encoder(ORDER, SR), make_decoder("stereo", ORDER), lambda b: sig[b], blocks,
                    azimuth=0.0, elevation=0.0, spread=sp, source_id="s")[:, 2 * BLOCK:]
        st[sp] = ic(y)
    # a frontal point source is fully coherent; spread 1 falls to the decoder's diffuse-field coherence
    D = make_decoder("stereo", ORDER).matrix
    Cd = (D / (2.0 * acn_order_index(ORDER) + 1.0)) @ D.T
    ic_diffuse = Cd[0, 1] / math.sqrt(Cd[0, 0] * Cd[1, 1])
    assert st[0.0] > 0.999
    assert abs(st[1.0] - ic_diffuse) < 0.05
    assert st[1.0] < st[0.0] - 0.35

    # binaural: diffuse IC falls with frequency like the HRTF set's own (Zaunschirm et al. 2018, Fig. 4):
    # high below ~300 Hz, low above 1 kHz
    y = _stream(Encoder(ORDER, SR), make_decoder("binaural", ORDER), lambda b: sig[b], blocks,
                azimuth=0.0, elevation=0.0, spread=1.0, source_id="b")[:, 4 * BLOCK:]
    from scipy.signal import butter, sosfilt
    lo = sosfilt(butter(4, [80, 250], btype="band", fs=SR, output="sos"), y, axis=1)
    hi = sosfilt(butter(4, [2000, 8000], btype="band", fs=SR, output="sos"), y, axis=1)
    assert ic(lo) > 0.6
    assert abs(ic(hi)) < 0.2


def test_moving_source_has_no_clicks(rng):
    # a 150 Hz tone whirling round the head at 2 rev/s: every block's gains change
    t = np.arange(48 * BLOCK) / SR
    tone = 0.5 * np.sin(2 * math.pi * 150.0 * t)
    for layout in ("stereo", "binaural", "7.1.4"):
        y = _stream(Encoder(ORDER, SR), make_decoder(layout, ORDER), lambda b: tone[b * BLOCK:(b + 1) * BLOCK],
                    48, azimuth=lambda b: 2 * math.pi * 2.0 * b * BLOCK / SR, elevation=0.2, spread=0.3,
                    source_id="m")
        assert np.all(np.isfinite(y))
        # a jump at the block boundaries would stand out from the smooth interior; ramped gains keep it in
        assert _boundary_jump_ratio(y, BLOCK) < 1.3, layout
    # without smoothing (fresh state every block = hard gain switch) the same test fails
    hard = []
    for b in range(24):
        e = Encoder(ORDER, SR)
        hard.append(e.encode(Emission(tone[b * BLOCK:(b + 1) * BLOCK], azimuth=2 * math.pi * 2.0 * b * BLOCK / SR,
                                      source_id="m"), BLOCK))
    assert _boundary_jump_ratio(make_decoder("stereo", ORDER).process(np.concatenate(hard, axis=1)), BLOCK) > 2.0


def test_spread_to_zero_flushes_tail_without_click(rng):
    tone = 0.5 * np.sin(2 * math.pi * 220.0 * np.arange(20 * BLOCK) / SR)
    enc = Encoder(ORDER, SR)
    y = _stream(enc, None, lambda b: tone[b * BLOCK:(b + 1) * BLOCK], 20, azimuth=1.0,
                spread=lambda b: 0.8 if b < 10 else 0.0, source_id="f")
    assert _boundary_jump_ratio(y, BLOCK) < 1.3
    # the decorrelator tail is released, after which the bus is the exact point-source encoding
    st = enc._states["f"]
    assert st.tail is None
    np.testing.assert_allclose(y[:, 12 * BLOCK:], np.outer(sh_real(ORDER, 1.0, 0.0), tone[12 * BLOCK:]), atol=1e-12)


@pytest.mark.parametrize("n", [1, 7, 333, 1024, 2500])
def test_any_block_length_gives_the_same_stream(rng, n):
    total = 1500 if n < 10 else 6000
    x = rng.standard_normal(total)
    ref = Encoder(ORDER, SR)
    want = ref.encode(Emission(x, azimuth=-0.5, elevation=0.1, spread=0.5, source_id="a"), total)
    enc = Encoder(ORDER, SR)
    parts = []
    for s in range(0, total, n):
        seg = x[s:s + n]
        parts.append(enc.encode(Emission(seg, azimuth=-0.5, elevation=0.1, spread=0.5, source_id="a"), seg.shape[0]))
    got = np.concatenate(parts, axis=1)
    # static source: block partitioning must not matter (LTI decorrelator + carried tails)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_batched_stem_path_equals_per_emission_path(rng):
    """encode_into/finish_block (one decorrelator pass per stem) == sum of encode() (per source)."""
    a, b = Encoder(ORDER, SR), Encoder(ORDER, SR)
    route = {"rain_near": "rain_near", "rain_far": "rain_far", "mystery": "ambience"}
    for blk in range(10):
        ems = []
        for k in range(6):
            ems.append(Emission(rng.standard_normal(BLOCK), azimuth=0.3 * k + (0.05 * blk if k % 2 else 0.0),
                                elevation=0.1 * k - 0.2, spread=(0.0 if (k == 0 or (k == 3 and blk >= 5)) else 0.2 * k),
                                stem=("rain_near", "rain_far", "mystery")[k % 3], source_id=f"s{k}"))
        ems.append(Emission(rng.standard_normal((4, BLOCK)), stem="rain_far", is_ambisonic=True))
        want: dict = {}
        for em in ems:
            s = route[em.stem]
            want[s] = want.get(s, 0.0) + a.encode(em, BLOCK)
        got = b.encode_block(ems, BLOCK, stems=("rain_near", "rain_far", "ambience", "wind"),
                             stem_of=lambda s: route[s])
        assert set(got) == {"rain_near", "rain_far", "ambience", "wind"}
        assert not got["wind"].any()
        for s, v in want.items():
            np.testing.assert_allclose(got[s], v, atol=1e-12)


@pytest.mark.parametrize("sizes", [(1024, 1024, 1024, 1024), (100, 7, 2000, 1989), (5, 900, 1, 3190)])
def test_batched_path_any_block_length_and_ring_out(rng, sizes):
    total = sum(sizes)
    x = rng.standard_normal(total)
    want = Encoder(ORDER, SR).encode(Emission(x, azimuth=2.0, elevation=0.3, spread=0.6, source_id="z"), total)
    enc, got, s0 = Encoder(ORDER, SR), [], 0
    for n in sizes:
        buses = enc.encode_block([Emission(x[s0:s0 + n], azimuth=2.0, elevation=0.3, spread=0.6, stem="rain_far",
                                           source_id="z")], n)
        got.append(buses["rain_far"])
        s0 += n
    np.testing.assert_allclose(np.concatenate(got, axis=1), want, atol=1e-12)
    # the source stops: its decorrelator tail (< 20 ms) rings out into the next blocks, then silence
    ring = np.concatenate([enc.finish_block({}, 256).get("rain_far", np.zeros((N, 256))) for _ in range(6)], axis=1)
    assert np.any(ring[1:, :64]) and not ring[0].any()
    assert not ring[:, 960:].any()
    full = Encoder(ORDER, SR).encode(Emission(np.r_[x, np.zeros(1536)], azimuth=2.0, elevation=0.3, spread=0.6,
                                              source_id="z"), total + 1536)
    np.testing.assert_allclose(ring, full[:, total:], atol=1e-12)


def test_one_source_id_twice_in_a_block_keeps_both_positions(rng):
    """Two emissions sharing a source_id in one block must not share one ramp state (that would
    sweep the gains between the two positions every block); each stays an exact static plane wave."""
    enc = Encoder(ORDER, SR)
    ya, yb = sh_real(ORDER, 1.2, 0.0), sh_real(ORDER, -1.2, 0.3)
    for _ in range(4):
        xa, xb = rng.standard_normal(BLOCK), rng.standard_normal(BLOCK)
        bus = enc.encode_block([Emission(xa, azimuth=1.2, source_id="dup"),
                                Emission(xb, azimuth=-1.2, elevation=0.3, source_id="dup")], BLOCK)["rain_near"]
        np.testing.assert_allclose(bus, np.outer(ya, xa) + np.outer(yb, xb), atol=1e-12)


def test_legacy_decorrelator_process(decorrelators, rng):
    x = rng.standard_normal(700)
    a, ta = decorrelators.process(x, None)
    b, tb = decorrelators.process_gains(x, np.ones(N - 1), None, None, None)
    np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_allclose(ta, tb, atol=1e-14)
    np.testing.assert_allclose(a, np.stack([np.convolve(x, h)[:700] for h in decorrelators.firs]), atol=1e-12)


def test_static_sources_reuse_their_gains(monkeypatch, rng):
    from raingen.spatial import encoder as enc_mod
    calls = []
    real = enc_mod.spread_gains
    monkeypatch.setattr(enc_mod, "spread_gains", lambda *a: calls.append(a) or real(*a))
    enc = Encoder(ORDER, SR)
    for _ in range(5):
        enc.encode(Emission(rng.standard_normal(64), azimuth=0.2, spread=0.1, source_id="s"), 64)
    assert len(calls) == 1
    enc.encode(Emission(rng.standard_normal(64), azimuth=0.25, spread=0.1, source_id="s"), 64)
    assert len(calls) == 2


def test_encoder_determinism_and_seeded_decorrelators(rng):
    x = rng.standard_normal((6, BLOCK))

    def run(e):
        return _stream(e, None, lambda b: x[b], 6, azimuth=0.9, elevation=-0.3, spread=0.7, source_id="d")

    y1, y2 = run(Encoder(ORDER, SR)), run(Encoder(ORDER, SR))
    assert np.array_equal(y1, y2)                       # bit-identical for the same seed
    y3 = run(Encoder(ORDER, SR, rngf=RngFactory(99)))
    assert np.array_equal(y1[0], y3[0])                 # W never decorrelated
    assert np.max(np.abs(y1[1:] - y3[1:])) > 1e-3       # different decorrelators -> different diffuse part


def test_ambisonic_emissions_pass_through(rng):
    enc = Encoder(ORDER, SR)
    foa = rng.standard_normal((4, 100))
    out = enc.encode(Emission(foa, is_ambisonic=True), 100)
    np.testing.assert_array_equal(out[:4], foa)
    assert not out[4:].any()
    big = rng.standard_normal((25, 100))
    np.testing.assert_array_equal(enc.encode(Emission(big, is_ambisonic=True), 80), big[:16, :80])


def test_short_and_long_signals_are_fitted_to_block(rng):
    enc = Encoder(ORDER, SR)
    out = enc.encode(Emission(np.ones(10), source_id="x"), 32)
    assert out.shape == (N, 32) and not out[:, 10:].any()
    out = enc.encode(Emission(np.ones(64), source_id="y"), 32)
    assert out.shape == (N, 32)


# ---------------------------------------------------------------------------
# loudspeaker decoders
# ---------------------------------------------------------------------------
def test_left_source_is_louder_on_the_left():
    for layout in ("stereo", "binaural"):
        imp = np.zeros(2048)
        imp[0] = 1.0
        for az, louder in ((90.0, 0), (-90.0, 1)):
            d = make_decoder(layout, ORDER)
            y = d.process(np.outer(sh_real(ORDER, az * DEG, 0.0), imp))
            e = np.sum(y ** 2, axis=1)
            # stereo panning law / head shadow: well over 6 dB at the ipsilateral side
            assert 10 * np.log10(e[louder] / e[1 - louder]) > 6.0, (layout, az)


def test_stereo_panning_law():
    D = make_decoder("stereo", ORDER).matrix
    g = lambda a, e=0.0: D @ sh_real(ORDER, a * DEG, e * DEG)
    p = lambda a: 10 * np.log10(np.sum(g(a) ** 2))
    # centre: equal gains (sin/cos constant-power law: each channel -3 dB re the pair's power)
    l, r = g(0.0)
    assert abs(l - r) < 1e-12 and abs(20 * np.log10(l) - (p(0.0) - 3.01)) < 0.05
    # constant power across the frontal image within 0.5 dB (third-order fit of the law)
    assert max(abs(p(a) - p(0.0)) for a in (10.0, 20.0, 30.0, 45.0)) < 0.5
    # +-30 deg loudspeaker positions sit well to one side
    l, r = g(30.0)
    assert 20 * np.log10(l / r) > 8.0
    # +-90 deg: hard left (> 30 dB channel separation)
    l, r = g(90.0)
    assert 20 * np.log10(abs(l) / max(abs(r), 1e-9)) > 30.0
    # the designed gentle rear attenuation: ~3 dB behind, half of it at the sides
    assert -3.6 < p(180.0) - p(0.0) < -2.4
    assert -2.3 < p(90.0) - p(0.0) < -0.8
    # symmetric
    np.testing.assert_allclose(g(40.0), g(-40.0)[::-1], atol=1e-12)


@pytest.mark.parametrize("layout", ["5.1", "7.1", "7.1.4"])
def test_channel_order_matches_wave_format_extensible(layout):
    d = make_decoder(layout, ORDER)
    names = d.channel_names
    bits = [WAVE_BITS[NAME_TO_BIT[layout][nm]] for nm in names]
    # channels must be in increasing speaker-bit order, and the mask is exactly those bits
    assert bits == sorted(bits)
    assert d.channel_mask == sum(bits) == CHANNEL_MASKS[layout] == channel_mask_for(layout)
    assert names[:4] == ["L", "R", "C", "LFE"]
    if layout == "5.1":
        assert names == ["L", "R", "C", "LFE", "Ls", "Rs"]
    if layout == "7.1":
        assert names == ["L", "R", "C", "LFE", "Lrs", "Rrs", "Lss", "Rss"]
    if layout == "7.1.4":
        assert names[8:] == ["Ltf", "Rtf", "Ltr", "Rtr"]


def test_smpte_order_option():
    d = make_decoder("7.1", ORDER, order_style="smpte")
    assert d.channel_names == ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs"]
    # SMPTE order is not the WAVE mask bit order, so the decoder does not claim that mapping
    assert d.channel_mask == 0
    w = make_decoder("7.1", ORDER)
    for nm in w.channel_names:
        np.testing.assert_allclose(d.matrix[d.channel_names.index(nm)], w.matrix[w.channel_names.index(nm)])
    d4 = make_decoder("7.1.4", ORDER, order_style="smpte")
    assert d4.channel_names[4:8] == ["Lss", "Rss", "Lrs", "Rrs"]


@pytest.mark.parametrize("layout", ["5.1", "7.1", "7.1.4"])
def test_front_source_lands_in_the_front_and_lfe_is_empty(layout):
    d = make_decoder(layout, ORDER)
    g = d.matrix @ sh_real(ORDER, 0.0, 0.0)
    nm = d.channel_names
    e = g ** 2
    front = sum(e[nm.index(c)] for c in ("L", "R", "C"))
    # >= 80 % of the energy in L/C/R (third-order max-rE lobe; 7.1.4 also feeds Ltf/Rtf ~ 6 % each)
    assert front / e.sum() > 0.8
    assert e[nm.index("C")] == max(e)
    assert abs(g[nm.index("L")] - g[nm.index("R")]) < 1e-9
    assert not d.matrix[nm.index("LFE")].any()             # LFE empty unless lfe_gain > 0
    # a source at the left side is mostly on left loudspeakers
    g = d.matrix @ sh_real(ORDER, 90 * DEG, 0.0)
    left = [i for i, c in enumerate(nm) if c.startswith("L") and c != "LFE"]
    assert np.sum(g[left] ** 2) / np.sum(g ** 2) > 0.85


def test_lfe_is_a_120hz_linkwitz_riley_lowpass():
    d = make_decoder("5.1", ORDER, lfe_gain=1.0)
    lfe = d.channel_names.index("LFE")
    t = np.arange(SR) / SR
    lev = {}
    for f in (30.0, 120.0, 1000.0):
        x = np.sin(2 * math.pi * f * t)
        d.reset()
        y = d.process(np.outer(sh_real(ORDER, 0.0, 0.0), x))[lfe, SR // 2:]
        lev[f] = 20 * np.log10(np.sqrt(2.0) * np.sqrt(np.mean(y ** 2)))
    # LR4: 0 dB in the pass band, -6 dB at the crossover, 24 dB/oct -> ~ -74 dB at 1 kHz (Linkwitz 1976)
    assert abs(lev[30.0]) < 0.3
    assert abs(lev[120.0] + 6.02) < 0.3
    assert lev[1000.0] < -60.0


def test_height_channels_light_up_for_elevated_sources():
    d = make_decoder("7.1.4", ORDER)
    nm = d.channel_names
    top = [nm.index(c) for c in ("Ltf", "Rtf", "Ltr", "Rtr")]
    for az in (0.0, 90.0, 180.0, -45.0):
        g_up = d.matrix @ sh_real(ORDER, az * DEG, 60 * DEG)
        g_hz = d.matrix @ sh_real(ORDER, az * DEG, 0.0)
        assert np.sum(g_up[top] ** 2) / np.sum(g_up ** 2) > 0.7
        # horizontal: < 20 % (-7 dB) in the tops; the third-order max-rE lobe (rE spread ~ 31 deg,
        # Zotter & Frank 2012) still reaches a top loudspeaker at the same azimuth and 45 deg up
        assert np.sum(g_hz[top] ** 2) / np.sum(g_hz ** 2) < 0.2
    # the energy vector of horizontal sources stays near the horizon (< 10 deg up)
    spk = {s[0]: s for s in SPEAKER_LAYOUTS["7.1.4"]}
    xyz = np.array([unit_vectors(spk[c][1] * DEG, spk[c][2] * DEG) if spk[c][1] is not None else np.zeros(3)
                    for c in nm])
    a = np.radians(np.arange(0, 360, 10.0))
    e = (d.matrix @ sh_real(ORDER, a, 0 * a)) ** 2
    rE = xyz.T @ e / e.sum(axis=0)
    assert np.max(np.degrees(np.arctan2(rE[2], np.hypot(rE[0], rE[1])))) < 10.0
    # straight up: the four tops share it equally (symmetrised AllRAD over the imaginary zenith)
    g = d.matrix @ sh_real(ORDER, 0.0, math.pi / 2)
    np.testing.assert_allclose(g[top[0]], g[top[1]], rtol=1e-9)
    np.testing.assert_allclose(g[top[2]], g[top[3]], rtol=1e-9)


@pytest.mark.parametrize("layout,el_min", [("5.1", -90.0), ("7.1", -90.0), ("7.1.4", -90.0), ("7.1.4", 0.0)])
def test_decoded_energy_is_independent_of_direction(layout, el_min):
    d = make_decoder(layout, ORDER)
    az, el = fibonacci_sphere(3000)
    keep = el >= el_min * DEG
    E = 10 * np.log10(np.sum((d.matrix @ sh_real(ORDER, az[keep], el[keep])) ** 2, axis=0))
    # AllRAD/EPAD-style decoders: total energy within +-1.5 dB of its mean (Zotter & Frank 2012, Fig. 7);
    # plain AllRAD reaches +-1.7 dB on 7.1.4, the energy-flattened refinement here stays within +-0.75 dB
    assert np.max(np.abs(E - E.mean())) < 0.75, (E.min(), E.max())
    # direction-averaged power is 1 (shared calibration of every loudspeaker decoder)
    az, el = fibonacci_sphere(3000)
    P = np.mean(np.sum((d.matrix @ sh_real(ORDER, az, el)) ** 2, axis=0))
    assert abs(10 * np.log10(P)) < 0.05


@pytest.mark.parametrize("layout", ["5.1", "7.1", "7.1.4"])
def test_speaker_decoders_are_left_right_symmetric(layout):
    d = make_decoder(layout, ORDER)
    spk = {s[0]: s for s in SPEAKER_LAYOUTS[layout]}
    nm = d.channel_names
    mirror = []
    for c in nm:
        if spk[c][1] is None or spk[c][1] == 0.0:
            mirror.append(nm.index(c))
        else:
            other = next(o for o in nm if spk[o][1] is not None and abs(spk[o][1] + spk[c][1]) < 1e-9
                         and spk[o][2] == spk[c][2])
            mirror.append(nm.index(other))
    for az, el in ((20.0, 10.0), (130.0, 50.0), (-100.0, -20.0)):
        g = d.matrix @ sh_real(ORDER, az * DEG, el * DEG)
        gm = d.matrix @ sh_real(ORDER, -az * DEG, el * DEG)
        np.testing.assert_allclose(g[mirror], gm, atol=1e-9)


def test_allrad_accepts_arbitrary_layouts():
    # a regular octagon + 4 tops: VBAP hull with imaginary nadir only
    az = list(range(0, 360, 45)) + [45, 135, 225, 315]
    el = [0] * 8 + [45] * 4
    D = allrad_matrix(ORDER, az, el)
    assert D.shape == (12, N) and np.all(np.isfinite(D))


def test_mono_foa_hoa_scaling(rng):
    x = rng.standard_normal(4096)
    for az, el in ((0.0, 0.0), (2.0, 0.5), (-1.0, -0.4)):
        bus = np.outer(sh_real(ORDER, az, el), x)
        mono = make_decoder("mono", ORDER).process(bus)
        foa = make_decoder("foa", ORDER).process(bus)
        hoa = make_decoder("hoa3", ORDER).process(bus)
        assert mono.shape == (1, 4096) and foa.shape == (4, 4096) and hoa.shape == (16, 4096)
        np.testing.assert_array_equal(mono[0], x)            # mono = W = the pressure at the listener
        np.testing.assert_array_equal(foa, hoa[:4])           # AmbiX FOA = first four ACN channels of HOA
        np.testing.assert_array_equal(foa[0], x)              # SN3D W is not attenuated (FuMa's -3 dB is not used)
        # SN3D plane wave: every directional channel is at most W, and the first-order vector has length W
        assert np.max(np.abs(hoa[1:])) <= np.max(np.abs(hoa[0])) + 1e-12
        np.testing.assert_allclose(np.sum(foa[1:] ** 2, axis=0), foa[0] ** 2, rtol=1e-12)
    assert make_decoder("foa", ORDER).channel_names == ["W", "Y", "Z", "X"]
    assert make_decoder("mono", ORDER).channel_mask == 0x4


def test_mono_loudness_matches_stereo_and_surround(rng):
    """Mono is W with unit gain: same power as the energy-normalised decoders for a diffuse field and
    within ~0.5 dB for a frontal source (BS.1770 sums channel powers, so loudness follows)."""
    n = acn_order_index(ORDER)
    Cdiff = np.diag(1.0 / (2 * n + 1))
    y0 = sh_real(ORDER, 0.0, 0.0)
    pm = None
    for layout in ("mono", "stereo", "5.1", "7.1.4"):
        D = make_decoder(layout, ORDER).matrix
        p_diff = 10 * np.log10(np.trace(D @ Cdiff @ D.T))
        p_front = 10 * np.log10(np.sum((D @ y0) ** 2))
        if pm is None:
            pm = (p_diff, p_front)
            assert abs(p_diff) < 1e-9 and abs(p_front) < 1e-9
        else:
            assert abs(p_diff - pm[0]) < 0.3, layout
            assert abs(p_front - pm[1]) < 1.5, layout


def test_layout_aliases_and_errors():
    assert normalize_layout("Headphones") == "binaural"
    assert normalize_layout("AmbiX") == "foa"
    assert normalize_layout("7.1.0") == "7.1"
    assert make_decoder("hoa2", ORDER).channels == 9
    with pytest.raises(ValueError):
        make_decoder("9.1.6", ORDER)
    with pytest.raises(ValueError):
        make_decoder("binaural", ORDER, method="nope")


def test_decoders_are_linear_and_block_size_independent(rng):
    total = 5000
    bus = rng.standard_normal((N, total))
    for layout in ("binaural", "5.1"):
        d1 = make_decoder(layout, ORDER, lfe_gain=0.5) if layout == "5.1" else make_decoder(layout, ORDER)
        whole = d1.process(bus)
        d2 = make_decoder(layout, ORDER, lfe_gain=0.5) if layout == "5.1" else make_decoder(layout, ORDER)
        parts = [d2.process(bus[:, s:s + n]) for s, n in ((0, 1), (1, 999), (1000, 1024), (2024, 2976))]
        np.testing.assert_allclose(np.concatenate(parts, axis=1), whole, atol=1e-11)
        d3 = make_decoder(layout, ORDER, lfe_gain=0.5) if layout == "5.1" else make_decoder(layout, ORDER)
        d4 = make_decoder(layout, ORDER, lfe_gain=0.5) if layout == "5.1" else make_decoder(layout, ORDER)
        half = d3.process(0.5 * bus) + d4.process(0.5 * bus)
        np.testing.assert_allclose(half, whole, atol=1e-11)


def test_zero_length_blocks_are_accepted():
    # rule: generators and DSP work for any block length, including an empty final block
    for layout in ("mono", "stereo", "5.1", "7.1.4", "foa", "binaural"):
        d = make_decoder(layout, ORDER, lfe_gain=1.0) if layout == "5.1" else make_decoder(layout, ORDER)
        assert d.process(np.zeros((N, 0))).shape == (d.channels, 0), layout
    enc = Encoder(ORDER, SR)
    assert enc.encode(Emission(np.zeros(0), spread=0.5, source_id="e"), 0).shape == (N, 0)
    assert enc.encode_block([Emission(np.zeros(0), spread=0.5, source_id="f")], 0)["rain_near"].shape == (N, 0)


def test_lfe_state_does_not_go_denormal():
    d = make_decoder("5.1", ORDER, lfe_gain=1.0)
    d.process(np.outer(sh_real(ORDER, 0, 0), np.ones(1024)))
    for _ in range(200):
        d.process(np.zeros((N, 1024)))
    assert not np.any((d._lfe_zi != 0) & (np.abs(d._lfe_zi) < 1e-300))


# ---------------------------------------------------------------------------
# HRTF model and binaural decoder
# ---------------------------------------------------------------------------
def _itd(H, freqs, lo, hi):
    """Interaural delay (s, right minus left arrival) from the phase slope of H_L conj(H_R) in [lo, hi)."""
    k = (freqs >= lo) & (freqs < hi)
    ph = np.unwrap(np.angle(H[..., 0, k] * np.conj(H[..., 1, k])), axis=-1)
    w = 2 * math.pi * freqs[k]
    return np.array([np.polyfit(w, p, 1)[0] for p in np.atleast_2d(ph)])


def test_spherical_head_itd_woodworth_and_kuhn():
    freqs = np.fft.rfftfreq(8192, 1.0 / SR)
    az = np.radians(np.arange(0, 181, 2.0))
    H = spherical_head_response(az, np.zeros_like(az), freqs, pinna=0.0, torso=0.0)
    a_c = HEAD_RADIUS_M / SPEED_OF_SOUND
    itd_hf = _itd(H, freqs, 8000.0, 16000.0)
    itd_lf = _itd(H, freqs, 100.0, 500.0)
    # Woodworth ray model, ears at +-100 deg: max (a/c)(cos 20deg + pi/2) = 0.64 ms at 80 deg azimuth;
    # canonical Woodworth with ears at +-90 deg gives (a/c)(1 + pi/2) = 0.656 ms ("about 0.65 ms")
    assert abs(itd_hf.max() - a_c * (math.cos(20 * DEG) + math.pi / 2)) < 10e-6
    assert abs(itd_hf.max() - 0.656e-3) < 0.03e-3
    # low frequencies: Kuhn (1977) ITD ~ 3 (a/c) sin(az) = 0.765 ms at 90 deg (the head-shadow filter's phase)
    i90 = int(np.argmin(np.abs(az - math.pi / 2)))
    assert abs(itd_lf[i90] - 3 * a_c) < 0.08e-3
    assert 1.1 < itd_lf[i90] / itd_hf[i90] < 1.5
    # zero in the median plane, antisymmetric left/right
    assert abs(itd_hf[0]) < 1e-7 and abs(itd_hf[-1]) < 1e-7
    Hm = spherical_head_response(-az, np.zeros_like(az), freqs, pinna=0.0, torso=0.0)
    np.testing.assert_allclose(_itd(Hm, freqs, 8000.0, 16000.0), -itd_hf, atol=1e-7)


def test_ild_grows_with_frequency_and_front_back_differ():
    hr = BuiltinHRTF(SR, 256)
    ir = hr.hrirs(np.radians([90.0, 0.0, 180.0, 30.0, 150.0]), np.zeros(5))
    ild = [float(np.subtract(*_band_db(ir[0], SR, lo, hi, 4096))) for lo, hi in
           ((200, 800), (1000, 3000), (5000, 12000))]
    # spherical-head shadow (Brown & Duda 1998, Fig. 3; Rayleigh): small at low f, > 10 dB above 5 kHz
    assert ild[0] < 4.0 and ild[0] < ild[1] < ild[2] and ild[2] > 10.0
    # median plane: no ILD; front and back differ in spectrum: the rear pinna shadow takes > 2 dB off
    # 12-16 kHz behind (Blauert 1997, directional bands; Brown & Duda 1998 rear low-pass)
    front, back = _band_db(ir[1], SR, 12000, 16000, 4096), _band_db(ir[2], SR, 12000, 16000, 4096)
    assert abs(front[0] - front[1]) < 1e-9
    assert front[0] - back[0] > 2.0
    # front/back confusable pair (30 deg vs 150 deg, same lateral angle): spectra differ by > 1 dB rms
    s30 = 20 * np.log10(np.abs(np.fft.rfft(ir[3, 0], 4096)) + 1e-9)
    s150 = 20 * np.log10(np.abs(np.fft.rfft(ir[4, 0], 4096)) + 1e-9)
    k = slice(int(2000 / SR * 4096), int(14000 / SR * 4096))
    assert np.sqrt(np.mean((s30[k] - s150[k]) ** 2)) > 1.0


def test_builtin_hrtf_is_diffuse_field_equalised():
    hr = BuiltinHRTF(SR, 256)
    _, _, w, ir = hr.reference()
    H = np.fft.rfft(ir, 1024, axis=-1)
    f = np.fft.rfftfreq(1024, 1.0 / SR)
    p = np.einsum("p,pef->ef", w / w.sum(), np.abs(H) ** 2)
    k = (f > 100) & (f < 16000)
    # direction-averaged power per ear = 1/2 (matches the loudspeaker decoders' unit total), 1/3-oct smoothing
    assert np.max(np.abs(10 * np.log10(p[:, k] / 0.5))) < 1.5


@pytest.fixture(scope="module")
def binaural_ref():
    hr = get_hrtf("builtin", SR, 256)
    az, el, w, ir = hr.reference()
    nfft = 2048
    freqs = np.fft.rfftfreq(nfft, 1.0 / SR)
    Href = np.fft.rfft(ir, nfft, axis=-1) * np.exp(-2j * math.pi * freqs * 32 / SR)[None, None, :]
    return az, el, w / w.sum(), Href, freqs, nfft


def _decoded_hrtfs(d, az, el, nfft):
    Hs = np.fft.rfft(d.filters, nfft, axis=-1)                      # (2, N, F)
    return np.einsum("ekf,kp->pef", Hs, sh_real(d.order, az, el))   # (P, 2, F)


BANDS = ((100, 500), (500, 1500), (2000, 4000), (4000, 8000), (8000, 16000))


def test_magls_matches_the_hrtf_set_and_preserves_the_diffuse_field(binaural_ref):
    az, el, w, Href, f, nfft = binaural_ref
    res = {}
    for method in ("magls", "vls"):
        d = make_decoder("binaural", ORDER, method=method)
        Hd = _decoded_hrtfs(d, az, el, nfft)
        err = []
        for lo, hi in BANDS:
            k = (f >= lo) & (f < hi)
            ed = 10 * np.log10(np.mean(np.abs(Hd[..., k]) ** 2, axis=-1) / np.mean(np.abs(Href[..., k]) ** 2, axis=-1))
            err.append(float(np.median(np.abs(ed))))
        res[method] = err
        # diffuse-field power per ear and interaural coherence follow the reference set
        # (covariance constraint, Zaunschirm, Schoerkhuber & Hoeldrich 2018, JASA 143(6):3616)
        pd = np.einsum("p,pef->ef", w, np.abs(Hd) ** 2)
        pr = np.einsum("p,pef->ef", w, np.abs(Href) ** 2)
        icd = np.abs(np.einsum("p,pf->f", w, Hd[:, 0] * np.conj(Hd[:, 1]))) / np.sqrt(pd[0] * pd[1])
        icr = np.abs(np.einsum("p,pf->f", w, Href[:, 0] * np.conj(Href[:, 1]))) / np.sqrt(pr[0] * pr[1])
        for lo, hi in BANDS:
            k = (f >= lo) & (f < hi)
            assert abs(10 * np.log10(np.mean(pd[:, k]) / np.mean(pr[:, k]))) < 0.5, (method, lo)
            assert abs(np.mean(icd[k]) - np.mean(icr[k])) < 0.1, (method, lo)
    # MagLS: band-level error vs the full HRTF set stays ~1 dB above the cutoff (Schoerkhuber et al. 2018:
    # ~1-2 dB at N = 3), clearly better than virtual loudspeakers with max-rE
    assert max(res["magls"]) < 1.2, res["magls"]
    assert np.mean(res["magls"][2:]) < 0.6 * np.mean(res["vls"][2:]), res


def test_binaural_itd_and_ild_follow_the_model(binaural_ref):
    az, el, w, Href, f, nfft = binaural_ref
    d = make_decoder("binaural", ORDER)
    dirs = np.radians([[90.0, 0.0], [45.0, 0.0], [-60.0, 20.0], [135.0, -10.0]])
    hr = get_hrtf("builtin", SR, 256)
    Hm = np.fft.rfft(hr.hrirs(dirs[:, 0], dirs[:, 1]), nfft, axis=-1)
    Hd = _decoded_hrtfs(d, dirs[:, 0], dirs[:, 1], nfft)
    # LF ITD (below the MagLS cutoff, LS fit keeps the phase): within 80 us (order-3 truncation, kr < N)
    np.testing.assert_allclose(_itd(Hd, f, 150, 700), _itd(Hm, f, 150, 700), atol=80e-6)
    # ITD of the decoded 90 deg source: 0.6-0.85 ms (spherical head, Woodworth / Kuhn)
    assert 0.6e-3 < _itd(Hd[:1], f, 150, 700)[0] < 0.85e-3
    # HF ILD (MagLS keeps magnitudes; the covariance constraint mixes the ears a little): rms error over
    # the whole reference set < 2 dB in 4-12 kHz (Zaunschirm et al. 2018 report ~1-2 dB), and a source
    # at 90 deg keeps > 12 dB
    Hall = _decoded_hrtfs(d, az, el, nfft)
    tdom = lambda H: np.fft.irfft(H, nfft, axis=-1)
    ild_d = np.diff(_band_db(tdom(Hall), SR, 4000, 12000)[:, ::-1], axis=-1)[:, 0]
    ild_r = np.diff(_band_db(tdom(Href), SR, 4000, 12000)[:, ::-1], axis=-1)[:, 0]
    assert np.sqrt(np.sum(w * (ild_d - ild_r) ** 2)) < 2.0
    lv = _band_db(tdom(Hd[:1]), SR, 4000, 12000)[0]
    assert lv[0] - lv[1] > 12.0


def test_binaural_itd_does_not_depend_on_sample_rate():
    """The head is the same at every rate: the default HRIR length scales with the rate (~5.3 ms),
    so the decoded low-frequency ITD of a 90 deg source agrees between 48 and 96 kHz."""
    nfft = 1 << 15

    def lf_itd(sr):
        d = make_decoder("binaural", ORDER, sr=sr)
        y = d.process(np.outer(sh_real(ORDER, math.pi / 2, 0.0), np.r_[1.0, np.zeros(4095)]))
        f = np.fft.rfftfreq(nfft, 1.0 / sr)
        return _itd(np.fft.rfft(y, nfft, axis=-1), f, 150, 700)[0], d

    itd48, d48 = lf_itd(48_000)
    itd96, d96 = lf_itd(96_000)
    assert d96.hrtf.length == 2 * d48.hrtf.length == 512
    # 20 us ~ 2 samples at 96 kHz; a 256-tap HRIR at 96 kHz lost ~55 us of the head-shadow phase delay
    assert abs(itd96 - itd48) < 20e-6, (itd48, itd96)


def test_binaural_filters_are_cached_and_latency_reported():
    import time
    d1 = make_decoder("binaural", ORDER)
    t = time.perf_counter()
    d2 = make_decoder("binaural", ORDER)
    assert time.perf_counter() - t < 0.05
    assert d1.filters is d2.filters
    assert d1.latency == 32 + 48          # pre-delay + the model's 1 ms bulk delay
    # independent overlap-add state per decoder
    y = d1.process(np.outer(sh_real(ORDER, 0, 0), np.r_[1.0, np.zeros(511)]))
    assert np.any(y) and not np.any(d2._tail)
    # the impulse response of a frontal source peaks at the reported latency (all bands aligned: MagLS
    # runs on the bulk-delay-free set), within the head's 2-sample frontal path difference
    yy = d2.process(np.outer(sh_real(ORDER, 0, 0), np.r_[1.0, np.zeros(1023)]))
    assert abs(int(np.argmax(np.abs(yy[0]))) - d1.latency) <= 3
    from scipy.signal import butter, sosfilt
    lo = sosfilt(butter(4, 1200, fs=SR, output="sos"), yy[0])
    hi = sosfilt(butter(4, 4000, btype="high", fs=SR, output="sos"), yy[0])
    cen = lambda x: np.sum(np.arange(x.size) * x * x) / np.sum(x * x)
    assert abs(cen(hi) - d1.latency) < 12          # high band not ~1 ms early (the zero-phase MagLS pitfall)
    assert cen(lo) > cen(hi) - 5


def test_sofa_without_reader_falls_back_gracefully(monkeypatch, tmp_out):
    path = tmp_out / "fake.sofa"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
    monkeypatch.setitem(sys.modules, "h5py", None)       # import h5py -> ImportError
    monkeypatch.setitem(sys.modules, "netCDF4", None)
    with pytest.raises(ImportError, match="h5py"):
        load_sofa(str(path))
    with pytest.warns(RuntimeWarning, match="built-in"):
        hr = get_hrtf(str(path), SR)
    assert isinstance(hr, BuiltinHRTF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = make_decoder("binaural", ORDER, hrtf=str(path))
    assert d.channels == 2 and np.all(np.isfinite(d.filters))
    with pytest.raises(ImportError):
        get_hrtf(str(path), SR, fallback=False)
    with pytest.raises(FileNotFoundError):
        get_hrtf(str(tmp_out / "missing.sofa"), SR)


def test_sofa_roundtrip_with_h5py(tmp_out):
    h5py = pytest.importorskip("h5py")
    hr = BuiltinHRTF(SR, 256)
    az, el, _, ir = hr.reference()
    path = str(tmp_out / "model.sofa")
    with h5py.File(path, "w") as fh:
        fh["Data.IR"] = ir
        fh["Data.SamplingRate"] = np.array([float(SR)])
        pos = fh.create_dataset("SourcePosition", data=np.stack([np.degrees(az), np.degrees(el), np.ones_like(az)], 1))
        pos.attrs["Type"] = b"spherical"
        pos.attrs["Units"] = b"degree, degree, metre"
    s = load_sofa(path, sr=SR, diffuse_eq=False)
    assert s.ir.shape[0] == az.shape[0]
    np.testing.assert_allclose(s.hrirs(np.array([0.3]), np.array([0.1]))[0], hr.hrirs(0.3, 0.1)[0], atol=0.2)


# ---------------------------------------------------------------------------
# whole chain
# ---------------------------------------------------------------------------
def test_chain_output_is_clean(rng):
    """Encoder -> decoder on noise sources: finite, no DC, no denormals, no clicks; more sources louder."""
    layouts = ("stereo", "binaural", "5.1", "7.1.4", "foa")
    blocks = 24
    for layout in layouts:
        sig = rng.standard_normal((2, blocks, BLOCK))
        power = []
        for which in ((0,), (1,), (0, 1)):
            enc, d = Encoder(ORDER, SR), make_decoder(layout, ORDER)
            y = []
            for b in range(blocks):
                ems = [Emission(0.1 * sig[0, b], azimuth=0.5, elevation=0.2, spread=0.3, source_id="a"),
                       Emission(0.1 * sig[1, b], azimuth=-2.0, elevation=0.5, spread=0.9, source_id="b")]
                y.append(d.process(enc.encode_block([ems[i] for i in which], BLOCK)["rain_near"]))
            y = np.concatenate(y, axis=1)
            assert np.all(np.isfinite(y)), layout
            tiny = (y != 0) & (np.abs(y) < np.finfo(np.float64).tiny)
            assert not tiny.any(), layout
            # no DC: the mean of a zero-mean noise stays within its 4-sigma sampling bound (4 rms / sqrt(n))
            body = y[:, 4 * BLOCK:]
            rms = np.sqrt(np.mean(body ** 2, axis=1))
            assert np.all(np.abs(np.mean(body, axis=1)) <= 4.0 * rms / math.sqrt(body.shape[1])), layout
            assert _boundary_jump_ratio(y, BLOCK) < 1.6, layout
            power.append(float(np.sum(body ** 2)))
        # independent sources add in power (more rain is louder): P(a + b) = P(a) + P(b) within 0.3 dB
        # (the residual is the finite-length cross term, ~ 1/sqrt(n))
        assert abs(10 * np.log10(power[2] / (power[0] + power[1]))) < 0.3, layout
        assert power[2] > max(power[0], power[1])


@pytest.mark.slow
def test_render_array_binaural_and_surround_smoke():
    from raingen.api import render_array
    for layout in ("binaural", "7.1.4"):
        audio, _ = render_array("steady-rain-leaves-night", seconds=0.5, layout=layout, seed=7)
        assert audio.shape[0] == (2 if layout == "binaural" else 12)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0
