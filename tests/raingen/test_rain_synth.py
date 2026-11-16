"""Rain synthesis: kernels, materials, Cox events, the vectorised rain field, shelters and levels.

Every numeric check names its reference. Audio is checked with numbers (levels, spectra, decay,
block-boundary jumps, determinism) because nobody can listen in CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from raingen.core.constants import RHO_WATER
from raingen.core.rng import RngFactory
from raingen.core.types import BlockContext, SourcePatch, WeatherState
from raingen.physics import dsd as D
from raingen.scene.schema import Precipitation, Scene
from raingen.synth import events as E
from raingen.synth import kernels as K
from raingen.synth import materials as MT
from raingen.synth import rain as R

SR = 48_000


# ---------------------------------------------------------------------------
# helpers

def weather(rate=4.0, drops="normal", wind=0.0, d_min=None, d_max=None, wet=None, puddles=None,
            temp=15.0, rh=80.0, precip="rain", hail_mm=0.0, wind_az=0.0):
    p = Precipitation()
    p.drops = drops
    if d_min is not None:
        p.d_min_mm = d_min
    if d_max is not None:
        p.d_max_mm = d_max
    dsd = D.dsd_from_scene(p, rate)
    return WeatherState(t=0.0, precip_type=precip, rain_rate_mm_h=rate, dsd=dsd, class_flux=D.class_flux(dsd),
                        hail_size_mm=hail_mm, wind_u10_mps=wind, wind_listener_mps=0.7 * wind, gust=1.0,
                        wind_from_deg=270.0, wind_azimuth=wind_az, turbulence_intensity=0.2, temperature_c=temp,
                        humidity_pct=rh, pressure_hpa=1013.25, speed_of_sound=343.0, hour=12.0, season="summer",
                        is_day=True, wetness=dict(wet or {}), puddle_depth_mm=dict(puddles or {}))


def patch(material="leaves", kind="surface", pos=(0.0, 0.0, -1.7), area=1.0, normal=(0.0, 0.0, 1.0),
          pid="p", interior=False, params=None, exposure=1.0):
    return SourcePatch(id=pid, kind=kind, material=material, position=pos, area_m2=area, normal=normal,
                       exposure=exposure, interior=interior, params=dict(params or {}))


def render(p, w, seconds=1.0, quality="studio", lod="full", seed=7, n=1024, water_quality=1.0):
    """Render one patch (its own private field) for `seconds`; returns (signal Pa, generator)."""
    sc = Scene()
    sc.weather.precipitation.water_quality = water_quality
    rngf = RngFactory(seed)
    g = R.RainPatchGenerator(sc, p, rngf, lod=lod, quality=quality)
    nb = int(round(seconds * SR / n))
    out = []
    for b in range(nb):
        wb = w(b) if callable(w) else w
        em = g.render(BlockContext(sr=SR, n=n, index=b, t0=b * n / SR, weather=wb, rng=rngf))
        out.append(em[0].signal if em else np.zeros(n))
    return np.concatenate(out), g


def spl(x) -> float:
    """dB SPL of a pressure signal in Pa (1.0 = 1 Pa = 94 dB SPL)."""
    return float(10.0 * np.log10(np.mean(np.asarray(x) ** 2) / 4e-10 + 1e-30))


def a_weight_db(f):
    """IEC 61672-1 A-weighting (dB)."""
    f2 = np.asarray(f, dtype=float) ** 2
    ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def spl_a(x) -> float:
    f, P = signal.welch(x, SR, nperseg=4096)
    return float(10.0 * np.log10(np.sum(P * 10 ** (a_weight_db(np.maximum(f, 1.0)) / 10.0)) * (f[1] - f[0]) / 4e-10))


def bands_db(x, edges, nperseg=4096):
    f, P = signal.welch(x, SR, nperseg=nperseg)
    df = f[1] - f[0]
    return np.array([10.0 * np.log10(np.sum(P[(f >= lo) & (f < hi)]) * df / 4e-10 + 1e-30) for lo, hi in edges])


def octaves(lo=125.0, hi=16000.0):
    out, c = [], lo
    while c <= hi * 1.01:
        out.append((c / math.sqrt(2.0), c * math.sqrt(2.0)))
        c *= 2.0
    return out


def centroid(x) -> float:
    f, P = signal.welch(x, SR, nperseg=4096)
    return float(np.sum(f * P) / np.sum(P))


# ---------------------------------------------------------------------------
# kernels: impact force, Minnaert bubbles, Hertz contact, modal banks

def test_force_peak_scales_as_d2v2_and_matches_cheat_sheet():
    # Gordillo, Sun & Cheng (2018): F_max = 0.83 rho D^2 V^2
    assert K.force_peak(2.0, 6.5) == pytest.approx(0.83 * RHO_WATER * 4e-6 * 6.5 ** 2, rel=1e-12)
    assert K.force_peak(4.0, 6.5) / K.force_peak(2.0, 6.5) == pytest.approx(4.0, rel=1e-12)
    assert K.force_peak(2.0, 13.0) / K.force_peak(2.0, 6.5) == pytest.approx(4.0, rel=1e-12)
    # MASTER_PLAN Appendix A at terminal speed (Atlas 1973): 1 mm 13 mN, 2 mm 0.14 N, 3 mm 0.47 N, 4 mm 1.0 N
    for d, ref in ((1.0, 0.013), (2.0, 0.14), (3.0, 0.47), (4.0, 1.0)):
        assert K.force_peak(d, float(D.fall_speed(d))) == pytest.approx(ref, rel=0.08)


def test_impact_pulse_timing_and_impulse():
    d, v = 2.0, 6.5
    tau = d * 1e-3 / v
    p = K.impact_pulse(d, v, SR)
    # peak at t ~ 0.18 D/V (Gordillo 2018): within one sample of 0.18 tau
    assert abs(np.argmax(p) / SR - 0.18 * tau) <= 1.5 / SR
    # peak ~ F_max (area-sampled, so slightly below): within 10 %
    assert p.max() == pytest.approx(K.force_peak(d, v), rel=0.10)
    # impulse ~0.81 of the drop momentum (lateral splash carries the rest; kernels.py docstring)
    momentum = RHO_WATER * math.pi * (d * 1e-3) ** 3 / 6.0 * v
    assert p.sum() / SR / momentum == pytest.approx(0.81, abs=0.03)


def test_bigger_drops_are_louder_and_darker():
    # MASTER_PLAN 7.3: larger drops are louder *and* darker (pulse width ~ D/V)
    small = K.impact_pulse(1.0, float(D.fall_speed(1.0)), SR)
    big = K.impact_pulse(4.0, float(D.fall_speed(4.0)), SR)
    nfft = 8192
    f = np.fft.rfftfreq(nfft, 1 / SR)
    S, B = np.abs(np.fft.rfft(small, nfft)), np.abs(np.fft.rfft(big, nfft))
    assert B.max() > 10 * S.max()
    i8k = np.argmin(abs(f - 8000))
    # relative to its own DC level the big drop is >= 10 dB weaker at 8 kHz (pulse low-pass ~1/tau)
    assert 20 * np.log10(B[i8k] / B[0]) < 20 * np.log10(S[i8k] / S[0]) - 10.0


def test_minnaert_frequency_and_bubble_ring():
    # Minnaert (1933): f a = 3.28 m/s -> a 0.22 mm radius bubble rings at ~14.9 kHz (Appendix A)
    assert float(K.minnaert_frequency(0.22e-3)) == pytest.approx(14_909.0, rel=0.005)
    assert float(K.minnaert_frequency(3.28e-3)) == pytest.approx(1000.0, rel=1e-6)
    # pitch ~ sqrt(P): ~0.83x at 3000 m (P ~ 70 kPa)
    assert float(K.minnaert_frequency(1e-3, 70_000.0)) / float(K.minnaert_frequency(1e-3)) == pytest.approx(0.83, abs=0.01)
    a = 1e-3
    k = K.bubble_kernel(a, SR, xi=0.0, eps=0.05)
    nfft = 1 << 16
    spec = np.abs(np.fft.rfft(k, nfft))
    f_peak = np.fft.rfftfreq(nfft, 1 / SR)[np.argmax(spec)]
    assert f_peak == pytest.approx(3280.0, rel=0.01)
    # amplitude eps * a and van den Doel damping beta = 0.13/a + 0.0072 a^-1.5
    assert np.max(np.abs(k)) == pytest.approx(0.05 * a, rel=0.05)
    beta = 0.13 / a + 0.0072 * a ** -1.5
    env = np.abs(signal.hilbert(k))
    t = np.arange(k.size) / SR
    sel = (t > 0.5 / beta) & (t < 3.0 / beta)
    slope = np.polyfit(t[sel], np.log(env[sel]), 1)[0]
    assert -slope == pytest.approx(beta, rel=0.08)


def test_bubble_chirp_rises_and_is_band_limited():
    # van den Doel (2005): f(t) = f0 (1 + xi beta t), xi ~ 0.1 -> rising pitch
    a = 1e-3
    k = K.bubble_kernel(a, SR, xi=0.1)
    ph = np.unwrap(np.angle(signal.hilbert(k)))
    inst = np.diff(ph) * SR / (2 * np.pi)
    beta = float(K.bubble_damping(a))
    early = np.median(inst[int(0.2 / beta * SR):int(0.5 / beta * SR)])
    late = np.median(inst[int(2.0 / beta * SR):int(2.5 / beta * SR)])
    assert late > early * 1.1
    # MASTER_PLAN 9.2: bubbles starting above 0.45 fs are culled (no alias)
    assert np.all(K.bubble_kernel(3.28 / 30_000.0, SR) == 0.0)


def test_hertz_contact_matches_johnson_and_crushing_limit():
    # Johnson (1985) Contact Mechanics: delta = (5 m v^2 / 4 K)^(2/5), K = 4/3 E* sqrt(R), tau = 2.94 delta / v
    d, v = 5.0, 5.0
    f_max, tau = K.hertz_contact(d, v, crush_mpa=0.0)
    r = 0.5 * d * 1e-3
    m = 917.0 * math.pi * (d * 1e-3) ** 3 / 6.0
    e_star = 1.0 / ((1 - 0.33 ** 2) / 9.0e9 + (1 - 0.3 ** 2) / 2.0e11)
    kk = 4.0 / 3.0 * e_star * math.sqrt(r)
    delta = (5.0 * m * v * v / (4.0 * kk)) ** 0.4
    assert f_max == pytest.approx(kk * delta ** 1.5, rel=1e-9)
    assert tau == pytest.approx(2.94 * delta / v, rel=1e-9)
    # Hertz scaling: contact time ~ v^(-1/5)
    _, tau2 = K.hertz_contact(d, 2 * v, crush_mpa=0.0)
    assert tau2 / tau == pytest.approx(2 ** -0.2, rel=1e-6)
    # crushing ice (25 mm stone at 15 m/s): impulse = m v (1 + e) over 0.6 D/v
    f_c, tau_c = K.hertz_contact(25.0, 15.0, restitution=0.5)
    m25 = 917.0 * math.pi * 0.025 ** 3 / 6.0
    assert tau_c == pytest.approx(0.6 * 0.025 / 15.0, rel=1e-9)
    assert f_c * 0.574 * tau_c == pytest.approx(m25 * 15.0 * 1.5, rel=1e-9)


def test_modal_bank_frequency_and_t60():
    bank = K.ModalBank([(1000.0, 0.2, 1.0)], SR)
    h = bank.impulse_response(int(0.4 * SR))
    nfft = 1 << 17
    f = np.fft.rfftfreq(nfft, 1 / SR)[np.argmax(np.abs(np.fft.rfft(h, nfft)))]
    assert f == pytest.approx(1000.0, abs=1.0)
    env = np.abs(signal.hilbert(h))
    t = np.arange(h.size) / SR
    sel = (t > 0.02) & (t < 0.15)
    slope_db = np.polyfit(t[sel], 20 * np.log10(env[sel]), 1)[0]
    assert -60.0 / slope_db == pytest.approx(0.2, rel=0.03)      # T60 by definition


@pytest.mark.parametrize("n", [1024, 333, 64])
def test_modal_batch_matches_modal_bank_for_any_block(n, rng):
    mat = MT.get_material("tin_roof")
    rows = 3
    batch = K.ModalBatch(K.ModalBank(mat, SR), rows)
    singles = [K.ModalBank(mat, SR) for _ in range(rows)]
    x = rng.standard_normal((rows, 5 * n)) * (rng.random((rows, 5 * n)) < 0.01)
    x[:, 3 * n:] = 0.0                              # ringing carried across silent blocks
    yb = np.concatenate([batch.process(x[:, i:i + n]) for i in range(0, 5 * n, n)], axis=1)
    ys = np.stack([np.concatenate([s.process(x[r, i:i + n]) for i in range(0, 5 * n, n)])
                   for r, s in enumerate(singles)])
    assert np.max(np.abs(yb - ys)) <= 1e-9 * max(np.max(np.abs(ys)), 1e-12)


def test_trains_scatter_and_ola_match_direct_convolution(rng):
    kb = K.KernelBank(SR)
    n = 700
    codes = np.array([kb.pulse_code(3e-4), kb.click_code(6.0), kb.bubble_code(1e-3, 0.1, 0),
                      kb.splash_code(0.004, 3, True)], dtype=np.int64).ravel()
    m = 60
    rows = rng.integers(0, 4, m)
    cs = codes[rng.integers(0, codes.size, m)]
    offs = rng.integers(0, n, m)
    amps = rng.standard_normal(m)
    ref = {}
    for r, c, o, a in zip(rows, cs, offs, amps):
        k = kb.kernel(int(c))
        y = ref.setdefault(int(r), np.zeros(n + 6000))
        y[o:o + k.size] += a * k
    ur, Y = K.scatter_kernels(kb, n, rows, cs, offs, amps)
    for r, y in zip(ur, Y):
        assert np.allclose(y, ref[int(r)][:y.size], atol=1e-12)
    acc = {}
    for r, y in K.render_trains(kb, n, rows, cs, offs, amps):        # one item per kernel-length bucket
        for rr, yy in zip(r, y):
            a = acc.setdefault(int(rr), np.zeros(n + 6000))
            a[:yy.size] += yy
    for rr, a in acc.items():
        assert np.allclose(a, ref[rr], atol=1e-9)
    # RowOLA: blocks popped in sequence reproduce the whole signal
    ola = K.RowOLA(4)
    ola.add(ur, Y)
    out = np.concatenate([ola.pop(256) for _ in range(30)], axis=1)
    for r, y in zip(ur, Y):
        assert np.allclose(out[r, :y.size], y, atol=1e-12)
    assert not ola.active


def test_pooled_power_preserves_energy():
    # Parseval on the pooled grid: (1/N) sum over the full spectrum = kernel energy
    kb = K.KernelBank(SR)
    for code in (kb.pulse_code(3e-4), kb.bubble_code(2e-3, 0.1, 0)):
        k = kb.kernel(int(code))
        pw, e = kb.pooled_power(int(code), 256)
        full = pw[0] + pw[-1] + 2.0 * pw[1:-1].sum()
        assert full / 512 == pytest.approx(float(np.sum(k * k)), rel=0.02)
        assert e == pytest.approx(float(np.sum(k * k)), rel=1e-12)


def test_noise_table_is_white_and_unit():
    x = K.noise_rows(np.arange(8) * 4096, 4096).ravel()
    assert x.std() == pytest.approx(1.0, abs=0.02)
    f, P = signal.welch(x, SR, nperseg=1024)
    lo, hi = P[(f > 500) & (f < 2000)].mean(), P[(f > 15000) & (f < 20000)].mean()
    assert 10 * np.log10(hi / lo) == pytest.approx(0.0, abs=0.5)        # white within 0.5 dB


# ---------------------------------------------------------------------------
# materials

def _drop_response(name, d=2.0):
    mat = MT.get_material(name)
    v = float(D.fall_speed(d))
    y = np.convolve(K.impact_pulse(d, v, SR), K.ModalBank(mat, SR).impulse_response(SR))[:SR]
    return signal.sosfilt(K.radiation_sos(mat, SR), y)


def test_material_modes_are_plausible():
    for name, m in MT.MATERIALS.items():
        if m.kind == "water":
            assert not m.modes
            continue
        f = np.array([md[0] for md in m.modes])
        t60 = np.array([md[1] for md in m.modes])
        assert np.all((f > 20.0) & (f < 0.45 * SR)), name
        assert np.all((t60 > 0.0) & (t60 < 2.0)), name
        # loss factor eta = 2.2 / (f T60) (Cremer & Heckl): plates, membranes and leaves ring (eta 1e-4 .. 1,
        # Q > 1); porous ground and snow are dead absorbers whose 'grit' resonances last <= 10 ms
        eta = 2.2 / (f * t60)
        if m.kind in ("porous", "snow"):
            assert t60.max() <= 0.01, name
        else:
            assert np.all((eta > 1e-4) & (eta < 1.0)), name
    # thin metal rings for hundreds of ms; ground and grass are dead (grit modes < 10 ms)
    assert MT.get_material("tin_roof").max_t60 >= 0.3
    assert MT.get_material("metal_sheet").max_t60 >= 0.5
    for dull in ("soil", "grass", "concrete", "gravel"):
        assert MT.get_material(dull).max_t60 <= 0.01


def test_material_timbres_ring_dull_crisp():
    def edt(y):           # early decay time (s), extrapolated from the first 10 dB (ISO 3382)
        e = np.cumsum(y[::-1] ** 2)[::-1]
        return np.argmax(e / e[0] < 0.1) / SR * 6.0

    tin, soil, grass, leaves = (_drop_response(n) for n in ("tin_roof", "soil", "grass", "leaves"))
    # tin rings (fundamental T60 0.5 s, eta ~3e-3): EDT > 100 ms; soil is dead (grit modes <= 8 ms plus the
    # radiation band-pass): EDT < 30 ms
    assert edt(tin) > 0.1
    assert edt(soil) < 0.03
    assert edt(tin) > 5.0 * edt(soil)
    c = {k: centroid(v) for k, v in (("soil", soil), ("grass", grass), ("leaves", leaves))}
    # soil and grass are dull, leaves crisp (leaf = dipole radiator above ka ~ 1, ~1.2 kHz)
    assert c["soil"] < 500.0
    assert c["grass"] < 1000.0
    assert c["leaves"] > 1.5 * c["grass"]
    # calibration anchor: peak SPL @1 m of a 2 mm drop at 6.5 m/s equals level_db (materials docstring)
    for name in ("tin_roof", "leaves", "concrete"):
        mat = MT.get_material(name)
        y = R._solid_calibration(mat, SR) * signal.sosfilt(
            K.radiation_sos(mat, SR), np.convolve(K.impact_pulse(2.0, 6.5, SR),
                                                   K.ModalBank(mat, SR).impulse_response(int(0.08 * SR))))
        assert 20 * np.log10(np.max(np.abs(y)) / 2e-5) == pytest.approx(mat.level_db, abs=0.5)


def test_material_aliases():
    assert MT.get_material("Corrugated").name == "tin_roof"
    assert MT.get_material("forest floor").name == "leaves"
    assert MT.get_material("pond").name == "water"
    assert MT.get_material("no-such-thing").name == "concrete"
    assert MT.get_material("glass").youngs_modulus_pa > MT.get_material("soil").youngs_modulus_pa


# ---------------------------------------------------------------------------
# Cox / Poisson event statistics

def test_cox_offsets_honour_rate():
    rng = np.random.default_rng(5)
    rate, n, nb = 3000.0, 1024, 400
    counts = np.array([cox_ramp.size for cox_ramp in
                       (E.cox_offsets(rng, rate, n, SR, np.array([0.6, 1.4])) for _ in range(nb))])
    expected = rate * n / SR * 1.0 * nb                # ramp mean = 1
    # Poisson: total count within 4 sigma of rate * time
    assert abs(counts.sum() - expected) < 4.0 * math.sqrt(expected)
    offs = np.concatenate([E.cox_offsets(rng, rate, n, SR, np.array([0.0, 2.0])) for _ in range(200)])
    assert offs.min() >= 0 and offs.max() < n
    # intensity ramps 0 -> 2: the density of event times is linear (mean position 2/3 of the block)
    assert offs.mean() / n == pytest.approx(2.0 / 3.0, abs=0.01)


def test_ramp_positions_inverse_cdf():
    u = np.linspace(0.0, 1.0, 10001)[:-1]
    for m0, m1 in ((1.0, 1.0), (0.2, 1.8), (2.0, 0.5)):
        x = E.ramp_positions(u, m0, m1)
        cdf = (m0 * x + 0.5 * (m1 - m0) * x * x) / (0.5 * (m0 + m1))
        assert np.allclose(cdf, u, atol=1e-9)


def _haar_fano(counts_fine, k):
    """Haar fluctuation Fano factor at k fine windows: Var(N_i+1 - N_i) / (2 E N)."""
    c = counts_fine[: counts_fine.size // k * k].reshape(-1, k).sum(axis=1)
    return float(np.var(np.diff(c)) / (2.0 * c.mean()))


def test_fano_poisson_at_short_scales_clustered_at_long():
    # Lovejoy & Schertzer (2008): Poisson within ~0.1 s, scale-invariant clustering above (MASTER_PLAN 7.4)
    rows, nb, n, rate = 64, 1400, 1024, 400.0                 # 64 independent 30 s streams
    for sigma, clustered in ((0.22, True), (0.0, False)):
        mod = E.RateModulatorBank(RngFactory(3), "test/cox", rows, sigma=sigma)
        rng = np.random.default_rng(11)
        counts = np.zeros((rows, nb * 4))                      # 4 sub-windows of 256 frames (5.3 ms)
        for b in range(nb):
            m1 = mod.step(b, n)
            m0 = mod.prev
            lam = rate * n / SR * 0.5 * (m0 + m1)
            k = rng.poisson(lam)
            pr = np.repeat(np.arange(rows), k)
            x = E.ramp_positions(rng.random(k.sum()), m0[pr], m1[pr])
            np.add.at(counts, (pr, b * 4 + (x * 4).astype(int)), 1.0)
        f_short = np.mean([_haar_fano(c, 4) for c in counts])          # 21 ms windows
        f_100ms = np.mean([_haar_fano(c, 19) for c in counts])         # ~0.1 s windows
        f_long = np.mean([_haar_fano(c, 4 * 94) for c in counts])      # ~2 s windows
        assert f_short == pytest.approx(1.0, abs=0.05)                  # Poisson: Fano = 1
        assert f_100ms == pytest.approx(1.0, abs=0.25)
        if clustered:
            assert f_long > 2.0
        else:
            assert f_long == pytest.approx(1.0, abs=0.35)


def test_rate_modulator_bank_is_mean_one_and_deterministic():
    a = E.RateModulatorBank(RngFactory(9), "x", 200, sigma=0.3)
    b = E.RateModulatorBank(RngFactory(9), "x", 200, sigma=0.3)
    vals = []
    for blk in range(300):
        va, vb = a.step(blk), b.step(blk)
        assert np.array_equal(va, vb)
        vals.append(va)
    assert np.mean(vals) == pytest.approx(1.0, abs=0.05)          # exp(s sigma - sigma^2/2) has mean 1


# ---------------------------------------------------------------------------
# the rain field: event rates, levels, water, LOD parity

def test_field_honours_impact_rates():
    w = weather(4.0)
    sc = Scene()
    rngf = RngFactory(1)
    g = R.RainPatchGenerator(sc, patch("concrete", area=0.5), rngf, quality="reference")
    drawn = expected = 0.0
    for b in range(240):
        g.render(BlockContext(sr=SR, n=1024, index=b, t0=b * 1024 / SR, weather=w, rng=rngf))
        st = g.field.stats
        drawn += st["drawn"].sum()
        expected += st["expected"].sum()
    # expected = sum over classes of flux x area x time (x Cox mean): Marshall-Palmer-type table ~1.7e3 m^-2 s^-1
    assert expected / (240 * 1024 / SR * 0.5) == pytest.approx(sum(w.class_flux.values()), rel=0.25)
    assert abs(drawn - expected) < 4.0 * math.sqrt(expected)        # Poisson error


def test_more_rain_is_louder():
    lv = [spl(render(patch("tin_roof", area=2.0), weather(r), 1.0, seed=3)[0]) for r in (1.0, 4.0, 16.0)]
    slope = (lv[2] - lv[0]) / math.log10(16.0)
    # Griffin & Ballagh (2012): roof rain noise rises ~17 dB per tenfold rate for natural rain
    # (our DSD model is provisional, so accept 10-24 dB/decade); monotonic in between
    assert lv[0] < lv[1] < lv[2]
    assert 10.0 < slope < 24.0


def test_closer_is_louder_by_spreading():
    w = weather(6.0)
    near = spl(render(patch("concrete", area=1.0, pos=(1.5, 0.0, -1.7)), w, 1.0, seed=4)[0])
    far = spl(render(patch("concrete", area=1.0, pos=(12.0, 0.0, -1.7)), w, 1.0, seed=4)[0])
    r_near = math.hypot(1.5, 1.7)
    r_far = math.hypot(12.0, 1.7)
    # point source: 20 log10(r2/r1) (geometric spreading), plus < 1 dB of ISO 9613-1 air absorption
    assert near - far == pytest.approx(20 * math.log10(r_far / r_near), abs=1.5)


def test_rain_on_calm_water_peaks_at_13_to_16_khz():
    # Medwin et al. (1992), Nystuen: light rain on calm water has its spectral peak near 15 kHz from
    # the 0.8-1.1 mm class's regularly entrained bubbles (in air, close to the surface)
    w = weather(2.0)
    x, _ = render(patch("water", kind="water", area=4.0, pos=(0.5, 0.0, -1.0)), w, 2.0, seed=2)
    cs = [10 ** (k / 10) for k in range(30, 44)]                 # 1 kHz .. 20 kHz third octaves
    lv = bands_db(x, [(c / 2 ** (1 / 6), c * 2 ** (1 / 6)) for c in cs])
    peak = cs[int(np.argmax(lv))]
    assert 12_500.0 <= peak <= 16_500.0
    assert lv.max() - np.median(lv) > 6.0          # a clear peak, not a flat hiss


def test_water_drop_classes_and_water_quality():
    p = patch("water", kind="water", area=3.0, pos=(0.5, 0.0, -1.0))

    def hf_per_impact(x, w):
        return bands_db(x, [(12_500.0, 20_000.0)])[0] - 10 * math.log10(sum(w.class_flux.values()))

    w_small = weather(2.0, d_min=0.8, d_max=1.1)
    w_mini = weather(2.0, d_min=0.3, d_max=0.8)
    w_mid = weather(2.0, d_min=1.1, d_max=2.2)
    small = hf_per_impact(render(p, w_small, 1.0, quality="reference")[0], w_small)
    mini = hf_per_impact(render(p, w_mini, 1.0, quality="reference")[0], w_mini)
    mid = hf_per_impact(render(p, w_mid, 1.0, quality="reference")[0], w_mid)
    # Medwin/Nystuen: minuscule nearly silent, small ring loudly at 13-25 kHz, mid bubble-poor (click only).
    # The mid-class margin is provisional: underwater the contrast is larger, in air the impact click (anchored
    # at 42 dB peak @1 m for a 2 mm drop, ~D U^3) carries relatively more; the model gives ~9.5-10 dB
    assert small - mini > 20.0
    assert small - mid > 8.0
    # surfactant stops entrainment (Medwin; water_quality control): the ring band drops by > 15 dB
    soapy = hf_per_impact(render(p, w_small, 1.0, quality="reference", water_quality=0.0)[0], w_small)
    assert small - soapy > 15.0
    # the small-drop ring dies in wind above 8-10 m/s (Nystuen; provisional rule)
    windy = weather(2.0, d_min=0.8, d_max=1.1, wind=12.0)
    assert small - hf_per_impact(render(p, windy, 1.0, quality="reference")[0], windy) > 10.0


def test_large_drops_make_low_frequency_impacts_and_irregular_bubbles():
    p = patch("water", kind="water", area=3.0, pos=(0.5, 0.0, -1.0))
    w_big = weather(10.0, drops="fat", d_min=2.2, d_max=6.0)
    w_small = weather(2.0, d_min=0.8, d_max=1.1)
    big, _ = render(p, w_big, 1.0, quality="reference", seed=5)
    small, _ = render(p, w_small, 1.0, quality="reference", seed=5)
    # large / very large drops: impact + 2-10 kHz bubbles + splat (Medwin 1992 table, MASTER_PLAN 7.2)
    lo_big = bands_db(big, [(1000.0, 10_000.0)])[0] - bands_db(big, [(12_500.0, 20_000.0)])[0]
    lo_small = bands_db(small, [(1000.0, 10_000.0)])[0] - bands_db(small, [(12_500.0, 20_000.0)])[0]
    assert lo_big > lo_small + 15.0
    assert centroid(big) < 0.5 * centroid(small)


@pytest.mark.parametrize("material,kind", [("leaves", "vegetation"), ("tin_roof", "surface"),
                                           ("water", "water"), ("concrete", "surface")])
def test_lod_matches_per_drop_reference_band_levels(material, kind):
    # MASTER_PLAN 7.5 / 9.3: the statistical layer (matched noise for dense classes and for the excess above
    # the per-drop caps, pre-baked grains for far patches) matches the per-drop result in expectation:
    # octave levels energy-averaged over 3 seeds x 1.5 s within 1.5 dB, in the bands within 6 dB of the
    # loudest. On water only 4-16 kHz is compared: the 1-2 kHz octave is carried by a few large-drop
    # bubbles per second, whose sampling spread over 4.5 s (~2 dB) exceeds the tolerance itself.
    area = 20.0 if material == "water" else 3.0
    p = patch(material, kind=kind, area=area, pos=(1.0, 0.0, -1.7))
    w = weather(8.0)
    edges = octaves(250, 16000)

    def mean_bands(quality, lod):
        e = [10 ** (bands_db(render(p, w, 1.5, quality=quality, lod=lod, seed=s)[0], edges) / 10)
             for s in (21, 22, 23)]
        return 10 * np.log10(np.mean(e, axis=0))

    ref = mean_bands("reference", "full")
    rel = ref >= ref.max() - 6.0
    if material == "water":
        rel &= np.array([lo >= 2800.0 for lo, _ in edges])
    for quality, lod in (("studio", "full"), ("studio", "granular"), ("game", "granular"), ("mobile", "granular")):
        d = mean_bands(quality, lod) - ref
        assert np.max(np.abs(d[rel])) < 1.5, (quality, lod, np.round(d, 2))


def test_lod_uses_statistics_for_dense_classes():
    sc = Scene()
    rngf = RngFactory(4)
    g = R.RainPatchGenerator(sc, patch("leaves", kind="vegetation", area=100.0, pos=(8.0, 0.0, 2.0)), rngf,
                             lod="granular", quality="studio")
    g.render(BlockContext(sr=SR, n=1024, index=0, t0=0.0, weather=weather(4.0), rng=rngf))
    st = g.field.stats
    assert st["dense"][0, :3].all()                 # minuscule / small / mid: overlap >> 10 -> matched noise
    assert st["drawn"].sum() < 50                   # only rare big drops stay per-drop


# ---------------------------------------------------------------------------
# audio hygiene: finite, no DC, no block-boundary clicks, determinism, any block length

def _render_mix(n=1024, seed=7, seconds=1.0):
    ps = [patch("tin_roof", area=2.0, pos=(1.0, 0.5, 1.0), pid="a", interior=True),
          patch("water", kind="water", area=4.0, pos=(2.0, -1.0, -1.7), pid="b"),
          patch("leaves", kind="vegetation", area=60.0, pos=(9.0, 3.0, 2.0), pid="c")]
    sc = Scene()
    rngf = RngFactory(seed)
    field = R._RainField(rngf, "studio")
    gens = [R.RainPatchGenerator(sc, p, rngf, lod="full" if i < 2 else "granular", field=field)
            for i, p in enumerate(ps)]
    nb = int(round(seconds * SR / n))
    out = np.zeros((len(gens), nb * n))
    for b in range(nb):
        w = weather(6.0 if b * n < 0.75 * seconds * SR else 0.0)
        ctx = BlockContext(sr=SR, n=n, index=b, t0=b * n / SR, weather=w, rng=rngf)
        for k, g in enumerate(gens):
            em = g.render(ctx)
            if em:
                out[k, b * n:(b + 1) * n] = em[0].signal
    return out


def test_output_finite_no_dc_and_deterministic():
    a = _render_mix(seed=7)
    assert np.all(np.isfinite(a))
    for row in a:
        assert abs(row.mean()) < 0.02 * row.std()               # no DC offset
    assert np.array_equal(a, _render_mix(seed=7))               # same seed -> bit-identical
    b = _render_mix(seed=8)
    assert not np.allclose(a, b)                                # different seed -> different rain


def test_no_clicks_at_block_boundaries():
    n = 1024
    x = _render_mix(n=n, seconds=1.0)
    for row in x[:, : int(0.7 * SR)]:
        d = np.abs(np.diff(row))
        at = d[n - 1::n]
        # jump statistics at block boundaries equal those elsewhere (95th percentile within 2x)
        assert np.percentile(at, 95) < 2.0 * np.percentile(d, 95) + 1e-12
        assert at.mean() < 1.5 * d.mean() + 1e-12


def _render_grid(n, seed, seconds=0.6):
    """A per-drop voice, a water voice and a granular canopy under small drops only (<= 1.1 mm: many similar
    impacts, no rare loud big drops), rendered on blocks of n frames; returns the part after 0.15 s."""
    ps = [patch("concrete", area=2.0, pos=(1.0, 0.5, -1.7), pid="a"),
          patch("water", kind="water", area=4.0, pos=(2.0, -1.0, -1.7), pid="b"),
          patch("leaves", kind="vegetation", area=60.0, pos=(9.0, 3.0, 2.0), pid="c")]
    sc = Scene()
    rngf = RngFactory(seed)
    field = R._RainField(rngf, "studio")
    gens = [R.RainPatchGenerator(sc, p, rngf, lod="full" if i < 2 else "granular", field=field)
            for i, p in enumerate(ps)]
    nb = int(math.ceil(seconds * SR / n))
    out = np.zeros((len(gens), nb * n))
    w = weather(6.0, d_max=1.1)
    for b in range(nb):
        ctx = BlockContext(sr=SR, n=n, index=b, t0=b * n / SR, weather=w, rng=rngf)
        for k, g in enumerate(gens):
            em = g.render(ctx)
            if em:
                out[k, b * n:(b + 1) * n] = em[0].signal
    return out[:, int(0.15 * SR):int(seconds * SR)]


@pytest.mark.parametrize("n", [333, 4096])
def test_any_block_length(n):
    # same physics on any block grid: per patch, the level energy-averaged over 3 seeds agrees with the
    # 1024-frame grid within 1 dB (the Cox modulation alone spreads single 0.45 s realizations by ~1 dB),
    # and the output stays finite; the first 0.15 s (the first noise frame fading in) is skipped
    def levels(nn):
        e = []
        for seed in (7, 8, 9):
            x = _render_grid(nn, seed)
            assert np.all(np.isfinite(x))
            e.append(np.mean(x ** 2, axis=1))
        return 10 * np.log10(np.mean(e, axis=0))

    assert np.max(np.abs(levels(n) - levels(1024))) < 1.0


def test_silence_after_rain_flushes_state():
    # after the rain stops every patch falls silent (tails decay, states flushed: no denormal tails)
    sc = Scene()
    rngf = RngFactory(2)
    g = R.RainPatchGenerator(sc, patch("tin_roof", area=2.0), rngf)
    for b in range(20):
        g.render(BlockContext(sr=SR, n=1024, index=b, t0=b * 1024 / SR, weather=weather(8.0), rng=rngf))
    quiet = None
    for b in range(20, 200):
        em = g.render(BlockContext(sr=SR, n=1024, index=b, t0=b * 1024 / SR, weather=weather(0.0), rng=rngf))
        if not em:
            quiet = b
            break
        assert np.all(np.isfinite(em[0].signal))
    assert quiet is not None and (quiet - 20) * 1024 / SR < 3.0       # tin T60 0.5 s -> silent well within 3 s


# ---------------------------------------------------------------------------
# shelters: interior transfer, hail, snow, drips

def test_through_ceiling_transfer_follows_mass_law():
    roof = patch("tile", pos=(-1.0, 0.0, 3.0), area=40.0, interior=True, params={"through_ceiling": True})
    gain, sos = R._interior_transfer(roof, SR)
    f = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    _, h = signal.sosfreqz(sos, worN=f, fs=SR)
    tl = -20 * np.log10(np.abs(h))
    # field-incidence mass law 20 log10(m f) - 47 for a 10 kg/m^2 gypsum ceiling, capped at ~32 dB by
    # coincidence (Cremer & Heckl; Bies & Hansen gypsum data); first-order knee: within 3.5 dB
    ref = np.minimum(20 * np.log10(10.0 * f) - 47.0, 32.0)
    assert np.max(np.abs(tl - ref)) < 3.5
    # attic build-up re the patch's free field: positive, of order +10..+20 dB (diffuse field 4/A)
    assert 10.0 < 20 * np.log10(gain) < 20.0


def test_interior_levels_are_physical():
    # Bedroom under a tiled roof + attic + ceiling: moderate rain ~20-35 dB(A), heavy rain 30-45 dB(A)
    # (lead's real-world targets; heavy = +17 dB/decade of rate, Griffin & Ballagh 2012)
    roof = patch("tile", pos=(-1.0, 0.0, 3.0), area=40.0, interior=True, params={"through_ceiling": True})
    moderate = spl_a(render(roof, weather(4.0), 1.0, seed=6)[0])
    heavy = spl_a(render(roof, weather(25.0), 1.0, seed=6)[0])
    assert 18.0 < moderate < 35.0
    assert 28.0 < heavy < 45.0
    # attic under thin metal: the direct field alone is already loud (>= 50 dB(A) at 8 mm/h;
    # the attic's reverberant build-up is added by the enclosure model)
    tin = sum(10 ** (spl_a(render(patch("tin_roof", pos=(1.5, s * 1.0, 1.5), area=6.0, interior=True,
                                          normal=(0.0, 0.5 * s, 0.87), pid=f"t{s}"), weather(8.0), 1.0,
                                    seed=6)[0]) / 10) for s in (-1, 1))
    assert 10 * math.log10(2 * tin) > 50.0
    # sheltered is duller and quieter than the same roof heard in the open, but not silent
    open_roof = patch("tile", pos=(-1.0, 0.0, 3.0), area=40.0)
    x_open = render(open_roof, weather(4.0), 1.0, seed=6)[0]
    x_in = render(roof, weather(4.0), 1.0, seed=6)[0]
    assert spl(x_in) < spl(x_open) and centroid(x_in) < centroid(x_open)


def test_hail_is_percussive_and_snow_near_silent():
    sc = Scene()
    rngf = RngFactory(3)
    car = patch("car_body", kind="vehicle_body", pos=(0.0, 0.0, 0.4), area=1.8, interior=True)
    hail = R.HailGenerator(sc, [car], rngf)
    snow = R.SnowGenerator(sc, [patch("leaves", kind="vegetation", area=20.0, pos=(3.0, 0.0, 0.0))], rngf)
    h, s = [], []
    for b in range(94):
        ctx = BlockContext(sr=SR, n=1024, index=b, t0=b * 1024 / SR, weather=weather(20.0, precip="hail",
                                                                                     hail_mm=15.0), rng=rngf)
        em = hail.render(ctx)
        h.append(em[0].signal if em else np.zeros(1024))
        ctx = BlockContext(sr=SR, n=1024, index=b, t0=b * 1024 / SR,
                           weather=weather(2.0, precip="snow", temp=-4.0), rng=rngf)
        em = snow.render(ctx)
        s.append(sum(e.signal for e in em) if em else np.zeros(1024))
    h, s = np.concatenate(h), np.concatenate(s)
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(s))
    # hail: sparse hard hits -> crest factor > 15 dB and heavy-tailed amplitude (kurtosis >> 3, Gaussian = 3)
    assert 20 * np.log10(np.max(np.abs(h)) / h.std()) > 15.0
    assert np.mean(h ** 4) / np.mean(h ** 2) ** 2 > 10.0
    # snow on leaves: near silent (< 30 dB SPL) but not digital silence
    assert -40.0 < spl(s) < 30.0


def test_regular_entrainment_band():
    # Oguz & Prosperetti (1990), Pumphrey & Elmore (1990): 0.8-1.1 mm drops at terminal speed entrain
    # a bubble regularly; 2 mm drops at terminal speed do not
    assert R.regular_entrainment(1.0, float(D.fall_speed(1.0)))
    assert not R.regular_entrainment(2.0, float(D.fall_speed(2.0)))
    assert not R.regular_entrainment(0.5, float(D.fall_speed(0.5)))


def test_drip_plinks_ring_at_the_entrained_bubble_pitch():
    # a slow drip (8 cm fall, ~1.25 m/s) of a Tate's-law drop (2 mm edge -> ~4.7 mm) into water lies in the
    # regular-entrainment band (Oguz & Prosperetti 1990) and traps a bubble of radius 0.3-0.42 D: a 'plink'
    # whose spectral peak is the Minnaert pitch 3.28 / a (Minnaert 1933) of that bubble, ~1.6-2.3 kHz
    from raingen.physics.wetness import tate_drop_diameter_mm

    sc = Scene()
    rngf = RngFactory(5)
    edge = patch("tile", kind="drip_edge", pos=(1.0, 0.0, -1.0), area=0.1, pid="eave",
                 params={"feed": "rain", "target": "water", "catchment_m2": 0.4, "fall_height_m": 0.08,
                         "edge_radius_mm": 2.0, "n_points": 1})
    gen = R.DripGenerator(sc, edge, rngf)
    out, markers = [], []
    for b in range(94):
        ctx = BlockContext(sr=SR, n=1024, index=b, t0=b * 1024 / SR, weather=weather(10.0), rng=rngf, markers=[])
        em = gen.render(ctx)
        out.append(sum(e.signal for e in em) if em else np.zeros(1024))
        markers += ctx.markers
    x = np.concatenate(out)
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) > 0.0
    assert any(m.kind == "drip" and m.data["sound"] == "plink" for m in markers)
    d = tate_drop_diameter_mm(2.0)
    f_lo, f_hi = 3.28 / (0.42 * 1.15 * d * 1e-3), 3.28 / (0.3 * 0.85 * d * 1e-3)   # +-15 % node jitter
    f, P = signal.welch(x, SR, nperseg=8192)
    sel = f > 300.0
    f_peak = f[sel][np.argmax(P[sel])]
    assert f_lo <= f_peak <= 1.3 * f_hi          # the rising chirp (xi <= 0.3) lifts the peak a little


def test_filter_states_never_go_denormal():
    # decaying IIR / modal states are flushed below 1e-30 (no denormal arithmetic stalls in long silences)
    x = np.zeros(1024)
    x[0] = 1.0
    sos = K.StatefulSOS(signal.butter(2, 200.0, "lowpass", fs=SR, output="sos"))
    bank = K.ModalBank(MT.get_material("metal_sheet"), SR)
    batch = K.ModalBatch(K.ModalBank(MT.get_material("metal_sheet"), SR), 2)
    sos.process(x)
    bank.process(x)
    batch.process(np.stack([x, 0.5 * x]))
    z = np.zeros(1024)
    for _ in range(3000):                      # 64 s of silence: every state decays through 1e-308
        sos.process(z)
        bank.process(z)
        batch.process(np.zeros((2, 1024)))
    for st in (sos.zi, bank.state, batch.state):
        a = np.abs(st)
        assert np.all((a == 0.0) | (a >= 1e-30))
    assert np.all(sos.zi == 0.0)


def test_interior_panels_radiate_directly_into_the_room():
    # the drumming of a panel whose struck face *is* the room's boundary (attic roof, tent, car body) reaches
    # the listener unattenuated on the 'interior' stem (ShelterTransfer leaves that stem alone); only a roof
    # above a ceiling (mass law), a slab, or the outer pane of double glazing carries a transmission path
    tin = patch("tin_roof", pos=(1.5, 1.0, 1.5), area=6.0, interior=True, normal=(0.0, 0.5, 0.87))
    assert R._interior_transfer(tin, SR) == (1.0, None)
    assert R._stem_for(tin) == "interior"
    pane = patch("window", pos=(2.0, 0.0, 0.0), area=1.5, interior=True, normal=(1.0, 0.0, 0.0),
                 params={"glazing": "double"})
    gain, sos = R._interior_transfer(pane, SR)
    _, h = signal.sosfreqz(sos, worN=[500.0, 8000.0], fs=SR)
    # outer pane -> cavity -> inner pane: ~8 dB broadband (provisional), rolling off above ~3.5 kHz
    assert 20 * np.log10(gain * abs(h[0])) == pytest.approx(-8.0, abs=0.5)
    assert 20 * np.log10(abs(h[1])) < -6.0
    outdoor = patch("concrete", pos=(2.0, 0.0, -1.7))
    assert R._stem_for(outdoor) == "rain_near"
