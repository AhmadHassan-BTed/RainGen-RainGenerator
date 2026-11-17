"""Physics layer: DSD, fall speed, ISO 9613-1, propagation, wind, gusts, wetness, storms, World, layout.

Every numeric assertion names its reference. Where a reference value is uncertain the
comment says so. Audio is not rendered here (except the absorption filter); generators
are tested elsewhere.
"""

import math

import numpy as np
import pytest
from scipy import integrate, signal, stats

from raingen.core.constants import SAMPLE_RATE, speed_of_sound
from raingen.core.rng import RngFactory
from raingen.core.types import DSDParams
from raingen.physics import atmosphere as atm
from raingen.physics import dsd as D
from raingen.physics import layout as lay
from raingen.physics import storm as stm
from raingen.physics import wetness as wet
from raingen.physics import wind as wnd
from raingen.physics import world as wld
from raingen.scene import Scene, SceneObject


# ----------------------------------------------------------------------------- helpers
class _Precip:
    """Minimal Precipitation stand-in for dsd_from_scene."""

    def __init__(self, **kw):
        self.type = "rain"
        self.dsd_model = "gamma"
        self.drops = "normal"
        self.d_min_mm = 0.3
        self.d_max_mm = 6.0
        self.mu = None
        self.dm_mm = None
        self.nw = None
        self.__dict__.update(kw)


def _flux_pdf_ref(dsd):
    """Independent reference for N(D) v(D) (formulas re-typed from the papers, not the module)."""
    f_mu = 6.0 / 4.0 ** 4 * (4.0 + dsd.mu) ** (dsd.mu + 4.0) / math.gamma(dsd.mu + 4.0)

    def n_of(d):
        return dsd.nw * f_mu * (d / dsd.dm_mm) ** dsd.mu * math.exp(-(4.0 + dsd.mu) * d / dsd.dm_mm)

    def v_of(d):
        if d >= 0.5:
            return 9.65 - 10.3 * math.exp(-0.6 * d)          # Atlas et al. 1973
        # Stokes v = g D^2 (rho_w - rho_a) / (18 mu) below 0.1 mm, linear bridge to Atlas at 0.5 mm
        v01 = 9.81 * (0.1e-3) ** 2 * (998.0 - 1.204) / (18 * 1.81e-5)
        v05 = 9.65 - 10.3 * math.exp(-0.3)
        if d >= 0.1:
            return v01 + (v05 - v01) * (d - 0.1) / 0.4
        return 9.81 * (d * 1e-3) ** 2 * (998.0 - 1.204) / (18 * 1.81e-5)
    return n_of, v_of


def _quad(fn, lo, hi):
    pts = [p for p in (0.1, 0.5, 0.8, 1.1, 1.2, 2.0, 2.2, 3.5) if lo < p < hi]
    val, _ = integrate.quad(fn, lo, hi, points=pts or None, limit=200, epsabs=0.0, epsrel=1e-9)
    return val


def _rain_scene(rate=5.0, seed=11, **wind):
    sc = Scene()
    sc.seed = seed
    sc.weather.precipitation.rate_mm_h = rate
    sc.weather.precipitation.type = "rain"
    for k, v in wind.items():
        setattr(sc.weather.wind, k, v)
    return sc


def _run_world(sc, n_blocks, block=1024, seed=None):
    rngf = RngFactory(sc.seed if seed is None else seed)
    w = wld.World(sc, rngf, block=block)
    out = []
    for b in range(n_blocks):
        out.append(w.step(b, b * block / SAMPLE_RATE, block))
    return w, out


# ============================================================================= fall speed
# Gunn & Kinzer (1949) terminal velocities, 1013 hPa / 20 C (J. Meteor. 6:243, Table 2)
GUNN_KINZER = {0.5: 2.06, 1.0: 4.03, 1.5: 5.40, 2.0: 6.49, 3.0: 8.06, 4.0: 8.83, 5.0: 9.09}


@pytest.mark.parametrize("d_mm, v_ref", sorted(GUNN_KINZER.items()))
def test_atlas_fall_speed_matches_gunn_kinzer(d_mm, v_ref):
    # Atlas, Srivastava & Sekhon (1973) fit to Gunn & Kinzer (1949): within 5 % for 0.5-5 mm
    assert D.fall_speed(d_mm) == pytest.approx(v_ref, rel=0.05)


def test_fall_speed_regimes_continuous_and_vectorized():
    d = np.linspace(0.0, 8.0, 8001)
    v = D.fall_speed(d)
    assert v[0] == 0.0 and np.all(np.diff(v) >= -1e-12)          # monotone, never negative
    scal = np.array([D.fall_speed(float(x)) for x in d[::97]])
    assert np.allclose(v[::97], scal, atol=1e-5)                   # lookup table vs closed form
    # continuity at the regime edges (Stokes 0.1 mm, Atlas 0.5 mm): no step > 1 cm/s
    for edge in (0.1, 0.5):
        assert abs(D.fall_speed(edge + 1e-9) - D.fall_speed(edge - 1e-9)) < 0.01
    # Stokes law v ~ D^2 below 0.1 mm (Stokes drag; Re << 1)
    assert D.fall_speed(0.05) / D.fall_speed(0.025) == pytest.approx(4.0, rel=1e-3)
    # Gunn & Kinzer 0.1 mm: 0.27 m/s; Stokes over-predicts slightly at Re ~ 2 (allow 15 %)
    assert D.fall_speed(0.1) == pytest.approx(0.27, rel=0.15)
    # Atlas asymptote 9.65 m/s; drops break up above ~6 mm (Pruppacher & Klett)
    assert 9.3 < D.fall_speed(6.0) < 9.65
    assert isinstance(D.fall_speed(2.0), float) and isinstance(D.fall_speed(np.float64(2.0)), float)


def test_fall_speed_air_density_correction_direction():
    # Foote & du Toit (1969): v = v0 (rho0 / rho)^0.4 -> thinner air, faster drops
    rho_sea = D.air_density(20.0, 1013.25)
    assert rho_sea == pytest.approx(1.204, rel=2e-3)                 # ideal gas, dry air (ISA-like)
    rho_3km = D.air_density(-4.5, 701.1)                             # ISA at 3 km (-4.5 C, 701 hPa): 0.909 kg/m^3
    assert rho_3km == pytest.approx(0.909, rel=0.01)
    v0, v3 = D.fall_speed(2.0), D.fall_speed(2.0, rho_air=rho_3km)
    assert v3 > v0
    assert v3 / v0 == pytest.approx((1.204 / 0.909) ** 0.4, rel=0.01)    # ~ +12 %
    assert D.fall_speed(2.0, rho_air=1.4) < v0                       # denser (cold, high pressure) -> slower
    vec = D.fall_speed(np.array([1.0, 2.0]), rho_air=rho_3km)
    assert vec[1] == pytest.approx(v3, rel=1e-6)


def test_fall_speed_from_height_free_fall_and_terminal_limits():
    vt = D.fall_speed(4.0)
    h = np.array([0.001, 0.01, 0.1, 1.0, 5.0, 20.0, 100.0])
    u = D.fall_speed_from_height(4.0, h)
    assert np.all(np.diff(u) > 0.0)                                  # higher fall -> faster impact
    # short falls: drag negligible, U ~ sqrt(2 g h) (free fall)
    assert u[0] == pytest.approx(math.sqrt(2 * 9.81 * 0.001), rel=0.01)
    assert u[1] == pytest.approx(math.sqrt(2 * 9.81 * 0.01), rel=0.02)
    # long falls approach terminal speed (U(h) = V_T sqrt(1 - exp(-2gh/V_T^2)))
    assert u[-1] == pytest.approx(vt, rel=1e-6)
    assert u[3] < vt


# ============================================================================= DSD
RATES = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 150.0]


@pytest.mark.parametrize("model", ["gamma", "marshall_palmer"])
@pytest.mark.parametrize("drops", ["fine", "normal", "fat"])
def test_dsd_closure_rain_rate(model, drops):
    # closure: rain_rate(dsd_from_scene(R)) == R; brief tolerance 5 %, achieved ~1e-9
    for R in RATES:
        dsd = D.dsd_from_scene(_Precip(dsd_model=model, drops=drops), R)
        assert D.rain_rate(dsd) == pytest.approx(R, rel=1e-6)


@pytest.mark.parametrize("R", [0.5, 5.0, 50.0])
def test_dsd_integrals_match_independent_quadrature(R):
    # the cached-table integrals vs adaptive quadrature of the same physics (R = 6 pi 1e-4 Int D^3 v N dD)
    for drops in ("fine", "fat"):
        dsd = D.dsd_from_scene(_Precip(drops=drops), R)
        n_of, v_of = _flux_pdf_ref(dsd)
        r_ref = 6 * math.pi * 1e-4 * _quad(lambda d, n=n_of, v=v_of: d ** 3 * v(d) * n(d), dsd.d_min_mm, dsd.d_max_mm)
        assert R == pytest.approx(r_ref, rel=2e-3)
        f_ref = _quad(lambda d, n=n_of, v=v_of: v(d) * n(d), dsd.d_min_mm, dsd.d_max_mm)
        assert D.total_flux(dsd) == pytest.approx(f_ref, rel=2e-3)
        lwc_ref = math.pi / 6 * 1e-3 * _quad(lambda d, n=n_of: d ** 3 * n(d), dsd.d_min_mm, dsd.d_max_mm)
        assert D.liquid_water_content(dsd) == pytest.approx(lwc_ref, rel=2e-3)


@pytest.mark.parametrize("R", [0.5, 1.0, 5.0, 25.0, 100.0])
def test_marshall_palmer_parameters(R):
    lam = 4.1 * R ** -0.21                                           # Marshall & Palmer (1948)
    mp = D.marshall_palmer(R, d_min_mm=1e-3, d_max_mm=12.0)
    assert mp.nw == 8000.0 and mp.mu == 0.0                          # N0 = 8000 m^-3 mm^-1
    assert mp.dm_mm == pytest.approx(4.0 / lam, rel=1e-12)           # exponential: Dm = 4 / Lambda
    d = np.array([0.5, 1.0, 2.0, 4.0])
    assert np.allclose(D.number_density(mp, d), 8000.0 * np.exp(-lam * d), rtol=1e-9)
    # LWC of the untruncated MP DSD: pi rho N0 / Lambda^4 = 0.0889 R^0.84 g/m^3 (analytic)
    assert D.liquid_water_content(mp) == pytest.approx(0.0889 * R ** 0.84, rel=0.01)
    # MP integrates back to roughly R with Atlas speeds (N0 was fitted with older speed data;
    # the rate implied by N0 = 8000 is 5-20 % high - uncertain reference, loose tolerance)
    assert 0.95 * R < D.rain_rate(mp) < 1.25 * R
    # dsd_from_scene keeps Lambda exactly and solves Nw for closure: Nw stays near N0
    sc = D.dsd_from_scene(_Precip(dsd_model="marshall_palmer"), R)
    assert sc.mu == 0.0 and sc.dm_mm == pytest.approx(4.0 / lam, rel=1e-12)
    assert 0.8 * 8000.0 < sc.nw < 1.1 * 8000.0


# Impact rate table of MASTER_PLAN 7.1 (Marshall-Palmer, D >= 0.5 mm), m^-2 s^-1
MP_IMPACTS = {0.5: 450, 1: 750, 2.5: 1380, 5: 2100, 10: 3100, 25: 5000, 50: 7100, 100: 9700}


@pytest.mark.parametrize("R, ref", sorted(MP_IMPACTS.items()))
def test_marshall_palmer_impact_rate_table(R, ref):
    mp = D.marshall_palmer(R, d_min_mm=0.5, d_max_mm=8.0)
    # plan table (computed, independently reproduced; ~750 R^0.56): within 3 %
    assert D.total_flux(mp) == pytest.approx(ref, rel=0.03)


def test_marshall_palmer_class_shares_on_water():
    # MASTER_PLAN 7.2 table (Medwin/Nystuen classes, MP, impacts with D >= 0.5 mm):
    # small ~21-27 % (21 % at 100 mm/h); mid 10 % at 0.5 -> 42 % at 100; large ~0.1 % -> 11 %;
    # very large ~0 -> 2 %. Tolerance +-1.5 percentage points.
    def shares(R):
        cf = D.class_flux(D.marshall_palmer(R, 0.5, 8.0))
        tot = sum(cf.values())
        return {k: 100.0 * v / tot for k, v in cf.items()}
    lo, hi = shares(0.5), shares(100.0)
    assert lo["mid"] == pytest.approx(10.0, abs=1.5) and hi["mid"] == pytest.approx(42.0, abs=1.5)
    assert lo["large"] == pytest.approx(0.1, abs=0.3) and hi["large"] == pytest.approx(11.0, abs=1.5)
    assert lo["very_large"] < 0.1 and hi["very_large"] == pytest.approx(2.0, abs=0.7)
    assert hi["small"] == pytest.approx(21.0, abs=1.5)
    for R in (0.5, 2.5, 10.0):
        assert 20.0 <= shares(R)["small"] <= 28.5


def test_class_bounds_medwin_and_nystuen():
    # Medwin, Nystuen et al. (1992) 0.8/1.1/2.2/3.5 mm; Nystuen (2001) 0.8/1.2/2.0/3.5 mm
    med = D.class_bounds("medwin", 6.0)
    nys = D.class_bounds("nystuen", 6.0)
    assert [med[c][1] for c in D.CLASS_NAMES[:4]] == [0.8, 1.1, 2.2, 3.5]
    assert [nys[c][1] for c in D.CLASS_NAMES[:4]] == [0.8, 1.2, 2.0, 3.5]
    assert med["very_large"] == (3.5, 6.0) and med["minuscule"][0] == 0.0
    with pytest.raises(ValueError):
        D.class_bounds("bogus")


@pytest.mark.parametrize("scheme", ["medwin", "nystuen"])
def test_class_flux_partitions_total_flux(scheme):
    for drops in ("fine", "normal", "fat"):
        for R in (0.2, 4.0, 60.0):
            dsd = D.dsd_from_scene(_Precip(drops=drops), R)
            cf = D.class_flux(dsd, scheme)
            assert list(cf) == list(D.CLASS_NAMES)
            # classes are a partition of [d_min, d_max]: they sum to the total flux
            assert sum(cf.values()) == pytest.approx(D.total_flux(dsd), rel=1e-10)
            n_of, v_of = _flux_pdf_ref(dsd)
            for name, (lo, hi) in D.class_bounds(scheme, dsd.d_max_mm).items():
                lo, hi = max(lo, dsd.d_min_mm), min(hi, dsd.d_max_mm)
                ref = _quad(lambda d, n=n_of, v=v_of: v(d) * n(d), lo, hi) if hi > lo else 0.0
                assert cf[name] == pytest.approx(ref, rel=3e-3, abs=1e-6 * D.total_flux(dsd))


def test_class_flux_nystuen_moves_mass_between_small_and_mid():
    dsd = D.dsd_from_scene(_Precip(), 10.0)
    med, nys = D.class_flux(dsd, "medwin"), D.class_flux(dsd, "nystuen")
    assert nys["small"] > med["small"]                    # 0.8-1.2 vs 0.8-1.1
    assert nys["large"] > med["large"]                    # 2.0-3.5 vs 2.2-3.5
    assert nys["minuscule"] == pytest.approx(med["minuscule"]) and nys["very_large"] == pytest.approx(med["very_large"])


@pytest.mark.parametrize("lo, hi", [(0.3, 6.0), (1.1, 2.2), (0.3, 0.8), (3.5, 6.0)])
def test_sample_diameters_follow_flux_distribution(lo, hi):
    dsd = D.dsd_from_scene(_Precip(drops="fat"), 20.0)
    rng = np.random.default_rng(20251116)
    x = D.sample_diameters(rng, dsd, lo, hi, 20000)
    assert x.shape == (20000,) and x.min() >= lo and x.max() <= hi
    n_of, v_of = _flux_pdf_ref(dsd)
    grid = np.linspace(lo, hi, 2001)
    pdf = np.array([n_of(d) * v_of(d) for d in grid])
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))))
    cdf /= cdf[-1]
    # impacts are flux-weighted, N(D) v(D) (Kolmogorov-Smirnov, fixed seed)
    res = stats.kstest(x, lambda q: np.interp(q, grid, cdf))
    assert res.pvalue > 0.01
    # and *not* number-weighted (N(D) alone) where v(D) varies enough to tell (small drops)
    if lo < 1.0 and hi - lo > 1.0:
        pdf_n = np.array([n_of(d) for d in grid])
        cdf_n = np.concatenate(([0.0], np.cumsum(0.5 * (pdf_n[1:] + pdf_n[:-1]) * np.diff(grid))))
        cdf_n /= cdf_n[-1]
        assert stats.kstest(x, lambda q: np.interp(q, grid, cdf_n)).pvalue < 1e-6


def test_sample_diameters_deterministic_and_edge_cases():
    dsd = D.dsd_from_scene(_Precip(), 5.0)
    a = D.sample_diameters(np.random.default_rng(3), dsd, 1.1, 2.2, 500)
    b = D.sample_diameters(np.random.default_rng(3), dsd, 1.1, 2.2, 500)
    c = D.sample_diameters(np.random.default_rng(4), dsd, 1.1, 2.2, 500)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert D.sample_diameters(np.random.default_rng(0), dsd, 1.0, 2.0, 0).size == 0
    # band outside the cutoffs -> constant midpoint (documented fallback)
    out = D.sample_diameters(np.random.default_rng(0), dsd, 7.0, 8.0, 4)
    assert np.all(out == 7.5)
    # exactly one rng.random(n) draw is consumed (callers share the generator)
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    D.sample_diameters(r1, dsd, 0.3, 6.0, 37)
    r2.random(37)
    assert r1.random() == r2.random()


def test_bringi_regime_clusters_at_convective_rates():
    # Bringi et al. (2003) convective clusters (cluster centres, mu = 3), as used in MASTER_PLAN 7.1:
    # maritime Dm 1.5-1.75 mm, log10 Nw 4-4.5; continental Dm 2-2.75 mm, log10 Nw 3-3.5;
    # ~3x more impacts per mm/h maritime; share of impacts > 2.2 mm 3.4 % vs 18.6 %.
    R = 30.0
    fine = D.dsd_from_scene(_Precip(drops="fine"), R)
    fat = D.dsd_from_scene(_Precip(drops="fat"), R)
    assert 1.5 <= fine.dm_mm <= 1.75 and 4.0 <= math.log10(fine.nw) <= 4.5
    assert 2.0 <= fat.dm_mm <= 2.75 and 3.0 <= math.log10(fat.nw) <= 3.6
    ratio = D.total_flux(fine) / D.total_flux(fat)
    assert ratio == pytest.approx(3.0, rel=0.15)
    share = {k: D.total_flux(v, 2.2, 99.0) / D.total_flux(v) for k, v in (("fine", fine), ("fat", fat))}
    assert share["fine"] == pytest.approx(0.034, abs=0.012)
    assert share["fat"] == pytest.approx(0.186, abs=0.03)


def test_regime_ordering_and_saturation():
    for R in np.geomspace(0.1, 150.0, 25):
        dms = [D.dsd_from_scene(_Precip(drops=k), R).dm_mm for k in ("fine", "normal", "fat")]
        assert dms[0] < dms[1] < dms[2]                       # drop-size control is monotone at every rate
    for k in ("fine", "normal", "fat"):
        dm = [D.dsd_from_scene(_Precip(drops=k), R).dm_mm for R in RATES]
        assert np.all(np.diff(dm) > 0.0)                      # Dm grows with rate ...
        assert dm[-1] < 2.9                                   # ... but saturates (Bringi 2003: <= ~2.75 mm)
    # heavy stratiform rain: D0 stabilises around 1.7-2.0 mm (MASTER_PLAN 7.1; +-0.1 mm)
    for R in (50.0, 100.0, 150.0):
        d0 = D.median_volume_diameter(D.dsd_from_scene(_Precip(), R))
        assert 1.6 <= d0 <= 2.1
    # extra rate comes from more drops: concentration grows faster than Dm
    lo, hi = D.dsd_from_scene(_Precip(), 10.0), D.dsd_from_scene(_Precip(), 100.0)
    assert D.concentration(hi) / D.concentration(lo) > (hi.dm_mm / lo.dm_mm) ** 3


def test_dsd_overrides_and_size_sorting():
    # nw pinned, Dm free -> Dm solved for closure
    dsd = D.dsd_from_scene(_Precip(nw=3000.0), 12.0)
    assert dsd.nw == 3000.0 and D.rain_rate(dsd) == pytest.approx(12.0, rel=1e-4)
    # both pinned -> the rate follows from them
    dsd2 = D.dsd_from_scene(_Precip(nw=3000.0, dm_mm=1.2), 12.0)
    assert dsd2.dm_mm == 1.2 and D.rain_rate(dsd2) != pytest.approx(12.0, rel=0.05)
    # mu override is honoured and clamped
    assert D.dsd_from_scene(_Precip(mu=7.5), 5.0).mu == 7.5
    assert D.dsd_from_scene(_Precip(mu=99.0), 5.0).mu == 20.0
    # size sorting: bigger Dm at the same rate means fewer impacts (MASTER_PLAN 7.4)
    base = D.dsd_from_scene(_Precip(), 8.0)
    big = D.dsd_with_dm(base, base.dm_mm * 1.4, 8.0)
    assert D.rain_rate(big) == pytest.approx(8.0, rel=1e-6)
    assert D.total_flux(big) < 0.75 * D.total_flux(base)
    # no rain -> empty DSD and zero flux
    z = D.dsd_from_scene(_Precip(), 0.0)
    assert z.nw == 0.0 and D.rain_rate(z) == 0.0 and sum(D.class_flux(z).values()) == 0.0


def test_drizzle_dsd():
    # WMO drizzle: drops < 0.5 mm; light rates
    dz = D.dsd_from_scene(_Precip(type="drizzle"), 0.2)
    assert dz.dm_mm <= 0.5 and dz.d_max_mm <= 1.2 and dz.d_min_mm <= 0.1
    assert D.rain_rate(dz) == pytest.approx(0.2, rel=1e-6)
    cf = D.class_flux(dz)
    assert cf["mid"] + cf["large"] + cf["very_large"] < 0.01 * sum(cf.values())


def test_dsd_from_spectrum_recovers_gamma_parameters(tmp_path):
    true = DSDParams(nw=6000.0, dm_mm=1.6, mu=4.0, d_min_mm=0.05, d_max_mm=9.0)
    d = np.linspace(0.05, 9.0, 600)
    nd = D.number_density(true, d)
    fit = D.dsd_from_spectrum(d, nd)
    # method of moments (Bringi & Chandrasekar 2001): exact for a fully sampled gamma
    assert fit.dm_mm == pytest.approx(1.6, rel=0.01)
    assert fit.nw == pytest.approx(6000.0, rel=0.02)
    assert fit.mu == pytest.approx(4.0, abs=0.15)
    p = tmp_path / "disdro.csv"
    p.write_text("diameter_mm,n_d\n" + "\n".join(f"{a},{b}" for a, b in zip(d, nd, strict=True)))
    assert D.dsd_from_disdrometer_csv(str(p)).dm_mm == pytest.approx(fit.dm_mm, rel=1e-9)


# ============================================================================= ISO 9613-1 and propagation
# ISO 9613-2:1996 Table 2 (computed with the ISO 9613-1 formulas at exact octave mid-band
# frequencies 1000 * 10^(3k/10)), dB/km: 63 125 250 500 1k 2k 4k 8k
ISO_TABLE = {
    (10.0, 70.0): (0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0),
    (20.0, 70.0): (0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6),
    (30.0, 70.0): (0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3),
    (15.0, 20.0): (0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0),
    (15.0, 50.0): (0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0),
    (15.0, 80.0): (0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8),
}


@pytest.mark.parametrize("cond", sorted(ISO_TABLE))
def test_iso9613_alpha_matches_standard_table(cond):
    T, H = cond
    f = 1000.0 * 10.0 ** (0.3 * np.arange(-4, 4))
    a = atm.iso9613_alpha_db_per_m(f, T, H) * 1000.0
    ref = np.array(ISO_TABLE[cond])
    # table is rounded to 0.1 dB/km (3 significant figures above 100): half a unit + 0.5 %
    tol = np.where(ref >= 100.0, 0.5, 0.05) + 0.005 * ref
    assert np.all(np.abs(a - ref) <= tol), (a, ref)


def test_iso9613_nominal_frequencies_and_trends():
    # ISO 9613-1 Table 1 / MASTER_PLAN 7.8 at 20 C, 70 % RH (nominal frequencies), dB/km
    ref = {1000.0: 4.98, 2000.0: 9.04, 4000.0: 23.1, 8000.0: 77.6}
    for f, v in ref.items():
        assert float(atm.iso9613_alpha_db_per_m(f, 20.0, 70.0)) * 1000.0 == pytest.approx(v, rel=0.01)
    # above ~20 % RH absorption at 8 kHz falls as humidity rises (oxygen relaxation moves up);
    # in very dry air it peaks (ISO 9613-1 Table 1: max near 10-20 % RH at 8 kHz)
    a = [float(atm.iso9613_alpha_db_per_m(8000.0, 15.0, h)) for h in (5, 20, 50, 80)]
    assert a[1] > a[2] > a[3] and a[1] > a[0]
    # pressure barely matters (MASTER_PLAN 7.8: 980 -> 1030 hPa is inaudible): < 5 % at 1-8 kHz
    f = np.array([1000.0, 4000.0, 8000.0])
    r = atm.iso9613_alpha_db_per_m(f, 15.0, 70.0, 980.0) / atm.iso9613_alpha_db_per_m(f, 15.0, 70.0, 1030.0)
    assert np.all(np.abs(r - 1.0) < 0.05)
    assert np.allclose(atm.air_absorption_db(f, 250.0, 15.0, 70.0), atm.iso9613_alpha_db_per_m(f, 15.0, 70.0) * 250.0)


def _check_absorption_design(sos, d, T, H, sr=SAMPLE_RATE):
    f = np.geomspace(20.0, 0.45 * sr, 500)
    att = -atm._sos_mag_db(sos, f, sr)
    loss = atm.iso9613_alpha_db_per_m(f, T, H) * d
    fit = (loss <= 40.0) & (f <= 10000.0)
    err = np.abs(att - loss)[fit]
    deep = loss > 40.0
    return (float(err.max()) if err.size else 0.0), (float(att[deep].min()) if np.any(deep) else math.inf)


@pytest.mark.parametrize("T, H", [(20.0, 70.0), (10.0, 90.0), (30.0, 30.0), (-5.0, 60.0), (0.0, 100.0)])
def test_air_absorption_design_matches_iso_within_1db(T, H):
    # filter magnitude vs alpha * distance: within 1 dB to 10 kHz wherever the ISO loss is
    # <= 40 dB; beyond that at least ~38 dB of attenuation (design contract)
    # (377 m at 30 C / 30 % RH was a 3.3 dB miss before the deep target ramped at 24 dB/oct)
    for d in (2.0, 7.0, 25.0, 80.0, 250.0, 377.0, 700.0, 2000.0, 6000.0):
        sos = atm.air_absorption_sos(d, T, H)
        err, deep_min = _check_absorption_design(sos, d, T, H)
        assert err <= 1.0, (d, err)
        assert deep_min >= 38.0, (d, deep_min)
        for row in sos:                                               # every section stable
            assert np.all(np.abs(np.roots(row[3:])) < 1.0)
    assert np.array_equal(atm.air_absorption_sos(0.5, 20.0, 70.0), atm._PASSTHROUGH)   # < 1 m: none


def test_air_absorption_filter_stream_matches_design_and_any_block_length(rng):
    sr = SAMPLE_RATE
    x = rng.standard_normal(3 * 4096 + 517)
    sos = atm.air_absorption_sos(60.0, 12.0, 80.0, sr)
    ref = signal.sosfilt(sos, x)
    for sizes in ([1024] * 13, [1, 7, 333, 2048, 5000, 4096, 1, 1319], [12805]):
        filt = atm.AirAbsorptionFilter(sr)
        out, i = [], 0
        for n in sizes:
            n = min(n, x.size - i)
            if n <= 0:
                break
            out.append(filt.process(x[i:i + n], 60.0, 12.0, 80.0))
            i += n
        y = np.concatenate(out)
        assert y.size == x.size and np.allclose(y, ref, atol=1e-12)
    # the measured transfer of the streamed filter matches alpha * d (1 dB, Welch estimate, 0.2-10 kHz)
    f, pxx = signal.welch(x, sr, nperseg=1024)
    _, pyy = signal.welch(ref, sr, nperseg=1024)
    sel = (f >= 200.0) & (f <= 10000.0)
    meas = 10.0 * np.log10(pxx[sel] / pyy[sel])
    loss = atm.iso9613_alpha_db_per_m(f[sel], 12.0, 80.0) * 60.0
    assert np.max(np.abs(meas - loss)) < 1.0


def test_air_absorption_filter_moving_source_is_click_free(rng):
    sr, n = SAMPLE_RATE, 512
    filt = atm.AirAbsorptionFilter(sr)
    x = 0.1 * rng.standard_normal(200 * n)
    y = np.empty_like(x)
    dists = np.geomspace(5.0, 400.0, 200)                     # a source flying away
    for k in range(200):
        y[k * n:(k + 1) * n] = filt.process(x[k * n:(k + 1) * n], dists[k], 15.0, 60.0)
    assert np.all(np.isfinite(y))
    dy = np.abs(np.diff(y))
    edges = np.arange(n, y.size, n) - 1
    inner = np.setdiff1d(np.arange(dy.size), edges)
    # redesigns crossfade: boundary steps are no larger than ordinary sample-to-sample steps
    assert np.percentile(dy[edges], 99) <= 1.5 * np.percentile(dy[inner], 99.9)
    # farther = duller: the HF share falls as the source recedes
    def hf_share(seg):
        sp = np.abs(np.fft.rfft(seg)) ** 2
        fr = np.fft.rfftfreq(seg.size, 1.0 / sr)
        return sp[fr > 6000].sum() / sp.sum()
    assert hf_share(y[5 * n:25 * n]) > 2.0 * hf_share(y[175 * n:195 * n])


def test_air_absorption_filter_state_flushes_to_zero_on_silence(rng):
    # IIR state decaying on silence would reach subnormals (< 2.2e-308), which stall x86
    # FPUs ~100x; states below 1e-30 are flushed, so after silence the state is exactly 0
    sr, n = SAMPLE_RATE, 1024
    filt = atm.AirAbsorptionFilter(sr)
    for _ in range(20):
        filt.process(rng.standard_normal(n), 300.0, 15.0, 60.0)
    assert np.any(filt._zi != 0.0)
    tiny = np.finfo(float).tiny
    for _ in range(int(3.0 * sr / n)):                         # 3 s of digital silence
        y = filt.process(np.zeros(n), 300.0, 15.0, 60.0)
        z = filt._zi
        assert not np.any((z != 0.0) & (np.abs(z) < tiny))     # never subnormal
    assert np.all(filt._zi == 0.0) and np.all(y == 0.0)
    # also while a redesign crossfades (moving source) and for multichannel input
    filt2 = atm.AirAbsorptionFilter(sr)
    filt2.process(rng.standard_normal((2, n)), 40.0, 15.0, 60.0)
    for k in range(int(3.0 * sr / n)):
        filt2.process(np.zeros((2, n)), 40.0 * 1.01 ** k, 15.0, 60.0)
    assert np.all(filt2._zi == 0.0)


@pytest.mark.parametrize("kind, db_per_doubling", [("point", 20.0 * math.log10(2.0)), ("line", 10.0 * math.log10(2.0)),
                                                   ("plane", 0.0)])
def test_spreading_gain_per_doubling(kind, db_per_doubling):
    # spherical 1/r (-6.02 dB/dd), cylindrical 1/sqrt(r) (-3.01 dB/dd), plane wave (0)
    for r in (1.0, 3.0, 40.0, 900.0):
        drop = 20.0 * math.log10(atm.spreading_gain(r, kind=kind) / atm.spreading_gain(2.0 * r, kind=kind))
        assert drop == pytest.approx(db_per_doubling, abs=1e-9)
    assert atm.spreading_gain(1.0, kind=kind) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        atm.spreading_gain(1.0, kind="cone")


def test_shadow_distance():
    # MASTER_PLAN 7.7: a 5 km source in a 9.8 K/km lapse is shadowed beyond ~25 km
    r = atm.shadow_distance_m(5000.0, 1.6, 15.0, 9.8)
    assert r == pytest.approx(25000.0, rel=0.05)
    # r = sqrt(2c/a)(sqrt hs + sqrt hr) with a = (c / 2T) dT/dz (geometric acoustics)
    c, T = speed_of_sound(15.0), 288.15
    a = c / (2 * T) * 0.0098
    assert r == pytest.approx(math.sqrt(2 * c / a) * (math.sqrt(5000.0) + math.sqrt(1.6)), rel=1e-12)
    assert atm.shadow_distance_m(5000.0, 1.6, 15.0, 0.0) == math.inf            # no lapse, no shadow
    assert atm.shadow_distance_m(100.0, 1.6, 15.0, 20.0) < atm.shadow_distance_m(100.0, 1.6, 15.0, 5.0)


def test_refraction_gain_sign_and_size():
    def g(r, wind, stab):
        return atm.refraction_gain_db(r, wind, stab, 2.0, 1.6, 15.0)
    # downwind (c_eff grows with height): sound channels -> gain, capped at +10 dB
    assert g(3000.0, 6.0, "neutral") > 3.0 and g(20000.0, 10.0, "neutral") <= 10.0
    # upwind: shadow zone -> loss, floored at -20 dB (turbulent scattering)
    assert g(3000.0, -6.0, "neutral") < -10.0 and g(20000.0, -10.0, "neutral") >= -20.0
    # clear calm night (inversion) carries, sunny afternoon (lapse) shadows (MASTER_PLAN 7.8)
    assert g(2000.0, 0.0, "stable") > 0.0
    assert g(2000.0, 0.0, "unstable") < 0.0
    # short range: no refraction effect either way
    for stab in ("stable", "neutral", "unstable"):
        for wind in (-8.0, 0.0, 8.0):
            assert g(30.0, wind, stab) == 0.0
    # monotone in range on both sides
    rr = np.geomspace(100.0, 30000.0, 30)
    up = [g(r, -5.0, "neutral") for r in rr]
    down = [g(r, 5.0, "neutral") for r in rr]
    assert np.all(np.diff(up) <= 1e-12) and np.all(np.diff(down) >= -1e-12)
    # wind gradient sign: + downwind raises the effective gradient
    assert atm.sound_speed_gradient(5.0, "neutral", 2.0, 1.6, 15.0) > atm.sound_speed_gradient(-5.0, "neutral", 2.0, 1.6, 15.0)


@pytest.mark.parametrize("T, s_per_km", [(-10.0, 3.08), (20.0, 2.91), (35.0, 2.84)])
def test_flash_to_bang(T, s_per_km):
    # MASTER_PLAN 7.7: 3.08 / 2.91 / 2.84 s per km at -10 / 20 / 35 C
    assert atm.flash_to_bang_s(1.0, T) == pytest.approx(s_per_km, abs=0.006)
    assert atm.flash_to_bang_s(7.3, T) == pytest.approx(7300.0 / speed_of_sound(T), rel=1e-12)
    assert atm.flash_to_bang_s(-1.0, T) == 0.0


def test_ground_effect_is_finite_and_flat_for_hard_ground():
    sos = atm.ground_effect_sos(30.0, 1.0, 1.6, "grass")
    assert np.all(np.isfinite(sos))
    for row in sos:
        assert np.all(np.abs(np.roots(row[3:])) < 1.0)
    # soft ground: an interference dip somewhere in 100 Hz - 2 kHz (Embleton; MASTER_PLAN 7.8 200-600 Hz)
    f = np.geomspace(50.0, 4000.0, 300)
    mag = atm._sos_mag_db(sos, f, SAMPLE_RATE)
    assert mag.min() < -3.0 and 100.0 <= f[np.argmin(mag)] <= 2000.0


# ============================================================================= wind
WMO_BEAUFORT = [(0, 0.0, 0.2), (1, 0.3, 1.5), (2, 1.6, 3.3), (3, 3.4, 5.4), (4, 5.5, 7.9), (5, 8.0, 10.7),
                (6, 10.8, 13.8), (7, 13.9, 17.1), (8, 17.2, 20.7), (9, 20.8, 24.4), (10, 24.5, 28.4),
                (11, 28.5, 32.6), (12, 32.7, math.inf)]


def test_beaufort_table_matches_wmo():
    # WMO-No. 306 / WMO Manual on Marine Meteorological Services, 10 m mean wind, m/s
    assert [(r["force"], r["min_mps"], r["max_mps"]) for r in wnd.BEAUFORT] == WMO_BEAUFORT
    assert wnd.beaufort_description(3)[0] == "Gentle breeze"
    assert wnd.beaufort_description(12.4)[0] == "Hurricane"


def test_beaufort_continuous_fit_round_trips():
    # WMO conversion v = 0.836 B^(3/2) m/s
    for b in np.linspace(0.0, 12.0, 49):
        assert wnd.mps_to_beaufort(wnd.beaufort_to_mps(b)) == pytest.approx(b, abs=1e-9)
    for force, lo, hi in WMO_BEAUFORT[1:12]:
        v = wnd.beaufort_to_mps(force)
        assert lo - 0.05 <= v <= hi + 0.05                    # the integer force lands in its range
        assert round(wnd.mps_to_beaufort(0.5 * (lo + hi))) == force
        assert force - 0.55 <= wnd.mps_to_beaufort(lo) <= force   # range edges ~ B +- 0.5
        assert force <= wnd.mps_to_beaufort(hi) <= force + 0.55


def test_log_wind_profile():
    # neutral log law u(z) = u10 ln(z/z0) / ln(10/z0)
    for z0 in (0.0002, 0.03, 0.3, 1.0):
        assert wnd.wind_at_height(8.0, 10.0, z0) == pytest.approx(8.0, rel=1e-12)
        for z in (1.6, 2.0, 25.0):
            if z > 3 * z0:
                assert wnd.wind_at_height(8.0, z, z0) == pytest.approx(
                    8.0 * math.log(z / z0) / math.log(10.0 / z0), rel=1e-12)
    # open terrain (z0 = 0.03 m) at 2 m: 72 % of the 10 m wind
    assert wnd.wind_at_height(10.0, 2.0, 0.03) == pytest.approx(7.23, abs=0.01)
    # rougher terrain slows the near-ground wind more; inside the roughness sublayer the profile is held
    assert wnd.wind_at_height(10.0, 2.0, 0.5) < wnd.wind_at_height(10.0, 2.0, 0.03)
    assert wnd.wind_at_height(10.0, 0.5, 1.0) == wnd.wind_at_height(10.0, 3.0, 1.0)
    assert wnd.wind_at_height(-3.0, 5.0, 0.1) == 0.0


def test_turbulence_intensity_and_gust_factors():
    # MASTER_PLAN 7.6: open terrain at 1.6 m I_u ~ 0.25, G ~ 1.74 (1.5 at 10 m)
    assert wnd.turbulence_intensity_for(0.03, 1.6) == pytest.approx(0.25, abs=0.01)
    assert wnd.gust_factor_for(0.03, 1.6) == pytest.approx(1.74, abs=0.03)
    assert wnd.gust_factor_for(0.03, 10.0) == pytest.approx(1.5, abs=0.03)
    # 10 m gust factors: sea 1.28, water 1.39, suburb 1.8, forest/town 2.15 (+-3 %)
    for place, g in (("open_sea", 1.28), ("lake", 1.39), ("suburb", 1.8), ("forest", 2.15), ("town", 2.15)):
        assert wnd.gust_factor_for(wnd.terrain_z0(place), 10.0) == pytest.approx(g, rel=0.03)
    assert wnd.terrain_z0("Deciduous Forest") == wnd.terrain_z0("deciduous_forest")
    assert wnd.terrain_z0("my_city_rooftop") == wnd.terrain_z0("city")


def test_kaimal_gust_periods():
    # Kaimal peak n = f z / U ~ 0.045: ~22 s at 10 m and 10 m/s, ~6 s at 2 m and 7 m/s (MASTER_PLAN 7.6)
    for u, z, period in ((10.0, 10.0, 22.2), (7.0, 2.0, 6.35)):
        T = wnd.kaimal_integral_time_s(u, z)
        assert 2.0 * math.pi * T == pytest.approx(period, rel=0.02)


def _gust_series(T=3.0, ti=0.25, n=1 << 15, seed=5, layer="wind.gust", cr=SAMPLE_RATE / 1024):
    gp = wnd.GustProcess(RngFactory(seed), layer, integral_time_s=T, ti=ti, control_rate=cr)
    return gp, np.array([gp.step(b) for b in range(n)])


@pytest.mark.slow
def test_gust_process_statistics_kaimal():
    cr = SAMPLE_RATE / 1024
    _, g = _gust_series(T=2.0, ti=0.25, n=1 << 16)
    # log-normal multiplier: mean 1 (mean wind preserved) and std = TI (I_u = sigma_u / U);
    # ~5k correlation times -> sampling error of the mean ~0.5 %, of the std ~2-3 %
    assert g.mean() == pytest.approx(1.0, abs=0.02)
    assert g.std() == pytest.approx(0.25, rel=0.08)
    assert stats.skew(g) > 0.3                               # gusts overshoot more than lulls undershoot
    assert g.min() >= 0.05
    # spectrum: flat below the integral-scale corner, ~f^-5/3 above it (Kaimal inertial subrange)
    f, p = signal.welch(np.log(g), cr, nperseg=4096)
    fc = 1.0 / (2.0 * math.pi * 2.0)
    sel = (f > 4.0 * fc) & (f < 2.0)
    slope = np.polyfit(np.log(f[sel]), np.log(p[sel]), 1)[0]
    assert slope == pytest.approx(-5.0 / 3.0, abs=0.25)
    low = (f > 0.0) & (f < 0.3 * fc)
    assert np.polyfit(np.log(f[low]), np.log(p[low]), 1)[0] > -0.6


def test_gust_process_turbulence_intensity_scales_variance():
    _, a = _gust_series(T=1.0, ti=0.1, n=8000)
    _, b = _gust_series(T=1.0, ti=0.4, n=8000)
    assert a.std() == pytest.approx(0.1, rel=0.2) and b.std() == pytest.approx(0.4, rel=0.2)
    _, c = _gust_series(T=1.0, ti=0.0, n=200)
    assert np.all(c == 1.0)                                   # TI 0 -> no gusts at all


def test_gust_process_deterministic_and_seekable():
    _, a = _gust_series(n=600)
    _, b = _gust_series(n=600)
    assert np.array_equal(a, b)
    _, c = _gust_series(n=600, seed=6)
    _, d = _gust_series(n=600, layer="env.gust_field")
    assert not np.allclose(a, c) and not np.allclose(a, d)
    # a fresh process seeking straight to block 450 replays exactly (max_replay_s = 120 s)
    gp = wnd.GustProcess(RngFactory(5), "wind.gust", integral_time_s=3.0, ti=0.25, control_rate=SAMPLE_RATE / 1024)
    assert gp.step(450) == a[450]
    assert gp.step(451) == a[451]
    # seeking backwards replays from the start: identical values
    assert gp.step(17) == a[17]


def test_gust_process_at_delay_is_causal_and_consistent():
    cr = SAMPLE_RATE / 1024
    dt = 1.0 / cr
    _, ref = _gust_series(n=1700)
    gp = wnd.GustProcess(RngFactory(5), "wind.gust", integral_time_s=3.0, ti=0.25, control_rate=cr)
    assert gp.at_delay(0.0) == 1.0                           # before the first step: neutral
    b = 1000
    for k in range(b + 1):
        gp.step(k)
    assert gp.at_delay(0.0) == ref[b]
    for k in (1, 7, 300):                                    # the past, exactly
        assert gp.at_delay(k * dt) == pytest.approx(ref[b - k], abs=1e-12)
    lead_blocks = int(math.floor(gp.lead_s * cr))
    for k in (1, 50, lead_blocks):                           # the near future is already synthesised
        assert gp.at_delay(-k * dt) == pytest.approx(ref[b + k], abs=1e-12)
    # fractional delays interpolate between neighbours
    mid = gp.at_delay(2.5 * dt)
    assert min(ref[b - 2], ref[b - 3]) - 1e-12 <= mid <= max(ref[b - 2], ref[b - 3]) + 1e-12
    # Taylor advection: a point L m upwind gets the listener's gust L/U s earlier
    U = 6.0
    assert gp.at_position(30.0, U) == pytest.approx(gp.at_delay(-30.0 / U), abs=1e-12)
    assert gp.at_position(-30.0, U) == pytest.approx(gp.at_delay(30.0 / U), abs=1e-12)
    k = int(round(30.0 / U * cr))
    assert gp.at_position(k * dt * U, U) == pytest.approx(ref[b + k], abs=1e-12)
    # stepping on does not change what was already returned (no retroactive edits)
    gp.step(b + 1)
    assert gp.at_delay(dt) == ref[b]


def test_world_gust_contract_reproducible_by_a_second_process():
    # KNOWN SEAM: a GustProcess built and driven per the World docstring reproduces WeatherState.gust
    sc = _rain_scene(rate=3.0, seed=77, speed_mps=7.0)
    rngf = RngFactory(sc.seed)
    w = wld.World(sc, rngf)
    T0, ti0 = w.gust_integral_time_s, w.gust_ti
    twin = wnd.GustProcess(RngFactory(sc.seed), wld.World.GUST_LAYER, integral_time_s=T0, ti=ti0,
                           control_rate=SAMPLE_RATE / 1024)
    for b in range(300):
        st = w.step(b, b * 1024 / SAMPLE_RATE, 1024)
        twin.set_params(w.gust_integral_time_s, st.turbulence_intensity)
        assert twin.step(b) == st.gust


# ============================================================================= wetness
def test_film_store_fills_saturates_and_dries_with_aftermath_tau():
    st = wet.WetnessStores(["glass", "concrete"], aftermath_tau_min=8.0)
    exp = {"glass": 1.0, "concrete": 1.0}
    st.equilibrate(5.0, exp, minutes=60.0)
    cap_g = wet.WetnessStores.params_for("glass")[1]
    assert st.film_mm("glass") == pytest.approx(cap_g, rel=1e-6)          # saturated at capacity
    assert st.film("glass") > 0.99 and st.runoff_mm_h["glass"] > 0.0
    w0 = st.film_mm("glass")
    tau_f = wet.WetnessStores.params_for("glass")[2] * 8.0 * 60.0         # drain factor x aftermath tau
    for _ in range(int(round(tau_f / 2.0))):
        st.update(2.0, 0.0, exp)
    # linear reservoir, integrated exactly: one time constant leaves e^-1
    assert st.film_mm("glass") == pytest.approx(w0 * math.exp(-1.0), rel=1e-3)
    st.decay(60.0, exp)
    assert st.film("glass") < 0.01                                        # dry an hour later
    # a longer aftermath tau dries more slowly
    slow = wet.WetnessStores(["glass"], aftermath_tau_min=30.0)
    slow.equilibrate(5.0, {"glass": 1.0}, minutes=60.0)
    slow.decay(10.0, {"glass": 1.0})
    fast = wet.WetnessStores(["glass"], aftermath_tau_min=8.0)
    fast.equilibrate(5.0, {"glass": 1.0}, minutes=60.0)
    fast.decay(10.0, {"glass": 1.0})
    assert slow.film_mm("glass") > 3.0 * fast.film_mm("glass")


def test_film_and_puddles_grow_with_rain_and_exposure():
    def wet_after(rate, exposure=1.0, minutes=30.0):
        st = wet.WetnessStores(["soil", "asphalt", "puddle"], aftermath_tau_min=8.0)
        st.equilibrate(rate, {"soil": exposure, "asphalt": exposure, "puddle": exposure}, minutes=minutes)
        return st
    light, heavy = wet_after(0.5), wet_after(20.0)
    assert heavy.film("soil") > light.film("soil")
    # sealed ground fills its hollows in any rain (depression storage) ...
    assert heavy.puddle_mm("asphalt") > light.puddle_mm("asphalt") > 0.0
    # ... pervious soil ponds only when rain exceeds what it absorbs (Horton infiltration excess)
    assert light.puddle_mm("soil") == 0.0 and heavy.puddle_mm("soil") > 1.0
    assert heavy.puddle_mm("puddle") > light.puddle_mm("puddle") >= 3.0      # puddles exist before rain
    sheltered = wet_after(20.0, exposure=0.0)
    assert sheltered.film("soil") == 0.0 and sheltered.puddle_mm("asphalt") == 0.0
    # puddles outlast films: after 15 dry minutes the film fraction left is smaller
    for st in (heavy,):
        f0, p0 = st.film_mm("asphalt"), st.puddle_mm("asphalt")
        st.decay(15.0, {"asphalt": 1.0})
        assert st.puddle_mm("asphalt") / p0 > st.film_mm("asphalt") / f0
    assert heavy.puddle_mm("water") == 1000.0 and heavy.film("water") == 1.0


def test_wetness_independent_of_time_step():
    # exact reservoir integration: block-rate updates == coarse updates (any block length)
    def run(dt):
        st = wet.WetnessStores(["wood", "asphalt"], aftermath_tau_min=5.0, roof_area_m2=30.0)
        exp = {"wood": 0.8, "asphalt": 1.0}
        t = 0.0
        while t < 600.0 - 1e-9:
            st.update(dt, 6.0 if t < 300.0 else 0.0, exp)
            t += dt
        return st
    fine, coarse = run(1024 / SAMPLE_RATE * 4), run(2.0)
    assert coarse.film_mm("wood") == pytest.approx(fine.film_mm("wood"), rel=0.02)
    assert coarse.roof_flow_l_s == pytest.approx(fine.roof_flow_l_s, rel=0.05)


def test_canopy_interception_and_aftermath_drip():
    st = wet.WetnessStores(["leaves"], aftermath_tau_min=8.0, canopy_capacity_mm=1.0)
    exp = {"leaves": 1.0, "canopy": 1.0}
    fresh = wet.WetnessStores(["leaves"], canopy_capacity_mm=1.0)
    fresh.update(1.0, 10.0, exp)
    assert 1.0 - fresh.throughfall_fraction == pytest.approx(0.40, abs=0.01)   # a dry canopy catches most
    for _ in range(1800):                                     # an hour of steady 10 mm/h rain
        st.update(2.0, 10.0, exp)
    # Rutter store: catch 10-40 % (MASTER_PLAN 7.9), falling to ~10 % once the store reaches its
    # capacity S, which sustained moderate rain does (Rutter et al. 1971; S ~1 mm fills after
    # a few mm of rain); the excess above S drains within minutes
    assert 1.0 - st.throughfall_fraction == pytest.approx(0.11, abs=0.01 + 1e-9)
    assert 1.0 <= st.canopy_mm <= 1.1
    # steady state mass balance: catch = drip + evaporation (storage constant), within 1 %
    assert st.canopy_catch_mm_h == pytest.approx(st.canopy_drip_mm_h + st.canopy_evap_mm_h, rel=0.01)
    # wet-canopy evaporation during heavy rain is small: ~0.1-0.3 mm/h (Gash 1979)
    assert 0.05 <= st.canopy_evap_mm_h <= 0.35
    # after the rain stops the canopy keeps dripping (MASTER_PLAN 2.5: drip decay 1-30 min) ...
    st.update(60.0, 0.0, exp)
    d1, c1 = st.canopy_drip_mm_h, st.canopy_mm
    assert d1 > 0.1
    # ... at or above Rutter's drainage at capacity (Ds = 0.002 mm/min = 0.12 mm/h)
    assert d1 >= 0.12
    st.update(600.0, 0.0, exp)
    assert st.canopy_drip_mm_h < d1 and st.canopy_mm < c1
    st.decay(60.0, exp)
    assert st.canopy_drip_mm_h < 0.1 * d1 and st.canopy_mm < 0.1 * c1   # dry within the hour (tau 8 min)
    # drying follows the aftermath tau: a slower place stays wetter
    slow = wet.WetnessStores(["leaves"], aftermath_tau_min=30.0, canopy_capacity_mm=1.0)
    for _ in range(1800):
        slow.update(2.0, 10.0, exp)
    slow.decay(20.0, exp)
    fast = wet.WetnessStores(["leaves"], aftermath_tau_min=8.0, canopy_capacity_mm=1.0)
    for _ in range(1800):
        fast.update(2.0, 10.0, exp)
    fast.decay(20.0, exp)
    assert slow.canopy_mm > 3.0 * fast.canopy_mm
    # light rain (1 mm/h) fills the store only partly: catch stays well above 10 %
    light = wet.WetnessStores(["leaves"], aftermath_tau_min=8.0, canopy_capacity_mm=1.0)
    for _ in range(1800):
        light.update(2.0, 1.0, exp)
    assert light.canopy_mm < 1.0 and 1.0 - light.throughfall_fraction > 0.15
    # long steps stay stable (no oscillation / negative storage) and conserve water
    st2 = wet.WetnessStores(["leaves"], canopy_capacity_mm=1.0)
    for _ in range(20):
        c0 = st2.canopy_mm
        st2.update(600.0, 30.0, exp)
        assert 0.0 <= st2.canopy_mm <= 1.5
        gain = (st2.canopy_catch_mm_h - st2.canopy_drip_mm_h - st2.canopy_evap_mm_h) * 600.0 / 3600.0
        assert st2.canopy_mm - c0 == pytest.approx(gain, abs=1e-9)


def test_roof_gutter_mass_balance():
    # 1 mm on 1 m^2 is 1 L: steady gutter flow = rate x area / 3600 L/s
    st = wet.WetnessStores(["tile"], aftermath_tau_min=8.0, roof_area_m2=40.0)
    for _ in range(900):
        st.update(2.0, 9.0, {"tile": 1.0, "roof": 1.0})
    assert st.roof_flow_l_s == pytest.approx(9.0 * 40.0 / 3600.0, rel=0.03)
    q0 = st.roof_flow_l_s
    st.update(60.0, 0.0, {"tile": 1.0})
    assert 0.0 < st.roof_flow_l_s < q0                        # the roof keeps draining after the rain
    st.decay(60.0, {"tile": 1.0})
    assert st.roof_flow_l_s < 0.02 * q0


def test_tate_drop_diameter():
    # Tate's law with Harkins-Brown f ~ 0.6: D = (12 r sigma f / (rho g))^(1/3)
    r = 1.0e-3
    d_ref = (12.0 * r * 0.0728 * 0.6 / (998.0 * 9.81)) ** (1.0 / 3.0) * 1e3
    assert wet.tate_drop_diameter_mm(1.0) == pytest.approx(d_ref, rel=1e-9)
    assert d_ref == pytest.approx(3.77, abs=0.02)
    # 3-5.5 mm for 0.5-3 mm edges (MASTER_PLAN 7.9); grows with edge radius as r^(1/3)
    assert 2.9 <= wet.tate_drop_diameter_mm(0.5) < wet.tate_drop_diameter_mm(3.0) <= 5.5
    assert wet.tate_drop_diameter_mm(4.0) / wet.tate_drop_diameter_mm(0.5) == pytest.approx(2.0, rel=1e-9)


def _drip_count(node, inflow_mm_h, seconds, n, storage=0.0, sr=SAMPLE_RATE):
    events, b = [], 0
    t_blocks = int(round(seconds * sr / n))
    for b in range(t_blocks):
        for off, dmm, v in node.step(b, n, sr, inflow_mm_h, storage):
            assert 0 <= off < n
            events.append((b * n + off, dmm, v))
    return events


def test_drip_node_rate_follows_feed():
    # a 50 cm^2 leaf in 5 mm/h rain drips every ~7 s (MASTER_PLAN 7.9: 6.9 s for a 48 mm^3 drop)
    r_edge = 1.7
    node = wet.DripNode("leaf", 0.005, r_edge, 1.5, RngFactory(1), "drip/leaf")
    ev = _drip_count(node, 5.0, 600.0, 1024)
    q = 5.0 / 3600.0 * 0.005 * 1e6                            # mm^3/s
    interval = node.volume_mm3 / q
    assert interval == pytest.approx(6.9, rel=0.1)
    # each drop releases ~96 % of the pendant volume (4 % residual): count = t q / (0.96 V)
    expected = 600.0 * q / (0.96 * node.volume_mm3)
    assert len(ev) == pytest.approx(expected, abs=3)
    assert node.rate_hz == pytest.approx(1.0 / interval, rel=1e-9)
    d = np.array([e[1] for e in ev])
    assert np.all(np.abs(d / node.d_mm - 1.0) < 0.1)          # Tate diameter (+ small jitter)
    v = np.array([e[2] for e in ev])
    assert np.allclose(v, D.fall_speed_from_height(d, 1.5), rtol=1e-9)
    # rate doubles with feed; no feed, no drops; upstream storage keeps feeding after the rain
    node2 = wet.DripNode("leaf", 0.005, r_edge, 1.5, RngFactory(1), "drip/leaf")
    assert len(_drip_count(node2, 10.0, 600.0, 1024)) == pytest.approx(2 * expected, rel=0.05)
    node3 = wet.DripNode("leaf", 0.005, r_edge, 1.5, RngFactory(1), "drip/leaf")
    assert _drip_count(node3, 0.0, 60.0, 1024) == []
    node4 = wet.DripNode("leaf", 0.005, r_edge, 1.5, RngFactory(1), "drip/leaf")
    assert len(_drip_count(node4, 0.0, 120.0, 1024, storage=2.0)) > 5


@pytest.mark.parametrize("n", [64, 1000, 4096])
def test_drip_node_any_block_length(n):
    node = wet.DripNode("eave", 0.08, 2.0, 2.4, RngFactory(2), "drip/eave")
    ev = _drip_count(node, 6.0, 120.0, n)
    q = 6.0 / 3600.0 * 0.08 * 1e6
    assert len(ev) == pytest.approx(120.0 * q / (0.96 * node.volume_mm3), rel=0.03)
    t = np.array([e[0] for e in ev]) / SAMPLE_RATE
    assert np.all(np.diff(t) > 0.0)


def test_drip_node_goes_chaotic_at_high_flow():
    # Shaw (1984) leaky faucet: periodic at low flow, irregular at high flow
    def cv(inflow):
        node = wet.DripNode("x", 0.05, 1.5, 1.0, RngFactory(3), "drip/x")
        ev = _drip_count(node, inflow, 60.0, 1024)
        iv = np.diff([e[0] for e in ev]) / SAMPLE_RATE
        return iv.std() / iv.mean(), node.chaos
    cv_lo, chaos_lo = cv(2.0)
    cv_hi, chaos_hi = cv(60.0)
    assert chaos_lo == 0.0 and chaos_hi > 0.9
    assert cv_lo < 0.08 and cv_hi > 2.0 * cv_lo


# ============================================================================= storms
def _thunder_scene(approach=0.0, dist=8.0, fpm=3.0, arc="steady", seed=5):
    sc = _rain_scene(rate=8.0, seed=seed)
    sc.weather.thunder.enabled = True
    sc.weather.thunder.distance_km = dist
    sc.weather.thunder.approach_kmh = approach
    sc.weather.thunder.flashes_per_min = fpm
    sc.weather.storm_arc = arc
    return sc


@pytest.mark.parametrize("approach", [36.0, -36.0])
def test_storm_approach_speed_and_sign(approach):
    sc = _thunder_scene(approach=approach, dist=8.0)
    ss = stm.StormSystem(sc, RngFactory(1))
    block_s = 1024 / SAMPLE_RATE
    d = []
    for b in range(0, 2813, 1):                               # 60 s of blocks
        _, cells, _ = ss.step(b, b * block_s, block_s)
        d.append(cells[0].distance_km)
    # 36 km/h = 0.6 km per minute along the track (lateral miss 0.3 km: d ~ s at 8 km)
    change = d[-1] - d[0]
    assert change == pytest.approx(-math.copysign(0.6, approach), rel=0.03)
    assert np.all(np.sign(np.diff(d)) == -math.copysign(1.0, approach))


def test_storm_passes_overhead_and_recedes():
    sc = _thunder_scene(approach=60.0, dist=1.0)                # 1 km/min
    ss = stm.StormSystem(sc, RngFactory(1))
    t, d, brg = 0.0, [], []
    for b in range(140):                                     # 140 s in 1 s steps
        _, cells, _ = ss.step(b, t, 1.0)
        d.append(cells[0].distance_km)
        brg.append(cells[0].bearing_deg)
        t += 1.0
    i = int(np.argmin(d))
    assert d[i] == pytest.approx(0.3, abs=0.02)              # closest approach = lateral miss
    assert d[-1] > 1.0 and d[0] == pytest.approx(1.0, rel=1e-6)
    sweep = (brg[-1] - brg[0]) % 360.0
    assert 120.0 < sweep < 240.0                             # it crossed the sky to the other side


def test_storm_steady_distance_without_approach():
    sc = _thunder_scene(approach=0.0, dist=5.0)
    ss = stm.StormSystem(sc, RngFactory(1))
    ds = [ss.step(b, b * 0.5, 0.5)[1][0].distance_km for b in range(100)]
    assert np.allclose(ds, 5.0, rtol=1e-9)


def test_flash_statistics_poisson():
    fpm = 12.0
    sc = _thunder_scene(fpm=fpm)
    ss = stm.StormSystem(sc, RngFactory(9), block=4096)        # 85 ms flash slots (any block length)
    ss.step(0, 0.0, 0.0)
    minutes = 10.0
    fl = ss.flashes_between(0.0, minutes * 60.0)
    n = len(fl)
    # Poisson count: mean 120, sd ~11 -> 4 sigma band
    assert abs(n - fpm * minutes) < 4.0 * math.sqrt(fpm * minutes)
    t = np.array([e[0] for e in fl])
    assert np.all(np.diff(t) >= 0.0) and t.min() >= 0.0 and t.max() < minutes * 60.0
    iv = np.diff(t)
    # exponential inter-arrival times with rate fpm/60 (KS, fixed seed)
    assert stats.kstest(iv, "expon", args=(0.0, 60.0 / fpm)).pvalue > 0.01
    # counts per minute are Poisson: variance ~ mean (index of dispersion, 10 bins: chi2 9 dof, 0.1 %-99.9 %)
    counts = np.histogram(t, bins=10, range=(0.0, minutes * 60.0))[0]
    disp = counts.var(ddof=1) * 9 / counts.mean()
    assert stats.chi2.ppf(0.001, 9) < disp < stats.chi2.ppf(0.999, 9)
    # deterministic and independent of how the interval is queried
    parts = ss.flashes_between(0.0, 250.0) + ss.flashes_between(250.0, 600.0)
    assert [p[0] for p in parts] == [p[0] for p in fl]
    assert all(0.02 <= p[1].stroke_intensity <= 1.0 for p in fl)
    ss2 = stm.StormSystem(sc, RngFactory(9), block=4096)
    ss2.step(0, 0.0, 0.0)
    assert [p[0] for p in ss2.flashes_between(0.0, 600.0)] == [p[0] for p in fl]
    # a different seed gives different flashes; zero rate gives none
    ss3 = stm.StormSystem(sc, RngFactory(10), block=4096)
    ss3.step(0, 0.0, 0.0)
    assert [p[0] for p in ss3.flashes_between(0.0, 120.0)] != [p[0] for p in fl if p[0] < 120.0]
    sc0 = _thunder_scene(fpm=0.0)
    ss0 = stm.StormSystem(sc0, RngFactory(9))
    ss0.step(0, 0.0, 0.0)
    assert ss0.flashes_between(0.0, 60.0) == []


def test_cox_modulation_mean_one_and_continuous():
    ss = stm.StormSystem(_rain_scene(), RngFactory(4))
    t = np.arange(0.0, 3600.0, 0.37)
    m = np.array([ss.cox_modulation(x, 0.22) for x in t])
    # mean-one multiplicative modulation (log-normal with the -0.375 sigma^2 correction)
    assert m.mean() == pytest.approx(1.0, abs=0.06)
    assert np.all(m > 0.0)
    # continuous (cosine-interpolated knots): the largest step shrinks in proportion to the
    # step (Lipschitz), unlike a jump process; finest octave 0.1 s -> < 20 % per 21 ms block
    def max_step(dt):
        tb = np.arange(0.0, 60.0, dt)
        return np.max(np.abs(np.diff(np.log([ss.cox_modulation(x, 0.35) for x in tb]))))
    big, small = max_step(1024 / SAMPLE_RATE), max_step(256 / SAMPLE_RATE)
    assert big < 0.25
    assert big / small == pytest.approx(4.0, rel=0.25)
    # a pure function of time (seekable): same t, same value on a fresh system
    ss2 = stm.StormSystem(_rain_scene(), RngFactory(4))
    assert ss2.cox_modulation(1234.5, 0.22) == ss.cox_modulation(1234.5, 0.22)


def test_storm_arc_size_sorting():
    # arriving cell: few fat drops first (Dm factor > 1, low envelope), then the downpour
    a0, a1 = stm.arc_values("passing", 0.02), stm.arc_values("passing", 0.3)
    assert a0["dm"] > 1.3 and a0["env"] < 0.1
    assert a1["env"] > 1.0 and a1["dm"] < a0["dm"]
    tail = stm.arc_values("passing", 0.7)
    assert tail["dm"] < 1.0                                   # stratiform tail: smaller drops
    assert stm.arc_phase_name("passing", 0.3) == "downpour"
    assert stm.arc_values("steady", 0.5)["env"] == 1.0


# ============================================================================= World
def _states_equal(a, b):
    for k, v in a.__dict__.items():
        w = b.__dict__[k]
        if k == "dsd":
            assert v.__dict__ == w.__dict__
        elif k == "storms":
            assert [s.__dict__ for s in v] == [s.__dict__ for s in w]
        else:
            assert v == w, k


def test_world_deterministic_and_seed_sensitive():
    sc = _rain_scene(rate=6.0, seed=123, speed_mps=6.0)
    sc.weather.thunder.enabled = True
    _, a = _run_world(sc, 120)
    _, b = _run_world(sc, 120)
    for x, y in zip(a, b, strict=True):
        _states_equal(x, y)
    _, c = _run_world(sc, 120, seed=124)
    assert [s.gust for s in a] != [s.gust for s in c]
    assert [s.rain_rate_mm_h for s in a] != [s.rain_rate_mm_h for s in c]


def test_world_continuity_between_blocks():
    sc = _rain_scene(rate=5.0, seed=21, speed_mps=8.0)
    sc.weather.precipitation.character = "showery"
    sc.weather.precipitation.shower_period_min = 2.0
    _, st = _run_world(sc, 1400)                              # 30 s
    rate = np.array([s.rain_rate_mm_h for s in st])
    gust = np.array([s.gust for s in st])
    wind = np.array([s.wind_listener_mps for s in st])
    assert np.all(np.isfinite(rate)) and np.all(rate > 0.0)
    # rain rate: no jumps. The fastest legitimate change is the finest Cox octave (0.1 s
    # knots, sigma 0.35/sqrt(13), cosine interpolation): <= ~0.18 in ln(rate) per 21 ms
    # block; gust curtains enter low-passed over 2 s
    assert np.max(np.abs(np.diff(np.log(rate)))) < 0.25
    # gust and wind at the ears: the point gust carries the Kaimal inertial range, so it
    # does change every block; continuity means the steps are those of the process.
    # (a) no outliers among ~1400 Gaussian-like steps (max ~3.5 sigma expected)
    for x in (np.log(gust), np.log(wind)):
        d1 = np.diff(x)
        assert np.abs(d1).max() < 5.0 * d1.std()
    # (b) Kolmogorov structure function in the inertial range, D(tau) ~ tau^(2/3): 4x the lag
    # gives 4^(1/3) = 1.59x the step std (a jump/white component would pull this to 1)
    lg = np.log(gust)
    sf = {k: (lg[k:] - lg[:-k]).std() for k in (4, 16)}
    assert sf[16] / sf[4] == pytest.approx(4.0 ** (1.0 / 3.0), rel=0.1)
    # wind at the ears is exactly the log-law wind times the gust (same continuity)
    assert np.allclose(wind / gust, wind[0] / gust[0], rtol=1e-12)
    # with gusts off the rate is a smooth function of time: 4x shorter blocks -> ~4x smaller steps
    sc.weather.wind.turbulence_intensity = 0.0
    steps = {}
    for blk in (1024, 256):
        _, s2 = _run_world(sc, int(20.0 * SAMPLE_RATE / blk), block=blk)
        r = np.array([x.rain_rate_mm_h for x in s2])
        steps[blk] = np.max(np.abs(np.diff(r))) / r.mean()
    assert steps[1024] / steps[256] == pytest.approx(4.0, rel=0.3)


@pytest.mark.parametrize("block", [pytest.param(256, marks=pytest.mark.slow), 1000, 4096])
def test_world_any_block_length(block):
    # with gusts off, the rain rate is a pure function of time (arc x showers x Cox): block-size independent
    sc = _rain_scene(rate=7.0, seed=31, speed_mps=5.0, turbulence_intensity=0.0)
    seconds = 6.0
    _, ref = _run_world(sc, int(seconds * SAMPLE_RATE / 1024), block=1024)
    _, st = _run_world(sc, int(seconds * SAMPLE_RATE / block), block=block)
    t_ref = np.array([s.t for s in ref])
    r_ref = np.array([s.rain_rate_mm_h for s in ref])
    for s in st[:: max(1, len(st) // 40)]:
        if s.t <= t_ref[-1]:
            j = int(np.argmin(np.abs(t_ref - s.t)))
            if abs(t_ref[j] - s.t) < 1e-9:
                assert s.rain_rate_mm_h == pytest.approx(r_ref[j], rel=1e-9)
    assert all(s.gust == 1.0 for s in st)
    # wetness evolves with elapsed time, not with the number of blocks
    assert st[-1].wetness["concrete"] == pytest.approx(ref[-1].wetness["concrete"], rel=0.01)
    # with gusts on, statistics do not depend on the block length
    sc2 = _rain_scene(rate=7.0, seed=31, speed_mps=8.0)
    _, gs = _run_world(sc2, int(60.0 * SAMPLE_RATE / block), block=block)       # ~40 gust integral times
    g = np.array([s.gust for s in gs])
    assert g.std() == pytest.approx(gs[-1].turbulence_intensity, rel=0.4)
    assert g.mean() == pytest.approx(1.0, abs=0.1)


def test_world_physical_relationships():
    def first(sc):
        return wld.World(sc, RngFactory(sc.seed)).step(0, 0.0, 1024)
    light, heavy = first(_rain_scene(rate=1.0, seed=8)), first(_rain_scene(rate=20.0, seed=8))
    assert sum(heavy.class_flux.values()) > 3.0 * sum(light.class_flux.values())
    assert heavy.class_flux["large"] > 10.0 * light.class_flux["large"]
    # wind at the ears is below the 10 m report (log law) and the gust multiplies it
    s = first(_rain_scene(rate=1.0, seed=8, speed_mps=10.0))
    sc = _rain_scene(rate=1.0, seed=8, speed_mps=10.0)
    z0 = wnd.terrain_z0(sc.place.preset) if sc.place.z0_m is None else sc.place.z0_m
    assert s.wind_listener_mps == pytest.approx(wnd.wind_at_height(10.0, sc.listener.height_m, z0) * s.gust, rel=1e-9)
    assert s.wind_u10_mps == 10.0 and s.wind_listener_mps < 10.0 * s.gust
    # wind direction: from the east with the listener facing north -> azimuth -pi/2 (right)
    sc.weather.wind.from_deg = 90.0
    assert first(sc).wind_azimuth == pytest.approx(-math.pi / 2)
    # temperature -> speed of sound
    assert s.speed_of_sound == pytest.approx(speed_of_sound(s.temperature_c))
    # snow: no liquid drops; clear calm night -> stable, rain -> neutral
    sn = _rain_scene(rate=2.0, seed=8)
    sn.weather.precipitation.type = "snow"
    assert sum(first(sn).class_flux.values()) == 0.0
    night = _rain_scene(rate=0.0, seed=8, speed_mps=1.0)
    night.weather.precipitation.type = "none"
    night.weather.cloud_okta, night.weather.hour = 1.0, 2.0
    assert first(night).stability == "stable"
    assert first(_rain_scene(rate=8.0, seed=8)).stability == "neutral"


def test_world_sheltered_scenes_still_get_rain():
    # interiors are quieter because of the shelter path, not because the physics stops the rain
    for shelter in ("attic", "room_closed", "tent", "car", "umbrella"):
        sc = _rain_scene(rate=6.0, seed=3)
        sc.place.shelter = shelter
        w = wld.World(sc, RngFactory(3))
        st = w.step(0, 0.0, 1024)
        assert sum(st.class_flux.values()) > 0.0
        roofs = [p for p in w.layout if p.id.startswith("shelter.") and p.kind in ("surface", "membrane", "vehicle_body")
                 and p.normal[2] > 0.5]
        assert roofs and max(p.exposure for p in roofs) > 0.5, shelter


def test_daylight_hours():
    # 50 N: ~16.3 h of daylight at the June solstice, ~12 h at the equinox (geometric sunrise)
    rise, sset = wld.daylight_hours(50.0, "summer")
    assert sset - rise == pytest.approx(16.3, abs=0.3)
    rise, sset = wld.daylight_hours(50.0, "spring")
    assert sset - rise == pytest.approx(12.0, abs=0.3)
    rise, sset = wld.daylight_hours(-50.0, "summer")          # southern 'summer' is December
    assert sset - rise == pytest.approx(16.3, abs=0.3)
    assert wld.is_daytime(12.0, 50.0, "winter") and not wld.is_daytime(23.0, 50.0, "summer")


# ============================================================================= layout and frames
def test_listener_frame_conventions(scene):
    L = scene.listener
    L.position, L.height_m, L.heading_deg = (0.0, 0.0, 0.0), 1.7, 0.0
    north = lay.world_to_listener((0.0, 10.0, 1.7), L)
    east = lay.world_to_listener((10.0, 0.0, 1.7), L)
    west = lay.world_to_listener((-10.0, 0.0, 1.7), L)
    assert math.atan2(north[1], north[0]) == pytest.approx(0.0, abs=1e-12)        # front
    assert math.atan2(east[1], east[0]) == pytest.approx(-math.pi / 2, abs=1e-12)  # right
    assert math.atan2(west[1], west[0]) == pytest.approx(math.pi / 2, abs=1e-12)   # left
    assert north[2] == pytest.approx(0.0)                                           # ears origin
    L.heading_deg = 90.0
    east = lay.world_to_listener((10.0, 0.0, 1.7), L)
    assert math.atan2(east[1], east[0]) == pytest.approx(0.0, abs=1e-12)           # facing east
    # round trip, any heading and position
    L.position, L.heading_deg = (3.0, -2.0, 0.5), 137.0
    for p in ((1.0, 2.0, 3.0), (-40.0, 7.0, 0.0)):
        assert np.allclose(lay.listener_to_world(lay.world_to_listener(p, L), L), p, atol=1e-12)


def test_build_layout_deterministic_and_sane():
    sc = Scene()
    a, b = lay.build_layout(sc), lay.build_layout(sc)
    assert [(p.id, p.material, p.position, p.exposure, p.area_m2) for p in a] == \
           [(p.id, p.material, p.position, p.exposure, p.area_m2) for p in b]
    ids = [p.id for p in a]
    assert len(ids) == len(set(ids))
    for p in a:
        assert p.area_m2 > 0.0 and 0.0 <= p.exposure <= 1.0
        assert abs(np.linalg.norm(p.normal) - 1.0) < 1e-9
        assert "upwind_m" in p.params and "world_position" in p.params
    ground = [p for p in a if p.params.get("ground")]
    assert len(ground) == len(lay.RINGS) * lay.N_SECTORS
    # ground sectors tile the annulus 0.5-30 m exactly
    assert sum(p.area_m2 for p in ground) == pytest.approx(math.pi * (30.0 ** 2 - 0.5 ** 2), rel=1e-12)
    # material shares follow place.surfaces (area-weighted, +-15 % of the annulus per material)
    tot = sum(p.area_m2 for p in ground)
    share = {}
    for p in ground:
        share[p.material] = share.get(p.material, 0.0) + p.area_m2 / tot
    want = {lay.normalize_material(k): v / sum(sc.place.surfaces.values()) for k, v in sc.place.surfaces.items()}
    for m, v in want.items():
        assert abs(share.get(m, 0.0) - v) < 0.3, (m, share)
    sc2 = Scene()
    sc2.seed = sc.seed + 1
    c = lay.build_layout(sc2)
    assert [p.material for p in c if p.params.get("ground")] != [p.material for p in ground]


def test_layout_shelters_cover_the_ground_beneath():
    for shelter in ("attic", "room_closed", "underpass"):
        sc = Scene()
        sc.place.shelter = shelter
        pts = lay.build_layout(sc)
        ring0 = [p for p in pts if p.params.get("ground") and p.params["ring"] == 0]
        assert all(p.exposure == 0.0 and p.sheltered_by for p in ring0)   # nothing falls on the floor
    # an umbrella (r = 0.55 m) only shades inside the first ring (0.5-1.5 m): its own panel is
    # fully exposed and sheds water from 8 rib drips
    sc = Scene()
    sc.place.shelter = "umbrella"
    pts = lay.build_layout(sc)
    umb = next(p for p in pts if p.id == "shelter.umbrella")
    assert umb.exposure == 1.0 and umb.area_m2 == pytest.approx(math.pi * 0.55 ** 2)
    assert len([p for p in pts if p.id.startswith("shelter.umbrella.drip")]) == 8
    # rain on an open scene reaches every ground patch
    assert all(p.exposure == 1.0 for p in lay.build_layout(Scene()) if p.params.get("ground"))


def test_wind_driven_rain_hits_windward_walls():
    sc = Scene()
    sc.weather.wind.speed_mps, sc.weather.wind.from_deg = 10.0, 90.0      # from the east
    sc.place.objects = [
        SceneObject(id="wall_e", type="wall", position=(5.0, 0.0, 0.0), size=(3.0, 0.2, 2.5), rotation_deg=90.0),
        SceneObject(id="wall_w", type="wall", position=(-5.0, 0.0, 0.0), size=(3.0, 0.2, 2.5), rotation_deg=270.0),
    ]
    pts = {p.id: p for p in lay.build_layout(sc)}
    east, west = pts["obj.wall_e"], pts["obj.wall_w"]
    # ISO 15927-3 driving rain: windward faces wet, leeward nearly dry
    assert east.exposure > 0.5 and west.exposure < 0.1
    calm = Scene()
    calm.weather.wind.speed_mps = 0.0
    calm.place.objects = sc.place.objects
    assert {p.id: p for p in lay.build_layout(calm)}["obj.wall_e"].exposure < 0.1


def test_exposure_by_material_is_area_weighted():
    pts = lay.build_layout(Scene())
    ex = lay.exposure_by_material(pts)
    for m, v in ex.items():
        sel = [p for p in pts if p.material == m and p.kind != "drip_edge"]
        num = sum(p.exposure * p.area_m2 for p in sel)
        den = sum(p.area_m2 for p in sel)
        assert v == pytest.approx(num / den, rel=1e-12)
