"""Environment synthesis: wind, vegetation, aeolian tones, thunder, biophony, traffic, places.

Every test checks behaviour against physics, a published reference value or a stated
contract, with the source named next to each tolerance. Generators are driven by the real
physics World (WeatherState per block) and measured on their omnidirectional (W) pressure,
which in SN3D equals the emitted pressure in pascals (1.0 = 94 dB SPL).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from raingen.core.constants import BLOCK, N_AMBI, SAMPLE_RATE, speed_of_sound
from raingen.core.rng import RngFactory
from raingen.core.types import BlockContext, SourcePatch, StormCellState
from raingen.physics.layout import build_layout
from raingen.physics.world import World
from raingen.scene import Scene

SR = SAMPLE_RATE
P_REF = 20e-6


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def calm_scene(seed: int = 1234) -> Scene:
    """Dry open meadow at midday with nothing in it: each test adds only what it measures."""
    sc = Scene()
    sc.seed = seed
    w = sc.weather
    w.precipitation.type = "none"
    w.precipitation.rate_mm_h = 0.0
    w.thunder.enabled = False
    w.wind.speed_mps = 0.0
    w.temperature_c = 20.0
    w.humidity_pct = 70.0
    w.hour = 13.0
    w.season = "summer"
    w.cloud_okta = 8.0
    p = sc.place
    p.preset = "meadow"
    p.shelter = "open"
    p.surfaces = {"soil": 100.0}
    p.vegetation = {"distance_m": 30.0}
    p.emitters = []
    p.objects = []
    p.biophony = {"birds": "off", "insects": "off", "frogs": "off", "traffic": 0.0, "region": "temperate_europe"}
    return sc


def steady_wind(sc: Scene, u10: float) -> Scene:
    """Wind without gusts: a user turbulence intensity of 0 makes the gust multiplier exactly 1."""
    sc.weather.wind.speed_mps = float(u10)
    sc.weather.wind.turbulence_intensity = 0.0
    return sc


def run_env(sc: Scene, seconds: float, make, n: int = BLOCK, layout=None, keep_sh: bool = False):
    """Drive generators built by `make(scene, layout, rngf)` with the World for `seconds`.

    Returns (signals, weathers, markers): signals maps generator name -> W pressure (Pa), or
    -> (N_AMBI, T) when keep_sh (mono emissions encoded with the engine's Encoder).
    """
    rngf = RngFactory(int(sc.seed))
    if layout is None:
        layout = build_layout(sc)
    world = World(sc, rngf, block=n)
    gens = make(sc, layout, rngf)
    gens = gens if isinstance(gens, (list, tuple)) else [gens]
    nb = int(math.ceil(seconds * SR / n))
    out = {g.name: np.zeros((N_AMBI, nb * n) if keep_sh else nb * n) for g in gens}
    enc = None
    if keep_sh:
        from raingen.spatial.encoder import Encoder
        enc = Encoder(3, SR)
    weathers, markers = [], []
    for b in range(nb):
        t0 = b * n / SR
        w = world.step(b, t0, n)
        weathers.append(w)
        ctx = BlockContext(sr=SR, n=n, index=b, t0=t0, weather=w, rng=rngf, markers=[])
        for g in gens:
            for em in g.render(ctx) or ():
                sl = slice(b * n, (b + 1) * n)
                if keep_sh:
                    out[g.name][:, sl] += enc.encode(em, n)
                else:
                    out[g.name][sl] += em.signal[0] if em.is_ambisonic else em.signal
        markers.extend(ctx.markers)
    return out, weathers, markers


def env_gens(only=None, **kw):
    from raingen.synth.environment import make_environment_generators

    def make(sc, layout, rngf):
        gens = make_environment_generators(sc, layout, rngf, **kw)
        return [g for g in gens if only is None or g.name in only]
    return make


def spl(x) -> float:
    """RMS level in dB SPL (re 20 uPa)."""
    r = math.sqrt(float(np.mean(np.square(x))))
    return 20.0 * math.log10(max(r, 1e-15) / P_REF)


def spectrum(x, nfft: int = 8192):
    f, p = signal.welch(x, fs=SR, nperseg=min(nfft, x.size), detrend=False)
    return f, p


def centroid(x) -> float:
    f, p = spectrum(x)
    return float(np.sum(f * p) / max(np.sum(p), 1e-300))


def band_db(x, lo: float, hi: float) -> float:
    f, p = spectrum(x)
    sel = (f >= lo) & (f < hi)
    return 10.0 * math.log10(max(float(np.sum(p[sel]) * (f[1] - f[0])), 1e-300) / P_REF ** 2)


def a_weight_db(f) -> np.ndarray:
    """IEC 61672-1 A-weighting (dB) at frequencies f."""
    f2 = np.asarray(f, dtype=float) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                    * (f2 + 12194.0 ** 2))
    return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def spl_a(x) -> float:
    """A-weighted level in dB(A) (frequency-domain weighting of the Welch spectrum)."""
    f, p = spectrum(x)
    w = 10.0 ** (a_weight_db(np.maximum(f, 1.0)) / 10.0)
    w[0] = 0.0
    return 10.0 * math.log10(max(float(np.sum(p * w) * (f[1] - f[0])), 1e-300) / P_REF ** 2)


def block_jump_ratio(x, n: int) -> float:
    """Mean |x[k+1] - x[k]| across block boundaries over the mean everywhere (1 = no clicks)."""
    d = np.abs(np.diff(x))
    idx = np.arange(n - 1, x.size - 1, n)
    return float(np.mean(d[idx]) / max(np.mean(d), 1e-30))


def block_envelope(x, n: int = BLOCK) -> np.ndarray:
    m = x.size // n
    return np.sqrt(np.mean(np.square(x[: m * n].reshape(m, n)), axis=1))


def veg_patch(pid, species, pos, height, **params) -> SourcePatch:
    prm = {"species": species, "height_m": height}
    prm.update(params)
    radius = prm.get("canopy_radius_m", 2.5)
    return SourcePatch(id=pid, kind="vegetation", material="needles" if species == "conifer" else "leaves",
                       position=tuple(float(v) for v in pos), area_m2=math.pi * radius ** 2, params=prm)


# --------------------------------------------------------------------------
# gusts: the wind sound follows WeatherState.gust
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1024, 512, 777])
def test_gust_field_reruns_world_gusts_bit_exactly(n):
    """Contract (physics.world docstring): a GustProcess built and driven like the World's returns
    WeatherState.gust bit for bit, for any engine block length."""
    from raingen.synth.wind import GustField

    sc = calm_scene()
    sc.weather.wind.speed_mps = 9.0
    sc.place.preset = "suburb"
    rngf = RngFactory(sc.seed)
    world = World(sc, rngf, block=n)
    field = GustField(sc, rngf)
    assert field.LAYER == World.GUST_LAYER
    for b in range(400):
        w = world.step(b, b * n / SR, n)
        field.advance(b, w, n)
        assert field.value == w.gust                      # exact equality, not approximate
        assert field.at(0.0, 5.0) == pytest.approx(max(w.gust, 0.05), abs=1e-12)


@pytest.mark.parametrize("preset", ["meadow", "suburb"])
def test_wind_bed_envelope_tracks_weather_gust(preset):
    """The wind sound's level envelope follows WeatherState.gust (brief: correlation > 0.8), on open
    ground (I_u ~ 0.25 at the ears) and in a gusty suburb (I_u ~ 0.5). Envelope: 0.5 s windows."""
    sc = calm_scene()
    sc.weather.wind.speed_mps = 10.0
    sc.place.preset = preset
    out, ws, _ = run_env(sc, 45.0, env_gens(only=("wind",)))
    env = block_envelope(out["wind"])
    gust = np.array([w.gust for w in ws])[: env.size]
    sm = np.ones(24) / 24
    le = np.log(np.convolve(env ** 2, sm, mode="valid"))
    lg = np.log(np.convolve(gust, sm, mode="valid"))
    r = np.corrcoef(le, lg)[0, 1]
    assert r > 0.8, r


def test_wind_level_rises_steeply_with_speed():
    """Curle (1955): aerodynamic dipole intensity ~ U^6 (+18 dB per doubling); the felt LF
    pseudo-sound follows dynamic pressure (U^4 power). Together: 15-18.6 dB per doubling."""
    levels = {}
    for u in (6.0, 12.0):
        sc = steady_wind(calm_scene(), u)
        sc.place.z0_m = 0.03
        out, ws, _ = run_env(sc, 3.0, env_gens(only=("wind",)))
        levels[u] = (spl(out["wind"][SR // 2:]), band_db(out["wind"][SR // 2:], 200.0, 20000.0))
    d = levels[12.0][0] - levels[6.0][0]
    assert 15.0 <= d <= 18.6, levels
    # above the pseudo-sound band the dipole bed alone: U^6 plus its upward spectral shift
    assert levels[12.0][1] - levels[6.0][1] >= 17.5, levels


def test_wind_spectrum_moves_up_with_speed():
    """Strouhal scaling of turbulence noise: the spectral corner (centroid) rises with U."""
    cents = []
    for u in (4.0, 8.0, 16.0):
        sc = steady_wind(calm_scene(), u)
        out, _, _ = run_env(sc, 2.0, env_gens(only=("wind",)))
        cents.append(centroid(out["wind"][SR // 2:]))
    assert cents[0] < cents[1] < cents[2], cents
    assert cents[2] > 1.8 * cents[0], cents                 # f ~ U/L: x4 in U moves the corner a lot


def test_wind_gust_markers_follow_gusts():
    """EventMarker('gust') fires when the gust multiplier exceeds 1 + 1.6 TI (the stated rule)."""
    sc = calm_scene()
    sc.weather.wind.speed_mps = 12.0
    sc.place.preset = "suburb"
    out, ws, markers = run_env(sc, 60.0, env_gens(only=("wind",)))
    gm = [m for m in markers if m.kind == "gust"]
    assert gm, "a 60 s suburban 12 m/s wind has gusts"
    for m in gm:
        b = int(round(m.t * SR / BLOCK))
        w = ws[b]
        assert w.gust >= 1.0 + 1.6 * max(0.05, w.turbulence_intensity) - 1e-9
        assert m.data["wind_10m_mps"] == pytest.approx(w.wind_u10_mps * w.gust, abs=0.01)


# --------------------------------------------------------------------------
# microphone wind rumble
# --------------------------------------------------------------------------

def _mic(windscreen: str, u10: float = 14.0, shelter: str = "open", n: int = BLOCK):
    from raingen.synth.wind import MicRumbleGenerator

    sc = steady_wind(calm_scene(), u10)
    sc.place.shelter = shelter
    sc.weather.wind.mic_rumble = True
    sc.weather.wind.windscreen = windscreen
    out, ws, _ = run_env(sc, 3.0, lambda s, lay, r: MicRumbleGenerator(s, r), n=n)
    return out["mic_rumble"][SR // 2:], ws[-1]


def test_mic_rumble_windscreens_attenuate_and_lower_the_rumble():
    """Windscreen table contract: broadband attenuation none 0 / foam 12 / blimp 22 / dead_cat 32 dB,
    and the turbulence corner fc = V / (3 D) drops with screen size D (Strasberg 1988; Wuttke 1992)."""
    from raingen.synth.wind import WINDSCREENS

    lv, lf_share = {}, {}
    for ws_name in ("none", "foam", "blimp", "dead_cat"):
        y, w = _mic(ws_name)
        lv[ws_name] = spl(y)
        lf_share[ws_name] = band_db(y, 5.0, 40.0) - spl(y)
    for name in ("foam", "blimp", "dead_cat"):
        expected = WINDSCREENS[name]["atten_db"]
        assert lv["none"] - lv[name] == pytest.approx(expected, abs=2.0), (name, lv)
    # bigger screens push the remaining rumble to lower frequencies
    assert lf_share["none"] < lf_share["foam"] < lf_share["blimp"], lf_share


def test_mic_rumble_level_follows_dynamic_pressure():
    """Pseudo-sound at the capsule ~ dynamic pressure 0.5 rho U^2: +12 dB per doubling of U."""
    a, _ = _mic("foam", u10=7.0)
    b, _ = _mic("foam", u10=14.0)
    assert spl(b) - spl(a) == pytest.approx(12.04, abs=1.5)


def test_mic_rumble_is_click_free_for_any_block_length():
    """Signal hygiene of the capsule rumble at a non-power-of-two block length (the busy hygiene
    scene is a car, where the capsule sees no wind)."""
    y, _ = _mic("foam", n=700)
    assert np.all(np.isfinite(y))
    assert abs(float(np.mean(y))) < 0.1 * math.sqrt(float(np.mean(y ** 2)))
    assert block_jump_ratio(y[(SR // 2) % 700:], 700) < 1.5


def test_mic_rumble_is_off_inside_a_car_and_by_default():
    from raingen.synth.environment import make_environment_generators

    y, _ = _mic("none", shelter="car")
    assert np.max(np.abs(y)) == 0.0
    sc = steady_wind(calm_scene(), 14.0)
    gens = make_environment_generators(sc, build_layout(sc), RngFactory(1))
    assert "mic_rumble" not in [g.name for g in gens]      # default-off recorded-realism artifact


# --------------------------------------------------------------------------
# vegetation
# --------------------------------------------------------------------------

def _veg(patches, u10: float, seconds: float = 3.0, *, steady: bool = True, sc: Scene = None, n: int = BLOCK,
         preset: str = "meadow", z0: float = 0.03):
    from raingen.synth.vegetation import VegetationGenerator

    sc = sc or calm_scene()
    sc.place.preset = preset
    sc.place.z0_m = z0
    if steady:
        steady_wind(sc, u10)
    else:
        sc.weather.wind.speed_mps = float(u10)
    out, ws, _ = run_env(sc, seconds, lambda s, lay, r: VegetationGenerator(s, patches, r), n=n, layout=list(patches))
    return out["vegetation"], ws


def test_single_spruce_matches_bolin_measurement():
    """Bolin (2009, KTH): one 18 m spruce, microphone 3 m from it, U = 6.3 m/s -> 47.2 dB(A).
    The crown centre (0.6 H) is ~9.6 m from the ears. Tolerance +-3 dB (one site; the
    anemometer height of the measurement is not stated, so U is taken as the 10 m wind)."""
    tree = veg_patch("spruce", "conifer", (3.0, 0.0, 0.6 * 18.0 - 1.7), 18.0, canopy_radius_m=2.5)
    y, _ = _veg([tree], 6.3)
    assert spl_a(y[SR // 2:]) == pytest.approx(47.2, abs=3.0)


@pytest.mark.slow
def test_spruce_forest_edge_matches_bolin_measurement():
    """Bolin (2009, KTH) site 1: a 120 m x 30 m stand of 12 m spruces, microphone 29 m from the
    edge, U = 5.1 m/s -> 55.2 dB(A). Stand modelled at 400 trees/ha (5 m grid, typical of
    mature spruce); stand density is not reported, hence +-5 dB."""
    trees = []
    for i, x in enumerate(np.arange(29.0 + 2.5, 59.0, 5.0)):
        for j, y in enumerate(np.arange(-60.0 + 2.5, 60.0, 5.0)):
            trees.append(veg_patch(f"s{i}_{j}", "conifer", (x, y, 0.6 * 12.0 - 1.7), 12.0, canopy_radius_m=1.8))
    y, _ = _veg(trees, 5.1, seconds=2.0)
    assert spl_a(y[SR // 2:]) == pytest.approx(55.2, abs=5.0)


def test_vegetation_level_follows_fegeant_power_law():
    """Fegeant (1999) / Bolin (2009): L ~ 20 chi log10(U); for conifers Bolin measured 23-35 log U
    (6.9-10.5 dB per doubling of wind). Steady winds 5 -> 10 m/s at crown height."""
    tree = veg_patch("spruce", "conifer", (10.0, 0.0, 6.0), 18.0)
    lv = []
    for u in (5.0, 10.0):
        y, _ = _veg([tree], u)
        lv.append(spl(y[SR // 2:]))
    assert 6.9 <= lv[1] - lv[0] <= 10.5, lv


def test_vegetation_brightens_with_wind():
    """Fegeant (1999): the vegetation-noise spectrum moves up with wind speed (faster leaf impacts,
    higher Strouhal frequencies)."""
    tree = veg_patch("oak", "deciduous", (8.0, 0.0, 5.5), 12.0, canopy_radius_m=4.0)
    cents = []
    for u in (3.0, 6.0, 12.0):
        y, _ = _veg([tree], u)
        cents.append(centroid(y[SR // 2:]))
    assert cents[0] < cents[1] < cents[2], cents


def test_vegetation_louder_with_wind_and_silent_below_beaufort_2():
    """WMO Beaufort land descriptions: leaves rustle from force 2 (>= 1.6 m/s); below force 1
    (< 0.3 m/s) nothing moves."""
    tree = veg_patch("oak", "deciduous", (8.0, 0.0, 5.5), 12.0, canopy_radius_m=4.0)
    calm, _ = _veg([tree], 0.2)
    breeze, _ = _veg([tree], 3.0)
    fresh, _ = _veg([tree], 9.0)
    assert np.max(np.abs(calm)) == 0.0
    assert spl(fresh) > spl(breeze) + 10.0


def _kurtosis_hf(y) -> float:
    hp = signal.sosfilt(signal.butter(4, [2000.0, 12000.0], "bandpass", fs=SR, output="sos"), y)
    hp = hp[SR // 2:]
    return float(np.mean(hp ** 4) / np.mean(hp ** 2) ** 2 - 3.0)


def test_conifers_hiss_and_deciduous_leaves_are_crisp():
    """Fegeant / Bolin species spectra: conifer needles give a Gaussian 'whoosh' (excess kurtosis
    ~0), broadleaf rustle is made of sparse leaf impacts (impulsive, excess kurtosis >> 0)."""
    con = veg_patch("pine", "conifer", (8.0, 0.0, 5.5), 18.0)
    dec = veg_patch("oak", "deciduous", (8.0, 0.0, 5.5), 12.0, canopy_radius_m=4.0)
    y_con, _ = _veg([con], 5.0)
    y_dec, _ = _veg([dec], 5.0)
    k_con, k_dec = _kurtosis_hf(y_con), _kurtosis_hf(y_dec)
    assert abs(k_con) < 1.0, k_con
    assert k_dec > k_con + 1.5, (k_dec, k_con)


def test_conifer_whoosh_sits_at_the_needle_strouhal_frequency():
    """Needle vortex shedding f = St U / d with St ~ 0.2 (Strouhal; Fegeant 1999): 1.2 mm needles
    at 8 m/s crown wind peak near 1.33 kHz. Tolerance +-30 % (broad log-Gaussian peak)."""
    from raingen.physics.wind import wind_at_height

    con = veg_patch("pine", "conifer", (8.0, 0.0, 5.5), 18.0, needle_d_mm=1.2)
    u_crown = wind_at_height(8.0, 0.7 * 18.0, 0.03)
    y, _ = _veg([con], 8.0)
    f, p = spectrum(y[SR // 2:], nfft=16384)
    sm = np.convolve(p, np.ones(9) / 9, mode="same")
    f_peak = f[np.argmax(sm * (f > 200.0))]
    assert f_peak == pytest.approx(0.2 * u_crown / 0.0012, rel=0.3)


@pytest.mark.slow
def test_vegetation_level_fluctuates_like_bolin_measurements():
    """Bolin (2009) measured sigma(L_A) = 2.6-5.2 dB for vegetation noise in natural turbulence
    (20 min records). One spruce stand, 1 s levels over 120 s; tolerance widened to 1.5-7 dB for
    the short record."""
    stand = veg_patch("stand", "conifer", (15.0, 0.0, 6.0), 18.0, count=6)
    sc = calm_scene()
    sc.place.preset = "farm"
    y, _ = _veg([stand], 6.0, seconds=120.0, steady=False, sc=sc, z0=0.1)
    seg = y[: (y.size // SR) * SR].reshape(-1, SR)
    levels = 10.0 * np.log10(np.mean(seg ** 2, axis=1) + 1e-30)
    assert 1.5 <= float(np.std(levels[2:])) <= 7.0, float(np.std(levels[2:]))


@pytest.mark.slow
def test_upwind_trees_hear_the_gust_first():
    """Taylor's frozen turbulence: a tree s metres upwind meets each gust s / U earlier than the
    listener (MASTER_PLAN 7.6: trees 50 m upwind rustle ~5 s before the gust arrives)."""
    from raingen.physics.wind import wind_at_height

    sc = calm_scene()
    sc.weather.wind.from_deg = 0.0                     # from the north = straight ahead of the listener
    sc.weather.wind.integral_time_s = 3.0
    s_up = 45.0
    tree = veg_patch("upwind", "conifer", (s_up, 0.0, 6.0), 18.0)
    y, ws = _veg([tree], 8.0, seconds=120.0, steady=False, sc=sc, z0=0.03)
    env = np.log(block_envelope(y) ** 2 + 1e-30)
    lg = np.log(np.array([w.gust for w in ws])[: env.size])
    env, lg = env - env.mean(), lg - lg.mean()
    u_crown = wind_at_height(8.0, 0.7 * 18.0, 0.03)
    lags = np.arange(-400, 401)
    xc = [np.dot(env[max(0, -L):env.size - max(0, L)], lg[max(0, L):lg.size - max(0, -L)]) for L in lags]
    lead_s = lags[int(np.argmax(xc))] * BLOCK / SR     # >0: the tree's envelope leads the listener's gust
    assert lead_s == pytest.approx(s_up / u_crown, abs=1.5)


def test_bare_winter_branches_whistle_only_in_strong_wind():
    """Bare twigs shed vortices at f = 0.2 U / d (300-600 Hz for twigs at 11.5 m/s, MASTER_PLAN
    7.6); the whistle layer appears from Beaufort 5-6 and is strongest without leaves."""
    sc = calm_scene()
    sc.weather.season = "winter"
    tree = veg_patch("bare", "deciduous", (8.0, 0.0, 5.5), 12.0, canopy_radius_m=4.0, leaf_fraction=0.05,
                     twig_diameter_m=0.004)
    y_lo, _ = _veg([tree], 4.0, sc=sc)
    sc2 = calm_scene()
    sc2.weather.season = "winter"
    y_hi, _ = _veg([tree], 12.0, sc=sc2)
    # the 250-900 Hz whistle band gains much more than the overall level
    gain_band = band_db(y_hi, 250.0, 900.0) - band_db(y_lo, 250.0, 900.0)
    gain_all = spl(y_hi) - spl(y_lo)
    assert gain_band > gain_all + 3.0, (gain_band, gain_all)


# --------------------------------------------------------------------------
# aeolian tones
# --------------------------------------------------------------------------

def _wire(u10: float, d: float = 0.005, f1: float = 0.0, seconds: float = 3.0, dist: float = 15.0,
          height: float = 10.0, length: float = 40.0):
    """A horizontal wire perpendicular to the wind (wind from the north, wire running east-west)."""
    from raingen.synth.aeolian import AeolianGenerator

    sc = steady_wind(calm_scene(), u10)
    sc.place.z0_m = 0.03
    sc.weather.wind.from_deg = 0.0
    wire = SourcePatch(id="wire", kind="wire", material="metal_sheet", position=(dist, 0.0, height - 1.7),
                       area_m2=d * length, params={"type": "wire", "diameter_m": d, "length_m": length,
                                                   "f1_hz": f1, "height_m": height, "axis_deg": 90.0})
    out, ws, _ = run_env(sc, seconds, lambda s, lay, r: AeolianGenerator(s, [wire], r), layout=[wire])
    return out["aeolian"], ws


def test_aeolian_tone_at_the_strouhal_frequency():
    """Vortex shedding f = St U / d, St = 0.2 (Strouhal 1878; Roshko 1954): a 5 mm wire at 10 m/s
    sings at 400 Hz (MASTER_PLAN 7.6). Tolerance 3 % (turbulent jitter 0.35 TI x 0.7 %)."""
    y, ws = _wire(10.0)
    f, p = spectrum(y[SR // 2:], nfft=32768)
    f_peak = f[np.argmax(p)]
    assert f_peak == pytest.approx(400.0, rel=0.03)


def test_aeolian_tone_only_above_onset_speed():
    """WMO Beaufort 6 (10.8-13.8 m/s): 'whistling heard in telegraph wires'; below force 4-5 the
    wire is silent."""
    calm, _ = _wire(5.0)
    windy, _ = _wire(12.0)
    assert np.max(np.abs(calm)) == 0.0
    assert spl(windy) > 20.0


def test_aeolian_level_follows_curle_dipole_law():
    """Phillips (1956) / Curle (1955): p_rms ~ rho U^3 St C_L' sqrt(l_c ...) / c: +60 log10 of the
    speed ratio (12 -> 18 m/s: +10.6 dB)."""
    a, _ = _wire(12.0)
    b, _ = _wire(18.0)
    assert spl(b) - spl(a) == pytest.approx(60.0 * math.log10(1.5), abs=1.5)


def test_aeolian_wire_level_matches_phillips_estimate():
    """Phillips (1956): p^2 = l_c St^2 rho^2 U^6 C_L'^2 / (16 c^2) (2/r) atan(L/2r) with l_c = 4 d,
    C_L' = 0.25 -> 5 mm wire, 40 m long, 12 m/s, heard from 15 m: ~45 dB SPL. The rendered
    tone adds turbulent amplitude flicker and broadband shedding: +-4 dB."""
    from raingen.core.constants import RHO_AIR

    d, L, r, U = 0.005, 40.0, 15.0, 12.0
    c = speed_of_sound(20.0)
    p2 = 4 * d * 0.2 ** 2 * RHO_AIR ** 2 * U ** 6 * 0.25 ** 2 / (16 * c ** 2) * (2 / r) * math.atan(L / (2 * r))
    expected = 10 * math.log10(p2 / P_REF ** 2)
    assert expected == pytest.approx(45.0, abs=1.5)
    y, _ = _wire(U, length=L, dist=r)
    assert spl(y[SR // 2:]) == pytest.approx(expected, abs=4.0)


def test_aeolian_lock_in_onto_string_mode():
    """Aeolian harp lock-in (Blevins): within a few per cent of a string mode the shedding locks
    onto it. Shedding 400 Hz, f1 = 98 Hz -> 4th mode 392 Hz (2 % detuned) wins."""
    y, _ = _wire(10.0, f1=98.0)
    f, p = spectrum(y[SR // 2:], nfft=32768)
    f_peak = f[np.argmax(p)]
    assert abs(f_peak - 392.0) < abs(f_peak - 400.0) or abs(f_peak - 392.0) < 4.0, f_peak


def test_wind_chime_modes_follow_free_free_beam():
    """Euler-Bernoulli free-free bar: bending modes in the ratio beta_k^2 = 1 : 2.757 : 5.404 : 8.933."""
    from raingen.synth.aeolian import _ChimeVoice

    v = _ChimeVoice("c", (2.0, 0.0, 0.5), 2.2, "wind", SR, 0.03, np.random.default_rng(0), n_tubes=1,
                    base_hz=500.0)
    f = np.sort(v.bank.freqs)
    assert f / f[0] == pytest.approx([1.0, 2.757, 5.404, 8.933], rel=0.01)


# --------------------------------------------------------------------------
# thunder
# --------------------------------------------------------------------------

def _thunder(distance_km: float, after_s: float = 3.0, *, intensity: float = 0.5, cg: bool = True,
             temp: float = 20.0, rh: float = 70.0, flash_id: str = "t", until_marker: bool = False,
             full: bool = False, n: int = BLOCK):
    """One explicit flash, timed so its first arrival (closest channel point) lands ~0.3 s into the render.

    Returns (W pressure from the first arrival on, markers, event)."""
    from raingen.synth.thunder import ThunderGenerator

    sc = calm_scene()
    sc.weather.temperature_c = temp
    sc.weather.humidity_pct = rh
    sc.weather.wind.speed_mps = 1.0
    rngf = RngFactory(sc.seed)
    world = World(sc, rngf, block=n)
    gen = ThunderGenerator(sc, [], rngf)
    c = speed_of_sound(temp)
    ys, markers, ev = [], [], None
    b = 0
    stop_at = None
    while True:
        t0 = b * n / SR
        w = world.step(b, t0, n)
        ctx = BlockContext(sr=SR, n=n, index=b, t0=t0, weather=w, rng=rngf, markers=[])
        if b == 0:
            # the channel is a pure function of flash_id: probe its closest point (a branch can pass much
            # nearer than the strike point), then launch the flash so that the first arrival is at 0.3 s
            probe = gen.strike(ctx, 0.0, distance_km, 0.0, intensity=intensity, cg=cg, flash_id=flash_id)
            r_min = probe.r_min if probe is not None else 1000.0 * distance_km
            gen.events.clear()
            gen.pending_markers.clear()
            ctx.markers.clear()
            t_flash = 0.3 - r_min / c                       # the flash happened before the render started
            ev = gen.strike(ctx, t_flash, distance_km, 0.0, intensity=intensity, cg=cg, flash_id=flash_id)
            assert ev is not None
            stop_at = ev.t_first + after_s if not full else ev.end_sample / SR + 0.5
        ems = gen.render(ctx)
        ys.append(ems[0].signal[0] if ems else np.zeros(n))
        markers.extend(ctx.markers)
        b += 1
        if until_marker and any(m.kind == "thunder" for m in markers):
            break
        if t0 > stop_at:
            break
    y = np.concatenate(ys)
    i0 = max(int(round(ev.t_first * SR)), 0)
    return y[i0:], markers, ev


def _fast_max_db(y) -> float:
    k = int(0.125 * SR)
    e = np.convolve(y ** 2, np.ones(k) / k, mode="valid")
    return 10.0 * math.log10(max(float(np.max(e)), 1e-30) / P_REF ** 2)


@pytest.mark.parametrize("temp,s_per_km", [(-10.0, 3.075), (20.0, 2.914), (35.0, 2.842)])
def test_thunder_follows_lightning_after_distance_over_c(temp, s_per_km):
    """Flash-to-bang = r / c(T): 3.075 s/km at -10 C, 2.914 at 20 C, 2.842 at 35 C (MASTER_PLAN 7.7,
    c = 331.3 sqrt(1 + T/273.15)). The 'lightning' marker comes first, 'thunder' at the closest
    channel point's arrival."""
    _, markers, ev = _thunder(2.0, temp=temp, until_marker=True)
    kinds = [m.kind for m in markers]
    assert kinds.index("lightning") < kinds.index("thunder")
    lt = next(m for m in markers if m.kind == "lightning")
    th = next(m for m in markers if m.kind == "thunder")
    d_km = th.data["distance_km"]
    delay = th.t - lt.t
    assert delay == pytest.approx(1000.0 * d_km / speed_of_sound(temp), abs=1e-3)
    assert delay / d_km == pytest.approx(s_per_km, abs=0.01)
    assert lt.data["flash_to_bang_s"] == pytest.approx(delay, abs=1e-3)
    assert ev.r_min == pytest.approx(1000.0 * d_km, abs=0.1)


def test_every_flash_gets_a_thunder_marker_at_its_first_arrival():
    """Marker contract (raingen.synth.thunder docstring): every flash, CG or IC, audible or not, emits
    'lightning' at t_flash and exactly one 'thunder' at t_flash + r_min / c(T). A CG stroke's closest
    point is about the slant range to its strike point (branches may come a little closer); an IC
    channel runs at 5-8 km and never dips below 3 km above the ground."""
    from raingen.synth.thunder import ThunderGenerator

    sc = calm_scene()
    rngf = RngFactory(sc.seed)
    w = World(sc, rngf).step(0, 0.0, BLOCK)
    gen = ThunderGenerator(sc, [], rngf)
    ctx = BlockContext(sr=SR, n=BLOCK, index=0, t0=0.0, weather=w, rng=rngf, markers=[])
    c = speed_of_sound(w.temperature_c)
    shots = [("cg2", 2.0, True), ("cg6", 6.0, True), ("cg40", 40.0, True), ("ic3", 3.0, False), ("ic12", 12.0, False)]
    for fid, d, cg in shots:
        gen.strike(ctx, 0.01, d, 45.0, intensity=0.5, cg=cg, flash_id=fid)
    assert all(isinstance(m.data["audible"], bool) for m in gen.pending_markers)
    gen.events.clear()                                                    # markers only: flush them all
    late = BlockContext(sr=SR, n=BLOCK, index=10_000, t0=200.0, weather=w, rng=rngf, markers=[])
    gen.render(late)
    for fid, d, cg in shots:
        lt = [m for m in ctx.markers if m.kind == "lightning" and m.data["id"] == fid]
        th = [m for m in late.markers if m.kind == "thunder" and m.data["id"] == fid]
        assert len(lt) == 1 and len(th) == 1, fid
        r_min = 1000.0 * th[0].data["distance_km"]
        assert th[0].t - lt[0].t == pytest.approx(r_min / c, abs=1e-3), fid       # r / c(T) (MASTER_PLAN 7.7)
        if cg:
            slant = math.hypot(1000.0 * d, 1.7)
            assert 0.4 * slant <= r_min <= slant + 1.0, (fid, r_min, slant)
        else:
            assert r_min >= 3000.0 - 1.7, (fid, r_min)
    assert [m.t for m in late.markers] == sorted(m.t for m in late.markers)


def test_close_strike_cracks_and_distant_thunder_rumbles():
    """Air absorption (ISO 9613-1: 1 km costs 5 dB at 1 kHz, 78 dB at 8 kHz) strips the crack of near
    strikes within a few km; far thunder keeps only the low rumble whose spectral peaks lie at
    ~28-50 Hz (Holmes et al. 1971; range < 4-125 Hz). The crack is judged on the first 250 ms."""
    near, _, ev_n = _thunder(0.5, after_s=2.0, flash_id="near")
    far, _, ev_f = _thunder(12.0, after_s=4.0, flash_id="far")
    q_n, q_f = near[: SR // 4], far[: SR // 4]
    hf_near = band_db(q_n, 2000.0, 16000.0) - spl(q_n)
    hf_far = band_db(q_f, 2000.0, 16000.0) - spl(q_f)
    assert ev_n.crack is not None and ev_f.crack is None
    # an N-wave spectrum peaking at fN ~ 50 Hz and falling as 1/f puts ~fN / 2 kHz (-16 dB) of its energy
    # above 2 kHz; 0.5 km of ISO 9613-1 costs 4.5-11 dB at 2-4 kHz; the crack layer only adds to that
    assert hf_near > -20.0, hf_near
    assert hf_far < hf_near - 30.0, (hf_near, hf_far)        # 12 km of ISO 9613-1 costs > 100 dB at 2 kHz
    assert centroid(far) < 150.0
    assert band_db(far, 10.0, 200.0) - spl(far) > -1.5       # >= 70 % of the far energy below 200 Hz


def test_distant_thunder_spectrum_peaks_low():
    """Thunder at 2.5 km (routed check: a clip measured a 960 Hz centroid, far too bright): the N-wave
    source spectrum peaks at f ~ 0.66 / T with Few's period T ~ 20 ms there, and ISO 9613-1 removes
    ~23 dB at 2 kHz over 2.5 km. Spectral peak inside Holmes et al.'s (1971) 4-125 Hz range (their
    28-50 Hz means are unverified per MASTER_PLAN 7.7) and a centroid below 200 Hz."""
    y, _, ev = _thunder(2.5, after_s=4.0, flash_id="mid")
    f, p = spectrum(y, nfft=16384)
    sm = np.convolve(p, np.ones(5) / 5, mode="same")
    f_peak = float(f[np.argmax(sm * (f > 4.0))])
    assert 4.0 <= f_peak <= 125.0, f_peak
    assert centroid(y) < 200.0, centroid(y)
    assert 20.0 <= ev.f_nwave <= 60.0, ev.f_nwave             # 0.66 / T, T = T0 (r / R0)^(1/4) (Few 1969)


@pytest.mark.slow
def test_thunder_level_falls_with_distance():
    """Weak-shock N-waves decay as r^-1 (R0/r)^(1/4) (Few 1969; Ribner & Roy 1982): 11.9 dB per factor 3
    once the channel is compact. A 3-6 km tall channel heard from 1-3 km is not compact (its lower part is
    seen end-on and smeared, while from 3 km it is broadside), so the median fast maximum over several
    channels falls more slowly there (>= 2 dB) and faster beyond (>= 6 dB from 3 to 9 km, with absorption
    and the refraction shadow). Calibration: ~100-104 dB median fast max at 1 km for intensity 0.5 (Few's
    ~0.2 % acoustic efficiency); thunder level data are sparse, hence 95-112 dB."""
    med = {}
    for d in (1.0, 3.0, 9.0):
        med[d] = float(np.median([_fast_max_db(_thunder(d, after_s=3.0, flash_id=f"lv{k}")[0]) for k in range(4)]))
    assert med[1.0] - med[3.0] >= 2.0, med
    assert med[3.0] - med[9.0] >= 6.0, med
    assert 95.0 <= med[1.0] <= 112.0, med


def test_thunder_level_rises_with_stroke_intensity():
    """Few: the N-wave amplitude grows with the energy per unit length E0 (x7 from intensity 0.2 to 0.7)."""
    weak = _fast_max_db(_thunder(3.0, after_s=3.0, intensity=0.2, flash_id="geo")[0])
    strong = _fast_max_db(_thunder(3.0, after_s=3.0, intensity=0.7, flash_id="geo")[0])
    assert strong > weak + 3.0, (weak, strong)


def test_stronger_strokes_are_deeper():
    """Few: relaxation radius R0 = sqrt(E0 / (pi p0)), source frequency f ~ c0 sqrt(p0 / E0)
    (485 Hz at E0 = 50 kJ/m): bigger strokes are deeper."""
    from raingen.synth.thunder import source_frequency_hz

    assert source_frequency_hz(5e4, 15.0) == pytest.approx(485.0, rel=0.03)
    weak = _thunder(2.0, after_s=3.0, intensity=0.15, flash_id="deep")[0]
    strong = _thunder(2.0, after_s=3.0, intensity=0.9, flash_id="deep")[0]
    assert centroid(strong) < centroid(weak), (centroid(strong), centroid(weak))


def _audible_s(y, db: float = 55.0) -> float:
    """Time between the first and last 125 ms (fast) level above `db` dB SPL."""
    k = int(0.125 * SR)
    e = np.convolve(y ** 2, np.ones(k) / k, mode="same")
    on = np.nonzero(e > (P_REF * 10 ** (db / 20.0)) ** 2)[0]
    return float(on[-1] - on[0]) / SR if on.size else 0.0


@pytest.mark.slow
def test_thunder_roll_is_longer_for_near_strikes_and_strong_strokes():
    """Ribner & Roy: the roll lasts (r_far - r_near) / c over the channel; a 5 km tall channel spans
    ~12 s seen from 1 km but only ~3.4 s seen from 10 km (sqrt(D^2 + H^2) - D), and in-cloud branches
    add to both. Heard over a 55 dB moderate-rain background: near and strong strokes roll longer."""
    near, _, _ = _thunder(1.0, full=True, flash_id="roll")
    far, _, _ = _thunder(10.0, full=True, flash_id="roll")
    assert _audible_s(near) > 1.2 * _audible_s(far), (_audible_s(near), _audible_s(far))
    assert _audible_s(near) < 45.0                             # 'a roll rarely exceeds 30-40 s' (MASTER_PLAN 7.7)
    weak, _, _ = _thunder(3.0, full=True, intensity=0.2, flash_id="roll")
    strong, _, _ = _thunder(3.0, full=True, intensity=0.8, flash_id="roll")
    assert _audible_s(strong) > _audible_s(weak) + 2.0, (_audible_s(weak), _audible_s(strong))


def test_thunder_signal_hygiene_for_any_block_length():
    """A strike rendered with 700- and 1024-frame blocks: finite, no DC, no clicks at block boundaries,
    and the same level (the rumble noise is framed independently of the block length): +-1 dB."""
    y7, _, _ = _thunder(2.0, after_s=3.0, flash_id="hyg", n=700)
    y10, _, _ = _thunder(2.0, after_s=3.0, flash_id="hyg", n=1024)
    for y, n in ((y7, 700), (y10, 1024)):
        assert np.all(np.isfinite(y))
        assert abs(float(np.mean(y))) < 0.05 * math.sqrt(float(np.mean(y ** 2)))
        off = (n - int(round(0.3 * SR)) % n) % n              # y starts at the first arrival, not a block edge
        assert block_jump_ratio(y[off:], n) < 1.5, n
    m = min(y7.size, y10.size)
    assert spl(y7[:m]) == pytest.approx(spl(y10[:m]), abs=1.0)


def test_cg_fraction_sets_flash_types():
    """StormCellState.cg_fraction is the share of cloud-to-ground flashes."""
    from raingen.synth.thunder import ThunderGenerator

    sc = calm_scene()
    rngf = RngFactory(sc.seed)
    w = World(sc, rngf).step(0, 0.0, BLOCK)
    for frac, lo, hi in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.4, 0.15, 0.65)):
        gen = ThunderGenerator(sc, [], rngf)
        cell = StormCellState(id="c", distance_km=25.0, bearing_deg=90.0, flash_rate_per_min=0.0,
                              stroke_intensity=0.5, cg_fraction=frac, rain_contribution=0.0)
        types = []
        for j in range(30):
            ctx = BlockContext(sr=SR, n=BLOCK, index=0, t0=0.0, weather=w, rng=rngf, markers=[])
            gen._spawn(ctx, 0.01, cell, 0.5, j, 0)
            types += [m.data["type"] for m in ctx.markers if m.kind == "lightning"]
            gen.events.clear()
        share = types.count("cg") / len(types)
        assert lo <= share <= hi, (frac, share)


def test_vectorised_refraction_matches_atmosphere_model():
    """thunder.refraction_gains_db must equal raingen.physics.atmosphere.refraction_gain_db."""
    from raingen.physics.atmosphere import refraction_gain_db
    from raingen.synth.thunder import refraction_gains_db

    rng = np.random.default_rng(3)
    r = rng.uniform(50.0, 40000.0, 300)
    wc = rng.uniform(-15.0, 15.0, 300)
    hs = rng.uniform(1.0, 8000.0, 300)
    for stab in ("stable", "neutral", "unstable"):
        got = refraction_gains_db(r, wc, stab, hs, 1.7, 12.0)
        ref = [refraction_gain_db(a, b, stab, c, 1.7, 12.0) for a, b, c in zip(r, wc, hs)]
        assert got == pytest.approx(ref, abs=1e-9)


def test_cg_channel_closest_point_is_near_the_strike_point():
    """A CG stroke's first arrival comes from about the slant range to its strike point: below the
    cloud base the stepped leader descends with a few hundred metres of lateral meander (Rakov &
    Uman 2003), so over many strokes 2 km away the closest channel point is typically 70-100 % of
    the slant range and never farther."""
    from raingen.synth.thunder import lightning_channel

    ratios = []
    for k in range(20):
        pts = lightning_channel(np.random.default_rng(100 + k), 2.0, 0.0, 0.0, True)
        r = np.linalg.norm(pts[np.all(np.isfinite(pts), axis=1)], axis=1)
        ratios.append(float(np.min(r)) / math.hypot(2000.0, 1.7))
    assert max(ratios) <= 1.0 + 1e-6
    assert 0.7 <= float(np.median(ratios)) <= 1.0, ratios


def test_lightning_channel_tortuosity():
    """Few (1969) / Hill (1968): ~16 deg mean direction change between channel segments; a CG
    channel reaches the ground at the strike point and starts several km up."""
    from raingen.synth.thunder import lightning_channel, split_polylines

    pts = lightning_channel(np.random.default_rng(5), 3.0, 90.0, 0.0, True)
    main = split_polylines(pts)[0]
    v = np.diff(main, axis=0)
    v = v[np.linalg.norm(v, axis=1) > 1.0]
    u = v / np.linalg.norm(v, axis=1)[:, None]
    ang = np.degrees(np.arccos(np.clip(np.sum(u[1:] * u[:-1], axis=1), -1.0, 1.0)))
    assert 10.0 < float(np.median(ang)) < 30.0
    assert main[-1, 2] == pytest.approx(-1.7, abs=1e-6)
    assert np.hypot(main[-1, 0], main[-1, 1]) == pytest.approx(3000.0, rel=1e-6)
    assert np.max(main[:, 2]) > 3000.0


# --------------------------------------------------------------------------
# biophony and traffic
# --------------------------------------------------------------------------

@pytest.mark.parametrize("temp_c", [13.0, 15.0, 20.0, 25.0, 30.0])
def test_dolbear_law(temp_c):
    """Dolbear (1897): T_F = 50 + (N60 - 40) / 4 for the snowy tree cricket."""
    from raingen.synth.biophony import dolbear_chirps_per_min

    n60 = dolbear_chirps_per_min(temp_c)
    t_f = 1.8 * temp_c + 32.0
    assert 50.0 + (n60 - 40.0) / 4.0 == pytest.approx(t_f, abs=1e-9)


def _crickets(temp_c: float, seconds: float = 20.0, region: str = "north_america"):
    from raingen.synth.biophony import BiophonyGenerator

    sc = calm_scene()
    sc.weather.temperature_c = temp_c
    sc.weather.hour = 23.0
    sc.place.biophony = {"birds": "off", "insects": "on", "frogs": "off", "traffic": 0.0, "region": region}
    holder = {}

    def make(s, lay, r):
        g = BiophonyGenerator(s, lay, r)
        g.chorus = None                       # measure the individual singers only
        g.cicadas = []
        holder["g"] = g
        return g
    out, ws, _ = run_env(sc, seconds, make)
    return out["biophony"], holder["g"]


@pytest.mark.parametrize("temp_c", [16.0, 26.0])
def test_tree_cricket_chirp_rate_follows_dolbear(temp_c):
    """Rendered snowy tree crickets chirp at Dolbear's rate N60 = 4 (T_F - 50) + 40; each insect has
    a +-5 % individual rate jitter, so the chorus periodicity is checked to +-8 %."""
    from raingen.synth.biophony import dolbear_chirps_per_min

    y, g = _crickets(temp_c)
    voices = [v for v in g.insect_bank.voices if v.species == "tree_cricket"]
    assert voices
    for v in voices:
        assert v.pattern(temp_c)[0] == pytest.approx(dolbear_chirps_per_min(temp_c) / 60.0 * v.jit, rel=1e-9)
    expected_hz = dolbear_chirps_per_min(temp_c) / 60.0
    fc = voices[0].pattern(temp_c)[3]
    band = signal.sosfilt(signal.butter(4, [fc * 0.85, fc * 1.15], "bandpass", fs=SR, output="sos"), y)
    env = signal.sosfilt(signal.butter(2, 20.0, fs=SR, output="sos"), np.abs(band))[::480]   # 100 Hz
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[env.size - 1:]
    lag_lo, lag_hi = int(100 / (expected_hz * 1.4)), int(100 / (expected_hz * 0.7))
    lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
    assert 100.0 / lag == pytest.approx(expected_hz, rel=0.08)


def _activity(**weather):
    from raingen.synth.biophony import BiophonyGenerator

    sc = calm_scene()
    sc.place.biophony = {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0,
                         "region": "temperate_europe"}
    g = BiophonyGenerator(sc, [], RngFactory(1))
    rngf = RngFactory(1)
    for k, v in weather.items():
        setattr(sc.weather, k, v)
    rate = weather.get("rate", 0.0)
    sc.weather.precipitation.type = "rain" if rate > 0 else "none"
    sc.weather.precipitation.rate_mm_h = rate
    w = World(sc, rngf).step(0, 0.0, BLOCK)
    w.rain_rate_mm_h = rate
    return g._targets(w, BLOCK / SR)


def test_birds_sing_at_dawn_and_by_day_not_at_night():
    """Dawn chorus around sunrise (Cooper 1969 declination for the season and latitude), scattered
    daytime song, silence at night (owls only)."""
    from raingen.synth.biophony import sun_times

    rise, sset = sun_times(50.0, "spring")
    dawn = _activity(hour=rise + 0.3, season="spring", latitude_deg=50.0)
    noon = _activity(hour=13.0, season="spring", latitude_deg=50.0)
    night = _activity(hour=2.0, season="spring", latitude_deg=50.0)
    assert dawn["birds"] > noon["birds"] > 0.0
    assert night["birds"] == pytest.approx(0.0, abs=1e-6)
    assert night["owls"] > 0.0 and noon["owls"] == 0.0


def test_rain_and_wind_silence_birds():
    """Lengagne & Slater (2002): birds sing much less in rain; strong wind suppresses song."""
    dry = _activity(hour=13.0, season="spring")
    wet = _activity(hour=13.0, season="spring", rate=8.0)
    assert wet["birds"] < 0.15 * dry["birds"]


def test_insects_and_frogs_follow_temperature_and_season():
    """Crickets sing on warm nights (silent below ~12 C, by day and in winter); frogs chorus on warm
    wet spring nights and are silent in winter."""
    warm_night = _activity(hour=23.0, season="summer", temperature_c=20.0)
    cold_night = _activity(hour=23.0, season="summer", temperature_c=8.0)
    winter = _activity(hour=23.0, season="winter", temperature_c=15.0)
    day = _activity(hour=12.0, season="summer", temperature_c=24.0)
    assert warm_night["insects"] > 0.8
    assert cold_night["insects"] == 0.0 and winter["insects"] == 0.0 and day["insects"] == 0.0
    wet_spring = _activity(hour=22.0, season="spring", temperature_c=15.0, rate=2.0)
    wet_winter = _activity(hour=22.0, season="winter", temperature_c=15.0, rate=2.0)
    assert wet_spring["frogs"] > 0.2 and wet_winter["frogs"] == 0.0


def _traffic(level: float, rain: float = 0.0, seconds: float = 4.0, preset: str = "meadow"):
    from raingen.synth.biophony import TrafficGenerator

    sc = calm_scene()
    sc.place.preset = preset
    sc.weather.hour = 17.0
    sc.place.biophony = dict(sc.place.biophony, traffic=level)
    if rain > 0:
        sc.weather.precipitation.type = "rain"
        sc.weather.precipitation.rate_mm_h = rain
    out, _, _ = run_env(sc, seconds, lambda s, lay, r: TrafficGenerator(s, lay, r))
    return out["traffic"][SR:]


def test_traffic_level_follows_its_control():
    """CNOSSOS-EU style road noise: L ~ 10 log10(flow); flow ~ the traffic control, so 0.2 -> 0.8 is
    +6.0 dB (distant rural road, no individual pass-bys)."""
    lo, hi = _traffic(0.2), _traffic(0.8)
    assert spl(hi) - spl(lo) == pytest.approx(10.0 * math.log10(4.0), abs=1.0)


def test_wet_roads_hiss():
    """Sandberg & Ejsmont (2002): a wet road adds ~5-15 dB of tyre spray hiss above 1 kHz."""
    dry = _traffic(0.6)
    wet = _traffic(0.6, rain=8.0)
    gain = band_db(wet, 2000.0, 8000.0) - band_db(dry, 2000.0, 8000.0)
    assert 5.0 <= gain <= 20.0, gain


# --------------------------------------------------------------------------
# places: shelters, rooms, wipers
# --------------------------------------------------------------------------

def test_mass_law_transmission_loss():
    """Field-incidence mass law (Bies & Hansen): TL ~ 20 log10(m f) - 47 dB for heavy leaves, +6 dB per
    octave, until leaks cap it (tau = (1 - s) 10^(-TL/10) + s)."""
    from raingen.synth.places import mass_law_tl_db

    m = 20.0
    tl = mass_law_tl_db(np.array([500.0, 1000.0]), m)
    assert tl[1] == pytest.approx(20.0 * math.log10(m * 1000.0) - 47.0, abs=1.5)
    assert tl[1] - tl[0] == pytest.approx(6.0, abs=0.3)
    capped = mass_law_tl_db(np.array([4000.0]), m, leak=1e-3)
    assert capped[0] == pytest.approx(-10.0 * math.log10(1e-3 + (1 - 1e-3) * 10 ** (-mass_law_tl_db(4000.0, m) / 10)),
                                      abs=1e-6)
    assert capped[0] < 30.1


def test_shelter_isolation_ranks_tent_car_room():
    """A tent (0.15 kg/m^2 fabric, permeable) barely isolates; a car body less than a closed room."""
    from raingen.synth.places import shelter_tl_db

    f = np.array([1000.0])
    tent, car, room = (float(shelter_tl_db(s, 0.0, f)[0]) for s in ("tent", "car", "room_closed"))
    assert tent < 10.0 < car < room
    assert float(shelter_tl_db("open", 0.0, f)[0]) == 0.0


def _level_difference_a(tl_fn) -> float:
    """A-weighted outdoor-indoor level difference for a pink (-3 dB/octave) exterior spectrum."""
    f = np.geomspace(50.0, 16000.0, 600)
    w = 10.0 ** (a_weight_db(f) / 10.0) / f * np.gradient(f)
    return 10.0 * math.log10(float(np.sum(w)) / float(np.sum(w * 10.0 ** (-tl_fn(f) / 10.0))))


def test_window_open_interpolates_room_isolation():
    """Opening the window adds its open area to the leak; isolation falls monotonically with the
    opening. Locher et al. (2018, IJERPH 15:149; 102 bedrooms): mean outdoor-indoor differences of
    27.8 dB closed, 15.5 dB tilted, 10.0 dB open. Tolerance +-3 dB (spectra and rooms vary)."""
    from raingen.synth.places import shelter_tl_db

    f = np.array([1000.0])
    tl = [float(shelter_tl_db("room_closed", wo, f)[0]) for wo in (0.0, 0.05, 0.25, 0.5, 1.0)]
    assert all(a > b for a, b in zip(tl, tl[1:])), tl
    closed = _level_difference_a(lambda ff: shelter_tl_db("room_closed", 0.0, ff))
    tilted = _level_difference_a(lambda ff: shelter_tl_db("room_closed", 0.3, ff))
    opened = _level_difference_a(lambda ff: shelter_tl_db("room_closed", 1.0, ff))
    assert closed == pytest.approx(27.8, abs=3.0)
    assert tilted == pytest.approx(15.5, abs=3.0)
    assert opened == pytest.approx(10.0, abs=3.0)


def _t20_rt60(hs, sr=SR, band=(500.0, 1000.0)):
    """ISO 3382 T20 (extrapolated to 60 dB) of the summed band energy of impulse responses `hs`
    (averaging outputs, as ISO 3382 averages positions: one FDN output is a single noisy realisation)."""
    e = 0.0
    for h in np.atleast_2d(hs):
        e = e + signal.sosfiltfilt(signal.butter(4, band, "bandpass", fs=sr, output="sos"), h) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-30)
    i5, i25 = int(np.argmax(edc_db <= -5.0)), int(np.argmax(edc_db <= -25.0))
    return 3.0 * (i25 - i5) / sr


@pytest.mark.parametrize("shelter", ["car", "tent", "room_closed", "attic", "underpass", "cave"])
def test_shelter_rooms_realise_their_reverberation_time(shelter):
    """The SH-domain FDN (Jot & Chaigne 1991) decays at the shelter's mid-frequency RT60 (ISO 3382
    T20 in the 500 Hz-1 kHz octaves, +-12 %); RT60 values sit in plausible ranges (car cabin
    < 0.15 s, living room 0.3-0.8 s, caves > 1 s)."""
    from raingen.synth.places import Room, room_for_shelter

    rt, wet = room_for_shelter(shelter)
    assert wet > 0.0
    room = Room(rt, SR)
    L = int((1.5 * rt + 0.3) * SR)
    hs = []
    for ch in range(4):                                         # W, Y, Z, X impulses
        room.reset()
        x = np.zeros((16, L))
        x[ch, 0] = 1.0
        hs.extend(np.concatenate([room.process(x[:, i:i + 700]) for i in range(0, L, 700)], axis=1))
    assert np.all(np.isfinite(hs))
    assert _t20_rt60(hs) == pytest.approx(rt, rel=0.12)
    assert _t20_rt60(hs, band=(5600.0, 11200.0)) < 0.9 * rt    # walls absorb more at high frequencies
    plausible = {"car": (0.03, 0.15), "tent": (0.05, 0.25), "room_closed": (0.3, 0.8), "attic": (0.4, 1.2),
                 "underpass": (0.8, 3.0), "cave": (1.0, 5.0)}[shelter]
    assert plausible[0] <= rt <= plausible[1]


def test_room_reverb_shortens_with_open_window():
    """An open window is a perfect absorber: RT60 and the reverberant send fall as it opens."""
    from raingen.synth.places import room_for_shelter

    rt0, wet0 = room_for_shelter("room_closed", 0.0)
    rt1, wet1 = room_for_shelter("room_closed", 1.0)
    assert rt1 < rt0 and wet1 < wet0


def _wipers(mode: str, seconds: float = 15.0, rain: float = 5.0):
    from raingen.synth.places import CarWipersGenerator

    sc = calm_scene()
    sc.place.shelter = "car"
    sc.place.wipers = mode
    sc.weather.precipitation.type = "rain"
    sc.weather.precipitation.rate_mm_h = rain
    out, _, _ = run_env(sc, seconds, lambda s, lay, r: CarWipersGenerator(s, lay, r))
    return out["car_wipers"]


def _reversals(y) -> int:
    """Linkage thunks at each blade reversal: peaks of the < 600 Hz envelope at least 0.3 s apart."""
    lp = signal.sosfilt(signal.butter(4, [70.0, 600.0], "bandpass", fs=SR, output="sos"), y)
    env = signal.sosfilt(signal.butter(2, 30.0, fs=SR, output="sos"), np.abs(lp))
    pk, _ = signal.find_peaks(env, height=0.4 * np.max(env), distance=int(0.3 * SR))
    return len(pk)


@pytest.mark.parametrize("mode,cpm", [("slow", 40.0), ("fast", 62.0)])
def test_wipers_follow_their_setting(mode, cpm):
    """FMVSS 104 S4.1.1: two speeds, >= 20 and >= 45 cycles/min, differing by >= 15 (here 40 and 62).
    Two reversals (top and park) per cycle: 15 s at 40 cycles/min -> 20 thunks."""
    from raingen.synth.places import WIPER_CPM

    assert WIPER_CPM["fast"] >= 45.0 and WIPER_CPM["slow"] >= 20.0
    assert WIPER_CPM["fast"] - WIPER_CPM["slow"] >= 15.0
    y = _wipers(mode)
    assert _reversals(y) == pytest.approx(2.0 * cpm * 15.0 / 60.0, abs=2)


def test_wipers_off_are_silent_and_intermittent_wipes_rarely():
    """Off: nothing; intermittent: one wipe (two reversals) every ~7.5 s."""
    assert np.max(np.abs(_wipers("off", seconds=4.0))) == 0.0
    y = _wipers("intermittent", seconds=16.0)
    assert 3 <= _reversals(y) <= 6


# --------------------------------------------------------------------------
# signal hygiene for every environment generator, any block length
# --------------------------------------------------------------------------

def busy_scene(seed: int = 77) -> Scene:
    """Everything at once: gusty fresh wind, trees, wires, chimes, a flag, life, traffic, a car."""
    sc = calm_scene(seed)
    w = sc.weather
    w.wind.speed_mps = 13.0
    w.wind.mic_rumble = True
    w.wind.windscreen = "foam"
    w.hour = 20.5
    w.temperature_c = 22.0
    w.season = "summer"
    w.precipitation.type = "rain"
    w.precipitation.rate_mm_h = 1.0
    p = sc.place
    p.preset = "suburb"
    p.shelter = "car"
    p.wipers = "fast"
    p.surfaces = {"grass": 60.0, "asphalt": 40.0}
    p.vegetation = {"deciduous": 3, "conifer": 2, "distance_m": 12.0}
    p.emitters = [{"type": "wire", "distance_m": 15.0, "bearing_deg": 90.0},
                  {"type": "chime", "distance_m": 3.0, "bearing_deg": 200.0},
                  {"type": "flag", "distance_m": 8.0, "bearing_deg": 300.0}]
    p.biophony = {"birds": "on", "insects": "on", "frogs": "on", "traffic": 0.6, "region": "north_america"}
    return sc


_HYGIENE: dict = {}


def _hygiene_run(n: int):
    if n not in _HYGIENE:
        _HYGIENE[n] = run_env(busy_scene(), 3.0, env_gens(), n=n)[0]
    return _HYGIENE[n]


@pytest.mark.parametrize("n", [1024, 700])
def test_environment_generators_signal_hygiene(n):
    """Finite, no DC, no clicks at block boundaries (jump statistics at multiples of the block length
    match the rest of the signal), no denormal-scale residue, for any engine block length."""
    out = _hygiene_run(n)
    expected = {"wind", "vegetation", "aeolian", "thunder", "biophony", "traffic", "car_wipers"}
    assert "mic_rumble" not in out                           # no wind reaches a capsule inside a car
    assert expected <= set(out), set(out)
    for name, y in out.items():
        assert np.all(np.isfinite(y)), name
        rms = math.sqrt(float(np.mean(y ** 2)))
        if rms == 0.0:
            assert name == "thunder"                         # no storm cell in this scene
            continue
        assert abs(float(np.mean(y))) < 0.1 * rms, name       # no DC (car buffeting keeps some < 8 Hz)
        assert block_jump_ratio(y, n) < 1.5, (name, block_jump_ratio(y, n))
        tiny = (np.abs(y) > 0) & (np.abs(y) < 1e-30)
        assert not np.any(tiny), name


def test_environment_levels_do_not_depend_on_block_length():
    """The same scene rendered with 1024- and 700-frame blocks has the same level per generator
    (the gust process follows the engine's control rate, so only fine detail differs): +-3 dB."""
    a, b = _hygiene_run(1024), _hygiene_run(700)
    for name in ("wind", "vegetation", "biophony", "traffic", "car_wipers"):
        assert spl(a[name][SR:]) == pytest.approx(spl(b[name][SR:]), abs=3.0), name


def test_environment_generators_are_deterministic():
    """Same seed -> bit-identical output; a different seed -> different output (MASTER_PLAN 9.4)."""
    sc1, sc2, sc3 = busy_scene(5), busy_scene(5), busy_scene(6)
    a = run_env(sc1, 1.0, env_gens())[0]
    b = run_env(sc2, 1.0, env_gens())[0]
    c = run_env(sc3, 1.0, env_gens())[0]
    for name in a:
        assert np.array_equal(a[name], b[name]), name
        if np.any(a[name]):
            assert not np.array_equal(a[name], c[name]), name


@pytest.mark.parametrize("shelter", ["car", "tent", "room_closed", "room_open", "attic", "porch", "cave"])
def test_shelter_transfer_realises_the_envelope_loss(shelter):
    """The exterior-stem filter matches -TL of the envelope model at the octave centres 63 Hz-8 kHz
    (graphic EQ of peaking sections; +-2.5 dB), so sheltered exteriors are duller but not silent."""
    from raingen.synth.places import BANDS_HZ, ShelterTransfer

    sc = calm_scene()
    sc.place.shelter = shelter
    tr = ShelterTransfer(sc, SR)
    assert tr.active
    sel = BANDS_HZ <= 8000.0
    if tr.sos is None:
        assert np.max(tr.tl_db) <= 0.05
        return
    _, h = signal.sosfreqz(tr.sos, worN=BANDS_HZ[sel], fs=SR)
    got = 20.0 * np.log10(np.abs(h))
    assert got == pytest.approx(-tr.tl_db[sel], abs=2.5)
    assert np.max(tr.tl_db) < 60.0                          # leaks keep every enclosure audible
    # duller: the loss grows with frequency (mass law) or stays flat (open structures)
    assert tr.tl_db[sel][-1] >= tr.tl_db[sel][0] - 0.5


@pytest.mark.slow
@pytest.mark.parametrize("name,wind_dba,rel", [("steady-rain-leaves-night", (25.0, 45.0), (-30.0, 0.0)),
                                               ("conifer-forest-gale", (65.0, 82.0), (10.0, 45.0))])
def test_environment_level_is_sane_relative_to_rain(name, wind_dba, rel):
    """Absolute levels from Bolin (2009): a spruce stand gave 55.2 dB(A) at 5.1 m/s heard from 29 m and
    conifers rise 23-35 log10(U), i.e. 68-75 dB(A) in a force-8 gale (18.5 m/s; up to +6 dB inside the
    stand); a light breeze (2.4 m/s) in leaves gives a soft rustle (Fegeant: ~30-40 dB(A)). Relative to
    the rain, from the preset descriptions: 'steady rain on leaves, light breeze' -> rustle below the
    rain; 'force 8 through spruce and pine' -> the gale drowns 5 mm/h of rain."""
    from raingen.render import BlockRenderer
    from raingen.scene.presets import preset

    br = BlockRenderer(preset(name), "mono", stems=True)
    acc: dict = {}
    for b in range(int(1.0 * SR / BLOCK)):
        out = br.render_block(b)
        if b * BLOCK < 0.25 * SR:
            continue
        for st, y in out.stems.items():
            acc.setdefault(st, []).append(y[0])
    sig = {st: np.concatenate(v) for st, v in acc.items()}
    wind = spl_a(sig["wind"])
    rain = spl_a(sig["rain_near"] + sig["rain_far"])
    assert wind_dba[0] <= wind <= wind_dba[1], (wind, rain)
    assert rel[0] <= wind - rain <= rel[1], (wind, rain)


@pytest.mark.slow
def test_tree_canopy_overhead_dulls_rain_and_adds_canopy_drips():
    """Under a tree the rain you hear falls on leaves overhead (intercepted, softer and duller than on
    open ground) and the canopy releases larger drip drops (MASTER_PLAN 7.9: canopies intercept
    10-40 % and re-release it as drips up to ~5-7 mm)."""
    from raingen.render import BlockRenderer

    res = {}
    for shelter in ("open", "tree"):
        sc = calm_scene(9)
        sc.weather.precipitation.type = "rain"
        sc.weather.precipitation.rate_mm_h = 6.0
        sc.weather.wind.speed_mps = 2.0
        sc.place.preset = "garden"
        sc.place.surfaces = {"grass": 50.0, "soil": 30.0, "concrete": 20.0}
        sc.place.vegetation = {"deciduous": 2, "distance_m": 12.0}
        sc.place.shelter = shelter
        br = BlockRenderer(sc, "mono", stems=True)
        acc: dict = {}
        for b in range(int(1.0 * SR / BLOCK)):
            out = br.render_block(b)
            for s, y in out.stems.items():
                acc.setdefault(s, []).append(y[0])
        res[shelter] = {s: np.concatenate(v) for s, v in acc.items()}
    rain = {k: v["rain_near"] + v["rain_far"] for k, v in res.items()}
    assert centroid(rain["tree"]) < centroid(rain["open"]), (centroid(rain["tree"]), centroid(rain["open"]))
    assert spl(rain["tree"]) > spl(rain["open"]) - 20.0                       # duller, not silent
    assert spl(res["tree"]["aftermath"]) > spl(res["open"]["aftermath"]) + 3.0  # canopy drips
