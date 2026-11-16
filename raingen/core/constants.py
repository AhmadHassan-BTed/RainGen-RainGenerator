"""Engine-wide constants and physical defaults."""

import math

SAMPLE_RATE = 48_000          # internal processing rate (film/broadcast/Atmos/Opus native)
BLOCK = 1024                  # internal block size in frames
CONTROL_RATE = SAMPLE_RATE / BLOCK  # ~46.9 Hz control/physics update rate

AMBI_ORDER = 3                # internal spherical-harmonic bus order (AmbiX: ACN / SN3D)
N_AMBI = (AMBI_ORDER + 1) ** 2  # 16 channels

# Level calibration (the one place physical pressure meets digital full scale).
# Every engine signal is sound pressure in pascals at the listener: digital 1.0 = 1 Pa, and an
# rms of 1 Pa is 94 dB SPL re 20 uPa (the standard 94 dB acoustic calibrator level). So a
# full-scale sine (peak 1.0, rms 1/sqrt 2) is 91 dB SPL = 0 dBFS (AES17), 94 dB SPL rms reads
# +3 dBFS, and by BS.1770 (L = -0.691 + 10 log10 sum G_i z_i, K-weighting ~0 dB for broadband
# sound) a sound of L dB SPL at one ear reads about L - 94.7 LUFS in one channel and L - 91.7 LUFS
# when both channels of a stereo pair carry it. Leaving ~90 dB of headroom above a 0 dB SPL floor
# keeps quiet beds (30-50 dB SPL) far above 24-bit dither (-141 dBFS) while hail on a car roof
# (~90 dB SPL peaks) stays below clipping. Generators state their levels in dB SPL or Pa and
# convert with db_spl_to_pa; the master never rescales a raw render.
P_REF = 20e-6                 # Pa, reference pressure of dB SPL
PA_PER_FULL_SCALE = 1.0       # Pa represented by digital 1.0
FULL_SCALE_DB_SPL = 20.0 * math.log10(PA_PER_FULL_SCALE / P_REF)   # 93.98 dB SPL: an rms of 1.0


def db_spl_to_pa(db_spl: float) -> float:
    """dB SPL (re 20 uPa) -> pascals (= digital units, see PA_PER_FULL_SCALE)."""
    return P_REF * 10.0 ** (float(db_spl) / 20.0)


def pa_to_db_spl(pa: float) -> float:
    """Pascals (rms or peak) -> dB SPL re 20 uPa; -inf for silence."""
    pa = abs(float(pa))
    return 20.0 * math.log10(pa / P_REF) if pa > 0.0 else float("-inf")


# Air / water
RHO_WATER = 998.0             # kg/m^3
RHO_AIR = 1.204               # kg/m^3 at 20 C
P0 = 101_325.0                # Pa
GAMMA_AIR = 1.4
SIGMA_WATER = 0.0728          # N/m surface tension
G = 9.81                      # m/s^2


def speed_of_sound(temperature_c: float) -> float:
    """Speed of sound in air (m/s) at the given temperature."""
    return 331.3 * (1.0 + temperature_c / 273.15) ** 0.5


# Standard stems every renderer understands
STEMS = (
    "rain_near",      # impacts on near / resonant surfaces, drips
    "rain_far",       # mid/far field rain texture and diffuse bed
    "wind",           # wind, vegetation, aeolian tones
    "thunder",
    "aftermath",      # drips, runoff, gutters after/while raining
    "ambience",       # biophony, traffic, distant hum
    "interior",       # in-shelter resonances (car body, tent fabric, roof drumming)
)
