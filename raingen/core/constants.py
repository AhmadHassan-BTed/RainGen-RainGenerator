"""Engine-wide constants and physical defaults."""

SAMPLE_RATE = 48_000          # internal processing rate (film/broadcast/Atmos/Opus native)
BLOCK = 1024                  # internal block size in frames
CONTROL_RATE = SAMPLE_RATE / BLOCK  # ~46.9 Hz control/physics update rate

AMBI_ORDER = 3                # internal spherical-harmonic bus order (AmbiX: ACN / SN3D)
N_AMBI = (AMBI_ORDER + 1) ** 2  # 16 channels

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
