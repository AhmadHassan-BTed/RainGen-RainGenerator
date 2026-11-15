"""Human units, labels and locale-aware parsing (MASTER_PLAN 2.5, 2.10, Appendix A).

Stored units are fixed (2.10): mm/h, m/s, degC, hPa, m / km, and degrees true
*from* (meteorological convention). Everything a person types or reads goes
through this module.

Exact conversion factors: 1 kn = 0.514444 m/s (1852 m / 3600 s), 1 mph =
0.44704 m/s, 1 km/h = 1/3.6 m/s, 1 in = 25.4 mm, 1 mi = 1.609344 km,
1 NM = 1.852 km, 1 ft = 0.3048 m, 1 inHg = 33.8639 hPa.

Beaufort: WMO / Met Office scale (WMO-No. 306), continuous fit
v = 0.836 * B^1.5 m/s at 10 m. Rain-rate words: WMO (WMO-No. 8; slight < 2.5,
moderate 2.5-10, heavy 10-50, intense >= 50, cloudburst >= 100 mm/h; drizzle
0.1 / 0.5 mm/h) and the AMS Glossary (light <= 2.5, moderate 2.6-7.6, heavy
> 7.6 mm/h). Flash-to-bang converts seconds to km with the speed of sound at
the air temperature (3.08 s/km at -10 degC, 2.91 at 20 degC).
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from ..core.constants import speed_of_sound

# ---------------------------------------------------------------------------
# Exact factors
# ---------------------------------------------------------------------------
KN_TO_MPS = 0.514444
MPH_TO_MPS = 0.44704
KMH_TO_MPS = 1.0 / 3.6
FTS_TO_MPS = 0.3048
IN_TO_MM = 25.4
MI_TO_KM = 1.609344
NM_TO_KM = 1.852
FT_TO_M = 0.3048
INHG_TO_HPA = 33.8639
MMHG_TO_HPA = 1.333224

COMPASS16 = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)

_COMPASS_LONG = {
    "n": "north", "e": "east", "s": "south", "w": "west",
}

# Beaufort scale (WMO / Met Office): name, upper bound of the band in m/s at 10 m, land cue.
BEAUFORT_NAMES = (
    "calm", "light air", "light breeze", "gentle breeze", "moderate breeze",
    "fresh breeze", "strong breeze", "near gale", "gale", "strong gale",
    "storm", "violent storm", "hurricane force",
)
BEAUFORT_UPPER_MPS = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, math.inf)
BEAUFORT_LAND = (
    "smoke rises vertically",
    "smoke drifts, vanes still",
    "leaves rustle, wind felt on face",
    "leaves and small twigs in constant motion",
    "dust and loose paper raised, small branches move",
    "small trees in leaf begin to sway",
    "large branches move, wires whistle, umbrellas hard to use",
    "whole trees in motion, hard to walk against the wind",
    "twigs break off trees, walking is impeded",
    "slight structural damage, slates and chimney pots removed",
    "trees uprooted, considerable structural damage",
    "widespread damage",
    "devastation",
)

# Additional names people use (mapped onto the force number).
_BEAUFORT_ALIASES = {
    "calm": 0, "dead calm": 0, "light air": 1, "light airs": 1, "light breeze": 2,
    "gentle breeze": 3, "moderate breeze": 4, "fresh breeze": 5, "strong breeze": 6,
    "near gale": 7, "moderate gale": 7, "gale": 8, "fresh gale": 8, "strong gale": 9,
    "severe gale": 9, "storm": 10, "whole gale": 10, "storm force": 10,
    "violent storm": 11, "hurricane": 12, "hurricane force": 12,
}

# Representative rates for rain words (mm/h), WMO-centred.
RATE_WORDS = {
    "none": 0.0, "dry": 0.0, "no rain": 0.0, "trace": 0.05,
    "light drizzle": 0.08, "slight drizzle": 0.08, "drizzle": 0.3, "moderate drizzle": 0.3,
    "heavy drizzle": 0.8, "mist": 0.05,
    "very light": 0.3, "slight": 1.0, "light": 1.0, "gentle": 1.2, "soft": 1.2,
    "moderate": 5.0, "steady": 5.0, "heavy": 20.0, "pouring": 25.0, "downpour": 30.0,
    "intense": 60.0, "violent": 60.0, "torrential": 60.0, "deluge": 80.0, "cloudburst": 120.0,
}

_HAIL_SIZES_MM = (
    ("pea", 6.4), ("marble", 12.7), ("penny", 19.0), ("nickel", 22.2), ("quarter", 25.4),
    ("half dollar", 31.8), ("walnut", 38.0), ("golf ball", 44.5), ("hen egg", 50.8),
    ("tennis ball", 63.5), ("baseball", 69.9), ("teacup", 76.2), ("grapefruit", 101.6),
    ("softball", 101.6),
)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def _num(s: str) -> float:
    """Parse a number, accepting decimal commas ("12,5") but not thousands commas ("1,000")."""
    s = s.strip().replace("−", "-").replace(" ", "")
    if "/" in s:                            # simple fraction "1/2"
        a, b = s.split("/", 1)
        return _num(a) / _num(b)
    if "," in s and "." not in s:
        if re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:,\d{3})+", s):
            s = s.replace(",", "")          # thousands separator
        else:
            s = s.replace(",", ".")         # decimal comma (locale)
    else:
        s = s.replace(",", "")
    return float(s)


_NUM_UNIT = re.compile(
    r"^\s*(?P<num>[+\-−]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:\s+\d+/\d+|/\d+)?)\s*(?P<unit>.*?)\s*$"
)


def _split(s: str) -> tuple[float, str]:
    m = _NUM_UNIT.match(s)
    if not m:
        raise ValueError(f"no number in {s!r}")
    numtxt = m.group("num")
    if " " in numtxt.strip() and "/" in numtxt:     # mixed number "1 1/2"
        whole, frac = numtxt.split()
        value = _num(whole) + _num(frac)
    else:
        value = _num(numtxt)
    return value, m.group("unit").strip().lower()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("°", "° ").replace("  ", " ")).strip()


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _clean_unit(u: str) -> str:
    return u.replace(" ", "").replace("°", "").replace("per", "/").strip(".")


# ---------------------------------------------------------------------------
# Beaufort
# ---------------------------------------------------------------------------
def beaufort_to_mps(b: float) -> float:
    """Continuous Beaufort force -> mean wind at 10 m (m/s): v = 0.836 B^1.5."""
    b = max(0.0, float(b))
    return 0.836 * b ** 1.5


def mps_to_beaufort(v: float) -> float:
    """Mean wind at 10 m (m/s) -> continuous Beaufort force (inverse of 0.836 B^1.5)."""
    v = max(0.0, float(v))
    return (v / 0.836) ** (2.0 / 3.0)


def beaufort_force(v_mps: float) -> int:
    """Integer Beaufort force from the WMO band table."""
    for force, hi in enumerate(BEAUFORT_UPPER_MPS):
        if v_mps < hi:
            return force
    return 12


def beaufort_name(b: float) -> str:
    return BEAUFORT_NAMES[int(min(12, max(0, round(b))))]


def beaufort_land(b: float) -> str:
    return BEAUFORT_LAND[int(min(12, max(0, round(b))))]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_rate(s: Any) -> float:
    """Rain rate -> mm/h. Accepts numbers (mm/h), '8 mm/h', '0.3 in/h', '2 mm/min', words ('heavy')."""
    if _is_number(s):
        return float(s)
    t = _norm(s)
    t = re.sub(r"\b(rain|of rain|rainfall|precipitation)\b", "", t).strip()
    if t in RATE_WORDS:
        return RATE_WORDS[t]
    if not re.search(r"\d", t):
        for word in sorted(RATE_WORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", t):
                return RATE_WORDS[word]
        raise ValueError(f"cannot parse rain rate {s!r}")
    v, unit = _split(t)
    u = _clean_unit(unit)
    if u in ("", "mm/h", "mm/hr", "mm/hour", "mmh", "mmph", "mmh-1", "mm.h-1", "mm/h.", "mmperhour", "mmanhour", "mm/an hour", "mm"):
        return v
    if u in ("in/h", "in/hr", "in/hour", "inh", "iph", "inches/hour", "inch/hour", "inchesanhour", '"/h', "in"):
        return v * IN_TO_MM
    if u in ("mm/min", "mmpm", "mm/minute"):
        return v * 60.0
    if u in ("mm/day", "mm/d", "mm/24h", "mmpd"):
        return v / 24.0
    if u in ("cm/h", "cm/hr", "cmh"):
        return v * 10.0
    if u in ("mm/10min", "mmin10min", "mm/10min."):
        return v * 6.0
    raise ValueError(f"unknown rain-rate unit {unit!r} in {s!r}")


def parse_speed(s: Any) -> float:
    """Wind speed -> m/s. Accepts 'force 5', 'Bft 5', 'F5', '30 km/h', '15 mph', '20 kt', '8 m/s', 'gale'."""
    if _is_number(s):
        return float(s)
    t = _norm(s)
    m = re.match(r"^(?:force|bft|beaufort|f)\s*(\d+(?:[.,]\d+)?)$", t)
    if m:
        return beaufort_to_mps(_num(m.group(1)))
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(?:bft|beaufort|on the beaufort scale)$", t)
    if m:
        return beaufort_to_mps(_num(m.group(1)))
    if not re.search(r"\d", t):
        for name in sorted(_BEAUFORT_ALIASES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", t):
                return beaufort_to_mps(_BEAUFORT_ALIASES[name])
        raise ValueError(f"cannot parse wind speed {s!r}")
    v, unit = _split(t)
    u = _clean_unit(unit)
    if u in ("", "m/s", "mps", "ms", "ms-1", "m.s-1", "metres/second", "meters/second", "m/sec"):
        return v
    if u in ("km/h", "kmh", "kph", "kmph", "km/hr", "kilometres/hour", "kilometers/hour", "km/hour"):
        return v * KMH_TO_MPS
    if u in ("mph", "mi/h", "mi/hr", "miles/hour", "milesanhour"):
        return v * MPH_TO_MPS
    if u in ("kt", "kts", "kn", "knot", "knots", "kt."):
        return v * KN_TO_MPS
    if u in ("ft/s", "fps", "ft/sec"):
        return v * FTS_TO_MPS
    raise ValueError(f"unknown speed unit {unit!r} in {s!r}")


def flash_to_bang_s(distance_km: float, temp_c: float = 20.0) -> float:
    """Seconds between flash and thunder at `distance_km` (speed of sound at temp_c)."""
    return float(distance_km) * 1000.0 / speed_of_sound(temp_c)


def seconds_to_km(seconds: float, temp_c: float = 20.0) -> float:
    """Flash-to-bang seconds -> distance in km at air temperature temp_c."""
    return float(seconds) * speed_of_sound(temp_c) / 1000.0


_DISTANCE_WORDS = {
    "overhead": 0.5, "right overhead": 0.4, "very close": 1.0, "close": 2.0, "near": 2.5,
    "nearby": 3.0, "distant": 12.0, "far": 15.0, "far away": 15.0, "far off": 15.0,
    "very distant": 20.0, "faint": 18.0,
}


def parse_distance_km(s: Any, temp_c: float = 20.0) -> float:
    """Distance -> km. Accepts '3 km', '800 m', '2 mi', '5 NM', '9 s' (flash-to-bang at temp_c), words."""
    if _is_number(s):
        return float(s)
    t = _norm(s)
    if t in _DISTANCE_WORDS:
        return _DISTANCE_WORDS[t]
    v, unit = _split(t)
    u = _clean_unit(unit).rstrip(".")
    if u in ("", "km", "kms", "kilometre", "kilometres", "kilometer", "kilometers", "k"):
        return v
    if u in ("m", "metre", "metres", "meter", "meters"):
        return v / 1000.0
    if u in ("mi", "mile", "miles", "sm", "statutemiles"):
        return v * MI_TO_KM
    if u in ("nm", "nmi", "nauticalmile", "nauticalmiles"):
        return v * NM_TO_KM
    if u in ("ft", "feet", "foot"):
        return v * FT_TO_M / 1000.0
    if u in ("yd", "yard", "yards"):
        return v * 0.9144 / 1000.0
    if u in ("s", "sec", "secs", "second", "seconds", "sflash-to-bang", "secondsflash-to-bang"):
        return seconds_to_km(v, temp_c)
    raise ValueError(f"unknown distance unit {unit!r} in {s!r}")


def parse_temperature_c(s: Any) -> float:
    """Temperature -> degC. Accepts '12 C', '12°C', '54 F', '285 K', 'M05' (METAR minus)."""
    if _is_number(s):
        return float(s)
    t = _norm(s)
    m = re.match(r"^m(\d+)$", t)          # METAR style
    if m:
        return -float(m.group(1))
    v, unit = _split(t)
    u = _clean_unit(unit)
    if u in ("", "c", "degc", "celsius", "degreescelsius", "deg", "degrees", "centigrade"):
        return v
    if u in ("f", "degf", "fahrenheit", "degreesfahrenheit"):
        return (v - 32.0) * 5.0 / 9.0
    if u in ("k", "kelvin"):
        return v - 273.15
    raise ValueError(f"unknown temperature unit {unit!r} in {s!r}")


def _compass_lookup() -> dict[str, float]:
    table: dict[str, float] = {}
    for i, code in enumerate(COMPASS16):
        deg = i * 22.5
        table[code.lower()] = deg
        long_parts = [_COMPASS_LONG[c] for c in code.lower()]
        if len(code) == 1:
            names = [long_parts[0]]
        elif len(code) == 2:
            names = ["".join(long_parts), "-".join(long_parts), " ".join(long_parts)]
        else:
            first = long_parts[0]
            rest = long_parts[1:]
            names = [
                f"{first}-{''.join(rest)}", f"{first} {''.join(rest)}",
                f"{first}-{'-'.join(rest)}", f"{first}{''.join(rest)}",
                f"{first} by {''.join(rest)}",
            ]
        for name in names:
            table[name] = deg
    return table


_COMPASS_TABLE = _compass_lookup()


def parse_direction_deg(s: Any) -> float:
    """Direction -> degrees true (0..360). Accepts 16-point compass ('WSW'), words ('north-west',
    'westerly', 'from the west') or degrees ('270', '270°', '270 deg')."""
    if _is_number(s):
        return float(s) % 360.0
    t = _norm(s)
    t = re.sub(r"^(?:wind\s+)?(?:coming\s+)?(?:from\s+)?(?:the\s+)?", "", t)
    t = re.sub(r"\s+(?:winds?|wind direction|true|t)$", "", t).strip()
    m = re.match(r"^(-?\d+(?:[.,]\d+)?)\s*(?:°|deg|degs|degree|degrees)?\s*(?:true|t)?$", t)
    if m:
        return _num(m.group(1)) % 360.0
    key = t.replace("_", " ").strip()
    if key in _COMPASS_TABLE:
        return _COMPASS_TABLE[key]
    if key.endswith("erly"):                 # westerly, northeasterly, south-westerly
        base = key[:-4]
        if base in _COMPASS_TABLE:
            return _COMPASS_TABLE[base]
    if key.endswith("ward") or key.endswith("wards"):
        raise ValueError(f"{s!r} names a 'to' direction; give where the wind comes FROM")
    key2 = key.replace(" ", "").replace("-", "")
    for name, deg in _COMPASS_TABLE.items():
        if name.replace(" ", "").replace("-", "") == key2:
            return deg
    raise ValueError(f"cannot parse direction {s!r}")


def compass_label(deg: float, points: int = 16) -> str:
    """Degrees true -> nearest compass point (16 or 8 points)."""
    deg = float(deg) % 360.0
    if points == 8:
        labels = COMPASS16[::2]
        return labels[int((deg + 22.5) // 45.0) % 8]
    return COMPASS16[int((deg + 11.25) // 22.5) % 16]


def relative_direction(from_deg: float, heading_deg: float) -> str:
    """Where a 'from' direction lies relative to the listener: front, front-left, left, ..."""
    rel = (float(from_deg) - float(heading_deg)) % 360.0
    names = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")
    return names[int((rel + 22.5) // 45.0) % 8]


def parse_pressure_hpa(s: Any) -> float:
    if _is_number(s):
        return float(s)
    v, unit = _split(_norm(s))
    u = _clean_unit(unit)
    if u in ("", "hpa", "mb", "mbar", "millibar", "millibars"):
        return v
    if u in ("inhg", "in", "inches"):
        return v * INHG_TO_HPA
    if u in ("mmhg", "torr"):
        return v * MMHG_TO_HPA
    if u in ("kpa",):
        return v * 10.0
    if u in ("pa",):
        return v / 100.0
    raise ValueError(f"unknown pressure unit {unit!r} in {s!r}")


def parse_length_m(s: Any) -> float:
    if _is_number(s):
        return float(s)
    v, unit = _split(_norm(s))
    u = _clean_unit(unit)
    if u in ("", "m", "metre", "metres", "meter", "meters"):
        return v
    if u in ("cm",):
        return v / 100.0
    if u in ("mm",):
        return v / 1000.0
    if u in ("km",):
        return v * 1000.0
    if u in ("ft", "feet", "foot", "'"):
        return v * FT_TO_M
    if u in ("in", "inch", "inches", '"'):
        return v * 0.0254
    raise ValueError(f"unknown length unit {unit!r} in {s!r}")


def parse_duration_s(s: Any) -> float:
    """'90', '90 s', '5 min', '1.5 h', '05:00' (mm:ss), '01:02:03' (h:m:s), '01:00:00:12' (TC @24 fps)."""
    if _is_number(s):
        return float(s)
    t = _norm(s)
    if re.match(r"^-?\d+(?::\d+(?:\.\d+)?){1,3}$", t):
        parts = t.split(":")
        if len(parts) == 4:
            h, m, sec, fr = parts
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(fr) / 24.0
        val = 0.0
        for p in parts:
            val = val * 60.0 + float(p)
        return val
    v, unit = _split(t)
    u = _clean_unit(unit)
    if u in ("", "s", "sec", "secs", "second", "seconds"):
        return v
    if u in ("ms",):
        return v / 1000.0
    if u in ("m", "min", "mins", "minute", "minutes"):
        return v * 60.0
    if u in ("h", "hr", "hrs", "hour", "hours"):
        return v * 3600.0
    raise ValueError(f"unknown duration unit {unit!r} in {s!r}")


def parse_hail_size_mm(s: Any) -> float:
    if _is_number(s):
        return float(s)
    t = _norm(s)
    for name, mm in sorted(_HAIL_SIZES_MM, key=lambda kv: -len(kv[0])):
        if name in t:
            return mm
    v, unit = _split(t)
    u = _clean_unit(unit)
    if u in ("", "mm"):
        return v
    if u in ("cm",):
        return v * 10.0
    if u in ("in", "inch", "inches", '"'):
        return v * IN_TO_MM
    raise ValueError(f"unknown hail size {s!r}")


def parse_bool(s: Any) -> bool:
    if isinstance(s, bool):
        return s
    if _is_number(s):
        return bool(s)
    t = _norm(s)
    if t in ("1", "true", "yes", "on", "enabled", "enable", "y", "t"):
        return True
    if t in ("0", "false", "no", "off", "disabled", "disable", "n", "f", "none"):
        return False
    raise ValueError(f"cannot parse boolean {s!r}")


def parse_value(path: str, value: Any, temp_c: float = 20.0) -> Any:
    """Parse a human value for the stored unit implied by a scene path's suffix.

    Non-string values pass through unchanged. Unknown suffixes fall back to float().
    """
    if not isinstance(value, str):
        return value
    leaf = path.rsplit(".", 1)[-1]
    if re.fullmatch(r"\s*[+\-−]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*", value):
        return _num(value)                     # bare numbers are already in stored units
    if leaf.endswith("rate_mm_h"):
        return parse_rate(value)
    if leaf.endswith("_mps"):
        return parse_speed(value)
    if leaf.endswith("_kmh"):
        return parse_speed(value) * 3.6
    if leaf.endswith("_km"):
        return parse_distance_km(value, temp_c)
    if leaf.endswith("temperature_c") or leaf.endswith("_c"):
        return parse_temperature_c(value)
    if leaf.endswith("_deg") and leaf not in ("latitude_deg", "rotation_deg"):
        return parse_direction_deg(value)
    if leaf.endswith("_hpa"):
        return parse_pressure_hpa(value)
    if leaf == "hail_size_mm":
        return parse_hail_size_mm(value)
    if leaf.endswith("_m"):
        return parse_length_m(value)
    if leaf.endswith("_per_min"):
        v, unit = _split(_norm(value))
        return v * 60.0 if _clean_unit(unit) in ("/s", "/sec", "/second") else v
    if leaf.endswith("_min"):
        return parse_duration_s(value) / 60.0 if re.search(r"[a-z:]", value.lower()) else _num(value)
    if leaf.endswith("_s"):
        return parse_duration_s(value)
    if leaf == "hour":
        return parse_hour(value)
    if leaf in ("enabled", "stems", "mic_rumble"):
        return parse_bool(value)
    try:
        return _num(value)
    except ValueError:
        return value


def parse_hour(s: Any) -> float:
    """'22:30', '10 pm', 'night', 'dawn' -> hour of day 0..24."""
    if _is_number(s):
        return float(s) % 24.0
    t = _norm(s)
    words = {"midnight": 0.0, "dawn": 6.0, "sunrise": 6.5, "morning": 8.5, "noon": 12.0,
             "midday": 12.0, "afternoon": 15.0, "evening": 19.0, "dusk": 20.0,
             "sunset": 19.5, "night": 23.0, "day": 13.0}
    if t in words:
        return words[t]
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$", t)
    if not m:
        raise ValueError(f"cannot parse time of day {s!r}")
    h = int(m.group(1)) % 24
    mins = int(m.group(2) or 0)
    ampm = (m.group(3) or "").replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return h + mins / 60.0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def rain_label(mm_h: float, scheme: str = "wmo", kind: str = "rain") -> str:
    """Intensity word for a rate. scheme 'wmo' (slight/moderate/heavy/intense/cloudburst) or 'ams'
    (light/moderate/heavy). kind='drizzle' uses drizzle thresholds (WMO 0.1 / 0.5 mm/h)."""
    r = float(mm_h)
    scheme = scheme.lower()
    if r < 0.02:
        return "none"
    if kind == "drizzle":
        if scheme == "ams":
            return "light drizzle" if r < 0.3 else ("moderate drizzle" if r <= 0.5 else "heavy drizzle")
        return "slight drizzle" if r < 0.1 else ("moderate drizzle" if r < 0.5 else "heavy drizzle")
    if scheme == "ams":
        if r <= 2.5:
            return "light"
        if r <= 7.6:
            return "moderate"
        return "heavy"
    if r < 2.5:
        return "slight"
    if r < 10.0:
        return "moderate"
    if r < 50.0:
        return "heavy"
    if r < 100.0:
        return "intense"
    return "cloudburst"


def hail_label(size_mm: float) -> str:
    """NOAA object comparison for a hailstone diameter."""
    best = _HAIL_SIZES_MM[0][0]
    for name, mm in _HAIL_SIZES_MM:
        if size_mm >= mm * 0.9:
            best = name
    return best


def time_of_day_label(hour: float) -> str:
    h = float(hour) % 24.0
    if h < 5.0:
        return "night"
    if h < 7.0:
        return "dawn"
    if h < 12.0:
        return "morning"
    if h < 17.0:
        return "afternoon"
    if h < 19.5:
        return "evening"
    if h < 21.0:
        return "dusk"
    return "night"


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Magnus-Tetens dew point (Alduchov & Eskridge 1996 coefficients)."""
    a, b = 17.625, 243.04
    rh = max(1e-3, min(100.0, float(rh_pct))) / 100.0
    g = math.log(rh) + a * temp_c / (b + temp_c)
    return b * g / (a - g)


def humidity_from_dew_point(temp_c: float, dew_c: float) -> float:
    """Relative humidity (%) from temperature and dew point (Magnus-Tetens)."""
    a, b = 17.625, 243.04
    return float(max(0.0, min(100.0, 100.0 * math.exp(a * dew_c / (b + dew_c) - a * temp_c / (b + temp_c)))))


def _sig(x: float, digits: int = 2) -> str:
    if x == 0 or not math.isfinite(x):
        return f"{x:g}"
    mag = math.floor(math.log10(abs(x)))
    decimals = max(0, digits - 1 - mag)
    return f"{x:.{decimals}f}"


def format_value(path: str, value: Any, system: str = "metric", temp_c: float = 20.0,
                 heading_deg: Optional[float] = None) -> str:
    """Human readout for a stored value, e.g. 'force 5, fresh breeze, 9.4 m/s (34 km/h)'.

    system: 'metric' (SI + km/h), 'uk' (degC, mph for wind, miles for distance), 'us'
    (degF, mph, in/h, inHg, AMS rain words). The number is always shown (2.5).
    """
    system = system.lower()
    leaf = path.rsplit(".", 1)[-1]
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return value.replace("_", " ")
    if isinstance(value, (tuple, list)):
        return ", ".join(_sig(float(v), 3) for v in value)
    v = float(value)

    if leaf.endswith("rate_mm_h"):
        kind = "drizzle" if v < 0.5 and "drizzle" in path else "rain"
        label = rain_label(v, "ams" if system == "us" else "wmo", kind)
        if system == "us":
            return f"{_sig(v / IN_TO_MM, 2)} in/h ({_sig(v, 2)} mm/h), {label}"
        return f"{_sig(v, 2)} mm/h, {label}"
    if leaf.endswith("_mps"):
        b = mps_to_beaufort(v)
        force = beaufort_force(v)
        name = BEAUFORT_NAMES[force]
        if system == "us":
            return f"{v / MPH_TO_MPS:.0f} mph, {name} (force {force})"
        if system == "uk":
            return f"force {force}, {name}, {v / MPH_TO_MPS:.0f} mph"
        return f"force {force}, {name}, {_sig(v, 2)} m/s ({v / KMH_TO_MPS:.0f} km/h)"
    if leaf.endswith("_kmh"):
        if system in ("us", "uk"):
            return f"{v / 1.609344:+.0f} mph" + (" (approaching)" if v > 0 else " (receding)" if v < 0 else "")
        return f"{v:+.0f} km/h" + (" (approaching)" if v > 0 else " (receding)" if v < 0 else "")
    if leaf.endswith("distance_km") or leaf == "visibility_km":
        fb = flash_to_bang_s(v, temp_c)
        dist = f"{_sig(v / MI_TO_KM, 2)} mi" if system in ("us", "uk") else f"{_sig(v, 2)} km"
        if "thunder" in path or leaf == "distance_km":
            return f"{dist} (about {fb:.0f} s flash-to-bang)"
        return dist
    if leaf.endswith("temperature_c") or leaf.endswith("_c"):
        if system == "us":
            return f"{v * 9.0 / 5.0 + 32.0:.0f} °F"
        txt = f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"
        return f"{txt} °C"
    if leaf.endswith("_deg") and leaf not in ("latitude_deg", "rotation_deg"):
        label = compass_label(v)
        prefix = "from " if leaf.startswith("from") else ""
        rel = f", {relative_direction(v, heading_deg)}" if heading_deg is not None and leaf.startswith("from") else ""
        return f"{prefix}{label} ({v % 360:.0f}°){rel}"
    if leaf.endswith("_pct"):
        return f"{v:.0f} %"
    if leaf.endswith("_hpa"):
        if system == "us":
            return f"{v / INHG_TO_HPA:.2f} inHg"
        return f"{v:.0f} hPa"
    if leaf == "hail_size_mm":
        if system == "us":
            return f"{v / IN_TO_MM:.2f} in ({hail_label(v)})"
        return f"{v:.0f} mm ({hail_label(v)})"
    if leaf == "hour":
        h = int(v) % 24
        m = int(round((v - int(v)) * 60)) % 60
        return f"{h:02d}:{m:02d} ({time_of_day_label(v)})"
    if leaf.endswith("_mm"):
        return f"{_sig(v, 2)} mm"
    if leaf.endswith("_m"):
        if system == "us":
            return f"{_sig(v / FT_TO_M, 2)} ft"
        return f"{_sig(v, 2)} m"
    if leaf.endswith("_min"):
        return f"{_sig(v, 2)} min"
    if leaf.endswith("_s"):
        return f"{_sig(v, 2)} s"
    if leaf.endswith("_db"):
        return f"{v:+.1f} dB"
    if leaf == "cloud_okta":
        return f"{v:.0f}/8 oktas"
    if leaf.endswith("_per_min"):
        return f"{_sig(v, 2)} /min"
    return _sig(v, 3)
