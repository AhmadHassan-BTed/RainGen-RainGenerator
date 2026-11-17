"""METAR parsing and METAR -> Scene mapping (MASTER_PLAN 2.8, Appendix A).

Follows WMO-No. 306 FM 15 and the US FMH-1 / ASOS conventions:

* wind ``dddffGggKT`` / ``MPS`` / ``KMH``, ``VRB``, variable sector ``dddVddd``;
* visibility in metres (``9999``, ``0800``), statute miles (``1SM``, ``1 1/2SM``,
  ``M1/4SM``, ``P6SM``) or ``CAVOK``; RVR groups are recognised and skipped;
* present weather: intensity ``-`` / ``+`` / ``VC``, descriptor (MI PR BC DR BL
  SH TS FZ), precipitation (DZ RA SN SG IC PL GR GS UP), obscuration (BR FG FU
  VA DU SA HZ PY) and other (PO SQ FC SS DS);
* sky ``FEW/SCT/BKN/OVC/VV`` with CB/TCU, temperature/dew point (``M`` = minus),
  altimeter ``A2992`` (inHg) or ``Q1013`` (hPa), recent weather ``RE..``;
* remarks: ``LTG`` with frequency (OCNL/FRQ/CONS), type (IC CC CG CA), ``DSNT``,
  ``VC``, ``OHD`` and direction; ``TS ... MOV E``; ``PK WND``.

Thunder distance follows ASOS: TS within 5 NM, VCTS 5-10 NM, LTG DSNT 10-30 NM.
A squall (SQ) is a sudden >= 16 kt increase sustained >= 22 kt for >= 1 min
(FMH-1), mapped to a high gust factor.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from . import units
from .schema import Scene

DESCRIPTORS = ("MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ")
PRECIP = ("DZ", "RA", "SN", "SG", "IC", "PL", "GR", "GS", "UP")
OBSCURATION = ("BR", "FG", "FU", "VA", "DU", "SA", "HZ", "PY")
OTHER = ("PO", "SQ", "FC", "SS", "DS")
_PHENOMENA = PRECIP + OBSCURATION + OTHER

_WX_RE = re.compile(
    r"^(?P<int>[-+]|VC)?(?P<desc>MI|PR|BC|DR|BL|SH|TS|FZ)?(?P<ph>(?:" + "|".join(_PHENOMENA) + r")*)$"
)
_WIND_RE = re.compile(r"^(?P<dir>\d{3}|VRB|///)(?P<spd>P?\d{2,3}|//)(?:G(?P<gust>P?\d{2,3}))?(?P<unit>KT|MPS|KMH)$")
_VARDIR_RE = re.compile(r"^(?P<a>\d{3})V(?P<b>\d{3})$")
_VIS_M_RE = re.compile(r"^(?P<m>\d{4})(?P<ndv>NDV)?(?:[NSEW]{1,2})?$")
_VIS_SM_RE = re.compile(r"^(?P<pm>[MP])?(?P<v>\d+(?:/\d+)?)SM$")
_RVR_RE = re.compile(r"^R\d{2}[LCR]?/[PM]?\d{4}(?:V[PM]?\d{4})?(?:FT)?[UDN]?$")
_SKY_RE = re.compile(r"^(?P<cov>FEW|SCT|BKN|OVC|VV|SKC|CLR|NSC|NCD)(?P<h>\d{3}|///)?(?P<cb>CB|TCU|///)?$")
_TEMP_RE = re.compile(r"^(?P<t>M?\d{2}|//)/(?P<d>M?\d{2}|//)?$")
_ALT_RE = re.compile(r"^(?P<k>[AQ])(?P<v>\d{4})$")
_TIME_RE = re.compile(r"^(?P<d>\d{2})(?P<h>\d{2})(?P<m>\d{2})Z$")
_STATION_RE = re.compile(r"^[A-Z][A-Z0-9]{3}$")

_TREND = ("NOSIG", "BECMG", "TEMPO", "FM", "TL", "AT", "PROB30", "PROB40")
_DIRS8 = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
_OKTA = {"SKC": 0.0, "CLR": 0.0, "NSC": 0.0, "NCD": 0.0, "FEW": 1.5, "SCT": 3.5, "BKN": 6.0, "OVC": 8.0, "VV": 8.0}

# Representative rates (mm/h, water equivalent) by phenomenon and intensity (light, moderate, heavy),
# inside the WMO bands: rain slight < 2.5 / moderate 2.5-10 / heavy 10-50; drizzle < 0.1 / 0.1-0.5 / >= 0.5.
_RATES = {
    "DZ": (0.05, 0.3, 0.8), "RA": (1.0, 5.0, 16.0), "SN": (0.5, 1.5, 4.0), "SG": (0.1, 0.3, 0.6),
    "PL": (0.5, 1.5, 4.0), "GR": (8.0, 20.0, 40.0), "GS": (1.0, 3.0, 8.0), "IC": (0.02, 0.05, 0.1),
    "UP": (1.0, 4.0, 10.0),
}
NM_KM = units.NM_TO_KM


def parse_weather_group(tok: str) -> Optional[dict]:
    """Parse one present-weather group ('+TSRA', '-FZDZ', 'VCSH', 'SQ'); None if it is not one."""
    tok = tok.strip().upper()
    if tok.startswith("RE") and len(tok) > 3:
        g = parse_weather_group(tok[2:])
        if g is not None:
            g["recent"] = True
        return g
    m = _WX_RE.match(tok)
    if not m or (not m.group("desc") and not m.group("ph")):
        return None
    ph = m.group("ph") or ""
    phenomena = [ph[i:i + 2] for i in range(0, len(ph), 2)]
    inten = m.group("int") or ""
    return {
        "raw": tok,
        "intensity": {"-": "light", "+": "heavy", "VC": "vicinity", "": "moderate"}[inten],
        "descriptor": m.group("desc") or "",
        "phenomena": phenomena,
        "recent": False,
    }


def _speed(v: str) -> float:
    return float(v.lstrip("P"))


def _parse_remarks(rmk: list[str]) -> dict:
    out: dict[str, Any] = {"raw": " ".join(rmk), "lightning": None, "thunderstorm": None, "peak_wind": None}
    text = " ".join(rmk)
    m = re.search(r"\b(?:(OCNL|FRQ|CONS)\s+)?LTG(IC|CC|CG|CA)*\s*(DSNT|VC|OHD|AP)?\s*"
                  r"((?:ALQDS|[NSEW]{1,2})(?=[\s-]|$)(?:[-\s](?:AND\s+)?[NSEW]{1,2}(?=[\s-]|$))*)?", text)
    if m:
        freq = m.group(1) or ""
        types = re.findall(r"IC|CC|CG|CA", m.group(0).split()[1] if m.group(1) else m.group(0).split()[0])
        dirs = re.findall(r"\b(ALQDS|[NSEW]{1,2})\b", m.group(4) or "")
        out["lightning"] = {"frequency": {"OCNL": "occasional", "FRQ": "frequent", "CONS": "continuous"}.get(freq, ""),
                            "types": types, "where": m.group(3) or "", "directions": dirs}
    # each optional group carries its own leading whitespace, so a skipped group never swallows
    # the space that 'MOV' needs ("TS OHD MOV E")
    m = re.search(r"\b(VC)?TS\b(?:\s+(OHD|DSNT|VC))?(?:\s+((?:ALQDS|[NSEW]{1,2})(?=[\s-]|$)(?:-[NSEW]{1,2}(?=[\s-]|$))*))?"
                  r"(?:\s+MOV\s+([NSEW]{1,2}|STNRY))?", text)
    if m and "TSNO" not in text:
        dirs = re.findall(r"ALQDS|[NSEW]{1,2}", m.group(3) or "")
        out["thunderstorm"] = {"vicinity": bool(m.group(1)), "where": m.group(2) or "", "directions": dirs,
                               "moving": m.group(4) or ""}
    m = re.search(r"\bPK WND (\d{3})(\d{2,3})/(\d{2,4})", text)
    if m:
        out["peak_wind"] = {"dir_deg": float(m.group(1)), "speed_kt": float(m.group(2)), "time": m.group(3)}
    return out


def parse_metar(text: str) -> dict:
    """Parse a METAR/SPECI string into a plain dict (units: kt and m/s, m, degC, hPa)."""
    raw = " ".join(text.strip().split())
    toks = raw.upper().split()
    out: dict[str, Any] = {
        "raw": raw, "type": "METAR", "station": None, "time": None, "auto": False, "corrected": False,
        "wind": None, "visibility_m": None, "cavok": False, "weather": [], "recent_weather": [], "sky": [],
        "temp_c": None, "dew_c": None, "altimeter_hpa": None, "remarks": {"raw": "", "lightning": None,
                                                                          "thunderstorm": None, "peak_wind": None},
    }
    i = 0
    if i < len(toks) and toks[i] in ("METAR", "SPECI"):
        out["type"] = toks[i]
        i += 1
    if i < len(toks) and toks[i] == "COR":
        out["corrected"] = True
        i += 1
    if i < len(toks) and _STATION_RE.match(toks[i]) and not _TIME_RE.match(toks[i]):
        out["station"] = toks[i]
        i += 1
    rmk: list[str] = []
    while i < len(toks):
        tok = toks[i]
        i += 1
        if tok == "RMK":
            rmk = toks[i:]
            break
        if tok in _TREND or tok.startswith(("FM", "TL")) and tok[2:].isdigit():
            # trend section: skip to remarks
            while i < len(toks) and toks[i] != "RMK":
                i += 1
            continue
        m = _TIME_RE.match(tok)
        if m and out["time"] is None:
            out["time"] = {"day": int(m.group("d")), "hour": int(m.group("h")), "minute": int(m.group("m"))}
            continue
        if tok in ("AUTO",):
            out["auto"] = True
            continue
        if tok == "COR":
            out["corrected"] = True
            continue
        m = _WIND_RE.match(tok)
        if m:
            unit = m.group("unit")
            spd = 0.0 if m.group("spd") == "//" else _speed(m.group("spd"))
            gust = _speed(m.group("gust")) if m.group("gust") else None
            factor = {"KT": units.KN_TO_MPS, "MPS": 1.0, "KMH": units.KMH_TO_MPS}[unit]
            to_kt = {"KT": 1.0, "MPS": 1.0 / units.KN_TO_MPS, "KMH": units.KMH_TO_MPS / units.KN_TO_MPS}[unit]
            out["wind"] = {
                "dir_deg": None if m.group("dir") in ("VRB", "///") else float(m.group("dir")),
                "variable": m.group("dir") == "VRB",
                "speed_kt": spd * to_kt, "gust_kt": gust * to_kt if gust is not None else None,
                "speed_mps": spd * factor, "gust_mps": gust * factor if gust is not None else None,
                "var_from": None, "var_to": None, "calm": spd == 0.0 and m.group("dir") == "000",
            }
            continue
        m = _VARDIR_RE.match(tok)
        if m and out["wind"] is not None:
            out["wind"]["var_from"] = float(m.group("a"))
            out["wind"]["var_to"] = float(m.group("b"))
            continue
        if tok == "CAVOK":
            out["cavok"] = True
            out["visibility_m"] = 10000.0
            continue
        # visibility in statute miles, possibly as "1 1/2SM" split over two tokens
        if re.match(r"^\d+$", tok) and i < len(toks) and re.match(r"^\d/\d+SM$", toks[i]) and len(tok) <= 2:
            whole = float(tok)
            frac = toks[i][:-2]
            i += 1
            out["visibility_m"] = (whole + units._num(frac)) * units.MI_TO_KM * 1000.0
            continue
        m = _VIS_SM_RE.match(tok)
        if m:
            v = units._num(m.group("v"))
            out["visibility_m"] = v * units.MI_TO_KM * 1000.0
            continue
        m = _VIS_M_RE.match(tok)
        if m and out["visibility_m"] is None and out["wind"] is not None:
            out["visibility_m"] = float(m.group("m"))
            continue
        if _RVR_RE.match(tok):
            continue
        m = _SKY_RE.match(tok)
        if m:
            h = m.group("h")
            out["sky"].append({"cover": m.group("cov"), "height_ft": None if not h or h == "///" else int(h) * 100,
                               "convective": (m.group("cb") or "").strip("/")})
            continue
        m = _TEMP_RE.match(tok)
        if m and out["temp_c"] is None:
            if m.group("t") != "//":
                out["temp_c"] = units.parse_temperature_c(m.group("t"))
            if m.group("d") and m.group("d") != "//":
                out["dew_c"] = units.parse_temperature_c(m.group("d"))
            continue
        m = _ALT_RE.match(tok)
        if m:
            v = float(m.group("v"))
            out["altimeter_hpa"] = v / 100.0 * units.INHG_TO_HPA if m.group("k") == "A" else v
            continue
        if tok == "NSW":
            continue
        g = parse_weather_group(tok)
        if g is not None:
            (out["recent_weather"] if g["recent"] else out["weather"]).append(g)
            continue
    if rmk:
        out["remarks"] = _parse_remarks(rmk)
    return out


# ---------------------------------------------------------------------------
# Mapping to scene parameters
# ---------------------------------------------------------------------------
def _intensity_index(g: dict) -> int:
    return {"light": 0, "moderate": 1, "heavy": 2, "vicinity": 0}[g["intensity"]]


def weather_assignments(groups: list[dict], temp_c: Optional[float] = None) -> dict[str, Any]:
    """Map present-weather groups to scene path assignments (shared with the text parser)."""
    a: dict[str, Any] = {}
    rate = 0.0
    ptype: Optional[str] = None
    showers = False
    thunder = False
    thunder_vicinity = False
    squall = False
    heaviest = 0
    for g in groups:
        desc, ph = g["descriptor"], g["phenomena"]
        idx = _intensity_index(g)
        if desc == "TS":
            if g["intensity"] == "vicinity":
                thunder_vicinity = True
            else:
                thunder = True
                heaviest = max(heaviest, idx)
        if desc == "SH":
            showers = True
        if "SQ" in ph:
            squall = True
        if g["intensity"] == "vicinity" and desc != "TS":
            continue                              # VCSH etc.: not at the station
        precip = [p for p in ph if p in _RATES]
        if not precip:
            continue
        r = sum(_RATES[p][idx] for p in precip)
        rate = max(rate, r)
        kinds = set(precip)
        if desc == "FZ" and kinds & {"RA", "DZ"}:
            t = "freezing_rain"
        elif "GR" in kinds:
            t = "hail"
        elif "GS" in kinds:
            t = "graupel"
        elif "PL" in kinds or ("RA" in kinds and "SN" in kinds) or ("DZ" in kinds and "SN" in kinds):
            t = "sleet"
        elif "SN" in kinds or "SG" in kinds or "IC" in kinds:
            t = "snow"
        elif kinds == {"DZ"}:
            t = "drizzle"
        else:
            t = "showers" if desc == "SH" else "rain"
        order = ("drizzle", "rain", "showers", "snow", "sleet", "freezing_rain", "graupel", "hail")
        if ptype is None or order.index(t) > order.index(ptype):
            ptype = t
    if ptype is not None:
        a["weather.precipitation.type"] = ptype
        a["weather.precipitation.rate_mm_h"] = rate
        if showers or thunder:
            a["weather.precipitation.character"] = "showery"
            a["weather.precipitation.shower_period_min"] = 10.0 if thunder else 14.0
        if ptype == "drizzle":
            a["weather.precipitation.drops"] = "fine"
        elif thunder:
            a["weather.precipitation.drops"] = "fat"
        if ptype == "hail":
            a["weather.precipitation.hail_size_mm"] = 19.0 if heaviest >= 2 else 10.0
    if thunder or thunder_vicinity:
        a["weather.thunder.enabled"] = True
        if thunder:
            nm = 2.0 if heaviest >= 2 else 3.0          # within the ASOS 5 NM 'TS' radius
            a["weather.thunder.distance_km"] = nm * NM_KM
            a["weather.thunder.flashes_per_min"] = 4.0 if heaviest >= 2 else 2.0
            a["weather.thunder.stroke_intensity"] = 0.75 if heaviest >= 2 else 0.55
        else:
            a["weather.thunder.distance_km"] = 7.5 * NM_KM
            a["weather.thunder.flashes_per_min"] = 1.0
    if squall:
        a["weather.wind.gust_factor"] = 2.0
        a["weather.wind.gust_rate_per_min"] = 4.0
    return a


def metar_assignments(parsed: dict, utc_offset_h: Optional[float] = None) -> dict[str, Any]:
    """Scene path assignments for a parsed METAR."""
    a: dict[str, Any] = {}
    w = parsed.get("wind")
    if w:
        a["weather.wind.speed_mps"] = float(w["speed_mps"])
        if w.get("calm"):
            pass                                          # 00000KT: calm has no direction
        elif w["dir_deg"] is not None:
            a["weather.wind.from_deg"] = float(w["dir_deg"]) % 360.0
        elif w.get("var_from") is not None:
            lo, hi = w["var_from"], w["var_to"]
            a["weather.wind.from_deg"] = (lo + ((hi - lo) % 360.0) / 2.0) % 360.0
        if w.get("gust_mps") and w["speed_mps"] > 0.5:
            a["weather.wind.gust_factor"] = float(min(2.5, max(1.05, w["gust_mps"] / w["speed_mps"])))
        if w.get("var_from") is not None or w.get("variable"):
            sector = ((w["var_to"] - w["var_from"]) % 360.0) if w.get("var_from") is not None else 180.0
            a["weather.wind.turbulence_intensity"] = float(min(0.5, 0.15 + sector / 600.0))
    if parsed.get("visibility_m") is not None:
        a["weather.visibility_km"] = float(min(50.0, max(0.05, parsed["visibility_m"] / 1000.0)))
    if parsed.get("temp_c") is not None:
        a["weather.temperature_c"] = float(parsed["temp_c"])
        if parsed.get("dew_c") is not None:
            a["weather.humidity_pct"] = units.humidity_from_dew_point(parsed["temp_c"], parsed["dew_c"])
    if parsed.get("altimeter_hpa") is not None:
        a["weather.pressure_hpa"] = float(parsed["altimeter_hpa"])
    if parsed.get("sky"):
        a["weather.cloud_okta"] = max(_OKTA.get(s["cover"], 0.0) for s in parsed["sky"])
    elif parsed.get("cavok"):
        a["weather.cloud_okta"] = 1.0
    if parsed.get("time") is not None and utc_offset_h is not None:
        t = parsed["time"]
        a["weather.hour"] = (t["hour"] + t["minute"] / 60.0 + float(utc_offset_h)) % 24.0

    wx = weather_assignments(parsed.get("weather") or [], parsed.get("temp_c"))
    a.update(wx)
    has_precip = "weather.precipitation.type" in wx
    if not has_precip:
        a["weather.precipitation.type"] = "none"
        a["weather.precipitation.rate_mm_h"] = 0.0
        recent = parsed.get("recent_weather") or []
        if any(set(g["phenomena"]) & set(PRECIP) or g["descriptor"] == "TS" for g in recent):
            a["weather.storm_arc"] = "clearing"           # aftermath: drips and runoff
    # convective sky without thunder at the station: fat drops
    if has_precip and any(s.get("convective") in ("CB", "TCU") for s in parsed.get("sky") or []):
        a.setdefault("weather.precipitation.drops", "fat")

    rmk = parsed.get("remarks") or {}
    ltg = rmk.get("lightning")
    ts = rmk.get("thunderstorm")
    if ltg:
        a["weather.thunder.enabled"] = True
        where = ltg.get("where", "")
        if where == "DSNT" and "weather.thunder.distance_km" not in wx:
            a["weather.thunder.distance_km"] = min(25.0, 12.0 * NM_KM)
        elif where == "VC" and "weather.thunder.distance_km" not in wx:
            a["weather.thunder.distance_km"] = 7.5 * NM_KM
        elif where == "OHD":
            a["weather.thunder.distance_km"] = 1.0
        elif "weather.thunder.distance_km" not in a:
            a["weather.thunder.distance_km"] = 3.0 * NM_KM
        freq = ltg.get("frequency", "")
        a["weather.thunder.flashes_per_min"] = {"occasional": 0.7, "frequent": 3.0, "continuous": 8.0}.get(
            freq, a.get("weather.thunder.flashes_per_min", 1.0))
        types = ltg.get("types") or []
        if types:
            a["weather.thunder.cg_fraction"] = float(min(1.0, sum(1 for t in types if t in ("CG", "CA")) / len(types) + 0.1))
        dirs = [d for d in ltg.get("directions", []) if d in _DIRS8]
        if dirs:
            a["weather.thunder.bearing_deg"] = _DIRS8[dirs[0]]
    if ts:
        a["weather.thunder.enabled"] = True
        if ts.get("where") == "OHD":
            a["weather.thunder.distance_km"] = 1.0
        elif ts.get("vicinity") and "weather.thunder.distance_km" not in a:
            a["weather.thunder.distance_km"] = 7.5 * NM_KM
        elif ts.get("where") == "DSNT":
            a["weather.thunder.distance_km"] = min(25.0, 12.0 * NM_KM)
        dirs = [d for d in ts.get("directions", []) if d in _DIRS8]
        if dirs:
            a["weather.thunder.bearing_deg"] = _DIRS8[dirs[0]]
        mov = ts.get("moving", "")
        if mov in _DIRS8:
            bearing = a.get("weather.thunder.bearing_deg")
            if bearing is not None:
                # storm moving toward the station when its motion points back along the bearing
                toward = (_DIRS8[mov] - (bearing + 180.0)) % 360.0
                closing = 1.0 if (toward < 45.0 or toward > 315.0) else (-1.0 if 135.0 < toward < 225.0 else 0.0)
                a["weather.thunder.approach_kmh"] = 25.0 * closing
        elif mov == "STNRY":
            a["weather.thunder.approach_kmh"] = 0.0
    if a.get("weather.thunder.enabled") and "weather.thunder.bearing_deg" not in a and "weather.wind.from_deg" in a:
        a["weather.thunder.bearing_deg"] = a["weather.wind.from_deg"]   # cells usually arrive from upwind
    pk = rmk.get("peak_wind")
    if pk and w and w["speed_kt"] > 1.0 and "weather.wind.gust_factor" not in a:
        a["weather.wind.gust_factor"] = float(min(2.5, max(1.05, pk["speed_kt"] / w["speed_kt"])))
    return a


def _set(scene: Scene, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = scene
    for part in parts[:-1]:
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    else:
        setattr(obj, parts[-1], value)


def scene_from_metar(text: str, base: Optional[Scene] = None, utc_offset_h: Optional[float] = None) -> Scene:
    """Scene from a METAR string, on top of `base` (a fresh Scene by default).

    The METAR time is UTC; pass `utc_offset_h` to set the local hour, otherwise the
    base scene's hour is kept.
    """
    parsed = parse_metar(text)
    scene = base.copy() if base is not None else Scene()
    for path, value in metar_assignments(parsed, utc_offset_h).items():
        _set(scene, path, value)
    if scene.weather.thunder.enabled and scene.weather.precipitation.type == "none":
        scene.weather.thunder.flashes_per_min = min(scene.weather.thunder.flashes_per_min, 1.0)
    scene.name = f"METAR {parsed['station'] or ''}".strip() + (f": {_summary(parsed)}" if _summary(parsed) else "")
    return scene


def _summary(parsed: dict) -> str:
    parts = []
    for g in parsed.get("weather") or []:
        parts.append(g["raw"])
    w = parsed.get("wind")
    if w:
        d = "VRB" if w["dir_deg"] is None else f"{w['dir_deg']:03.0f}"
        parts.append(f"wind {d}/{w['speed_kt']:.0f}kt" + (f" G{w['gust_kt']:.0f}" if w.get("gust_kt") else ""))
    if parsed.get("temp_c") is not None:
        parts.append(f"{parsed['temp_c']:.0f}°C")
    return ", ".join(parts)


def looks_like_metar(text: str) -> bool:
    toks = text.strip().upper().split()
    if len(toks) < 3:
        return False
    has_time = any(_TIME_RE.match(t) for t in toks[:4])
    has_wind = any(_WIND_RE.match(t) for t in toks[:6])
    return has_time and has_wind
