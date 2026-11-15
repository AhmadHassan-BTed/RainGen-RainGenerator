"""Timed event export: SRT / WebVTT captions, CSV cue lists and CMX3600 EDL markers.

Captions follow sound-description practice for deaf and hard-of-hearing viewers (square
brackets, sound source first, then distance and direction): "[Thunder, 3.1 km, west]".
Visual-only events (lightning flashes) are kept out of SRT/VTT by default but always go to
CSV/EDL, where they serve VFX and lighting sync. Thunder captions start at the sound's
arrival (data['arrival_t'] when the marker records the flash time).
"""

from __future__ import annotations

import csv
import json
import math
from typing import Iterable, Optional

from ..core.types import EventMarker

_COMPASS8 = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
_DEFAULT_DURATION = {"thunder": 4.0, "lightning": 0.5, "gust": 3.0, "drip": 1.5, "hail": 3.0,
                     "rain_start": 3.0, "rain_stop": 3.0, "bird": 2.0, "frog": 2.0, "traffic": 3.0}
VISUAL_KINDS = {"lightning", "flash"}


def _direction(data: dict) -> Optional[str]:
    if data.get("direction"):
        return str(data["direction"])
    b = data.get("bearing_deg")
    if b is not None:
        return _COMPASS8[int(math.floor(((float(b) % 360.0) + 22.5) / 45.0)) % 8]
    az = data.get("azimuth")
    if az is not None:
        deg = math.degrees(float(az)) % 360.0     # listener frame: 0 front, 90 left
        rel = ("ahead", "front-left", "left", "behind-left", "behind", "behind-right", "right", "front-right")
        return rel[int(math.floor((deg + 22.5) / 45.0)) % 8]
    return None


def _distance(data: dict) -> Optional[str]:
    d = data.get("distance_km")
    if d is None and data.get("distance_m") is not None:
        d = float(data["distance_m"]) / 1000.0
    if d is None:
        return None
    d = float(d)
    if d < 0.3:
        return "overhead"
    return f"{d:.1f} km" if d < 10.0 else f"{d:.0f} km"


def caption_for(marker: EventMarker) -> str:
    """Human caption text for one marker."""
    data = marker.data or {}
    if data.get("caption"):
        return str(data["caption"])
    kind = str(marker.kind).lower()
    dist = _distance(data)
    direc = _direction(data)
    if kind == "thunder":
        d = data.get("distance_km")
        d = float(d) if d is not None else None
        intensity = float(data.get("intensity", 0.5))
        if d is not None and d < 1.0:
            name = "Thunder crack" if intensity >= 0.3 else "Thunder"
        elif d is not None and d > 10.0:
            name = "Distant thunder rumbles"
        else:
            name = "Loud thunder" if intensity >= 0.85 else "Thunder"
    elif kind in ("lightning", "flash"):
        name = "Lightning flash"
    elif kind == "gust":
        g = float(data.get("gust", data.get("strength", 1.0)))
        speed = data.get("speed_mps")
        name = "Strong wind gust" if (g >= 1.6 or (speed is not None and float(speed) >= 13.9)) else "Wind gusts"
    elif kind == "drip":
        target = str(data.get("target") or "").replace("_", " ").strip()
        name = f"Water drips onto {target}" if target else "Dripping"
    elif kind == "hail":
        species = str(data.get("type") or "hail").lower()
        name = {"graupel": "Graupel patters", "sleet": "Sleet patters", "ice_pellets": "Ice pellets tick",
                "snow": "Snow hisses softly"}.get(species, "Hail clatters")
    elif kind == "rain_start":
        name = "Rain begins"
    elif kind == "rain_stop":
        name = "Rain eases off"
    else:
        name = str(data.get("label") or kind.replace("_", " ")).strip()
        name = name[:1].upper() + name[1:]
    parts = [name]
    if dist and kind not in ("drip",):
        parts.append(dist)
    if direc and dist != "overhead":
        parts.append(direc)
    return "[" + ", ".join(parts) + "]"


def _start(marker: EventMarker) -> float:
    data = marker.data or {}
    if str(marker.kind).lower() == "thunder" and data.get("arrival_t") is not None:
        return float(data["arrival_t"])
    return float(marker.t)


def _duration(marker: EventMarker) -> float:
    data = marker.data or {}
    if data.get("duration_s") is not None:
        return max(0.2, float(data["duration_s"]))
    kind = str(marker.kind).lower()
    if kind == "thunder" and data.get("distance_km") is not None:
        return min(8.0, 2.5 + 0.4 * float(data["distance_km"]))
    return _DEFAULT_DURATION.get(kind, 2.0)


def _cues(markers: Iterable[EventMarker], include_visual: bool = False, merge_gap_s: float = 0.75):
    """(start, end, text, marker) sorted; identical captions that touch are merged."""
    items = []
    for m in markers:
        if not include_visual and str(m.kind).lower() in VISUAL_KINDS:
            continue
        s = max(0.0, _start(m))
        items.append([s, s + _duration(m), caption_for(m), m])
    items.sort(key=lambda it: it[0])
    merged: list[list] = []
    for it in items:
        if merged and merged[-1][2] == it[2] and it[0] <= merged[-1][1] + merge_gap_s:
            merged[-1][1] = max(merged[-1][1], it[1])
        else:
            merged.append(it)
    # avoid overlapping display: trim each cue to the next one's start (min 0.5 s)
    for i in range(len(merged) - 1):
        nxt = merged[i + 1][0]
        if merged[i][1] > nxt:
            merged[i][1] = max(merged[i][0] + 0.5, nxt)
    return merged


def _ts(t: float, sep: str) -> str:
    ms_total = int(round(max(0.0, t) * 1000.0))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(markers: Iterable[EventMarker], path: str, include_visual: bool = False) -> int:
    """SubRip captions; returns the number of cues written."""
    cues = _cues(markers, include_visual)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i, (s, e, text, _) in enumerate(cues, start=1):
            fh.write(f"{i}\n{_ts(s, ',')} --> {_ts(e, ',')}\n{text}\n\n")
    return len(cues)


def write_vtt(markers: Iterable[EventMarker], path: str, include_visual: bool = False) -> int:
    """WebVTT captions; returns the number of cues written."""
    cues = _cues(markers, include_visual)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("WEBVTT\n\nNOTE raingen weather sound captions\n\n")
        for i, (s, e, text, _) in enumerate(cues, start=1):
            fh.write(f"{i}\n{_ts(s, '.')} --> {_ts(e, '.')}\n{text}\n\n")
    return len(cues)


def seconds_to_timecode(t: float, fps: float = 24, start_tc: str = "00:00:00:00") -> str:
    """Seconds -> non-drop-frame timecode HH:MM:SS:FF offset by start_tc."""
    nominal = int(round(float(fps)))
    parts = [int(p) for p in str(start_tc).replace(";", ":").split(":")]
    start_frames = ((parts[0] * 60 + parts[1]) * 60 + parts[2]) * nominal + parts[3]
    total = start_frames + int(math.floor(max(0.0, float(t)) * float(fps) + 1e-6))
    total %= 24 * 3600 * nominal
    ff = total % nominal
    ss = (total // nominal) % 60
    mm = (total // (nominal * 60)) % 60
    hh = total // (nominal * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def write_csv(markers: Iterable[EventMarker], path: str, fps: float = 24, start_tc: str = "00:00:00:00") -> int:
    """Every marker (including lightning) as a cue sheet row; returns rows written."""
    rows = sorted(list(markers), key=lambda m: float(m.t))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "timecode", "kind", "caption", "sound_time_s", "duration_s", "distance_km",
                    "bearing_deg", "direction", "intensity", "data_json"])
        for m in rows:
            d = m.data or {}
            dist = d.get("distance_km")
            if dist is None and d.get("distance_m") is not None:
                dist = float(d["distance_m"]) / 1000.0
            w.writerow([
                f"{float(m.t):.4f}",
                seconds_to_timecode(float(m.t), fps, start_tc),
                m.kind,
                caption_for(m),
                f"{_start(m):.4f}",
                f"{_duration(m):.3f}",
                "" if dist is None else f"{float(dist):.3f}",
                "" if d.get("bearing_deg") is None else f"{float(d['bearing_deg']):.1f}",
                _direction(d) or "",
                "" if d.get("intensity") is None else f"{float(d['intensity']):.3f}",
                json.dumps(d, default=str, sort_keys=True),
            ])
    return len(rows)


_EDL_COLORS = {"thunder": "RED", "lightning": "YELLOW", "gust": "CYAN", "drip": "BLUE", "hail": "WHITE"}


def write_edl(markers: Iterable[EventMarker], path: str, fps: float = 24, start_tc: str = "01:00:00:00",
              title: str = "raingen weather events") -> int:
    """CMX3600 EDL with one marker event (+ '* LOC:' locator) per weather event; returns events written."""
    rows = sorted(list(markers), key=lambda m: _start(m) if str(m.kind).lower() == "thunder" else float(m.t))
    nominal = int(round(float(fps)))
    with open(path, "w", encoding="ascii", errors="replace", newline="\r\n") as fh:
        fh.write(f"TITLE: {title}\n")
        fh.write("FCM: NON-DROP FRAME\n\n")
        for i, m in enumerate(rows, start=1):
            t0 = _start(m) if str(m.kind).lower() == "thunder" else float(m.t)
            t1 = t0 + max(1.0 / nominal, _duration(m))
            rec_in = seconds_to_timecode(t0, fps, start_tc)
            rec_out = seconds_to_timecode(t1, fps, start_tc)
            src_in = seconds_to_timecode(t0, fps, "00:00:00:00")
            src_out = seconds_to_timecode(t1, fps, "00:00:00:00")
            cap = caption_for(m).strip("[]")
            fh.write(f"{i % 1000:03d}  AX       AA    C        {src_in} {src_out} {rec_in} {rec_out}\n")
            fh.write(f"* FROM CLIP NAME: {cap}\n")
            fh.write(f"* LOC: {rec_in} {_EDL_COLORS.get(str(m.kind).lower(), 'GREEN'):<7} {cap}\n\n")
    return len(rows)
