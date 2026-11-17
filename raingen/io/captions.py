"""Timed event export: SRT / WebVTT captions, CSV cue lists and CMX3600 EDL markers.

Captions follow sound-description practice for deaf and hard-of-hearing viewers (plan
§2.9; DCMP Captioning Key, BBC and Netflix SDH guidelines): one short sentence in square
brackets, the sound source first, then what it does and where, in plain words:
"[Distant thunder rumbles to the west]". No numbers or units: the data form (km, degrees,
dB) stays in the cue CSV. Cue rules: events of the same kind within 10 s merge into one cue
(the most salient wording wins), every cue shows for at least 1.5 s, and a cue has at most
two lines of at most 42 characters. Visual-only events (lightning flashes) are kept out of
SRT/VTT by default but always go to CSV/EDL, where they serve VFX and lighting sync.
Thunder captions start at the sound's arrival (data['arrival_t'] when the marker records
the flash time).
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from typing import Iterable, Optional

from ..core.types import EventMarker
from .wav import exact_fps, timecode_to_frames

_COMPASS8 = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
# listener frame: azimuth 0 = ahead, +90 deg = left
_RELATIVE8 = ("ahead", "ahead to the left", "to the left", "behind to the left", "behind",
              "behind to the right", "to the right", "ahead to the right")
_DEFAULT_DURATION = {"thunder": 4.0, "lightning": 0.5, "gust": 3.0, "drip": 1.5, "hail": 3.0,
                     "rain_start": 3.0, "rain_stop": 3.0, "bird": 2.0, "frog": 2.0, "traffic": 3.0,
                     "gutter_overflow": 3.0}
VISUAL_KINDS = {"lightning", "flash"}
MERGE_WINDOW_S = 10.0          # plan §2.9: events within 10 s merge
MIN_CUE_S = 1.5                # plan §2.9: each cue shows >= 1.5 s
MAX_LINE_CHARS = 42            # common SDH line length (BBC / Netflix)
MAX_LINES = 2
_WATERY = {"puddle", "water", "pond", "bucket", "barrel", "pool", "lake"}


def _compass(bearing_deg: float) -> str:
    return _COMPASS8[int(math.floor(((float(bearing_deg) % 360.0) + 22.5) / 45.0)) % 8]


def _direction(data: dict) -> Optional[str]:
    """Direction as a word for the CSV ('west', 'left', ...)."""
    if data.get("direction"):
        return str(data["direction"])
    b = data.get("bearing_deg")
    if b is not None:
        return _compass(float(b))
    az = data.get("azimuth")
    if az is not None:
        deg = math.degrees(float(az)) % 360.0
        rel = ("ahead", "front-left", "left", "behind-left", "behind", "behind-right", "right", "front-right")
        return rel[int(math.floor((deg + 22.5) / 45.0)) % 8]
    return None


def _where(data: dict) -> Optional[str]:
    """Direction as a caption phrase: 'to the west', 'to the left', 'behind', 'overhead'."""
    d = _distance_km(data)
    if d is not None and d < 0.3:
        return "overhead"
    if data.get("direction"):
        word = str(data["direction"]).strip().lower()
        return f"to the {word}" if word in _COMPASS8 or word in ("left", "right") else word
    b = data.get("bearing_deg")
    if b is not None:
        return f"to the {_compass(float(b))}"
    az = data.get("azimuth")
    if az is not None:
        deg = math.degrees(float(az)) % 360.0
        return _RELATIVE8[int(math.floor((deg + 22.5) / 45.0)) % 8]
    return None


def _distance_km(data: dict) -> Optional[float]:
    d = data.get("distance_km")
    if d is None and data.get("distance_m") is not None:
        d = float(data["distance_m"]) / 1000.0
    return None if d is None else float(d)


def _with_where(name: str, where: Optional[str]) -> str:
    return f"{name} {where}" if where else name


def _thunder_text(data: dict) -> str:
    d = _distance_km(data)
    where = _where(data)
    peak = data.get("peak_db_spl")
    loud = (peak is not None and float(peak) >= 100.0) or float(data.get("intensity", 0.0) or 0.0) >= 0.85
    if d is None:
        return _with_where("Loud thunder crashes" if loud else "Thunder rumbles", where)
    if d < 1.0:
        return _with_where("Thunder cracks", where)
    if d < 3.0:
        return _with_where("Loud thunder crashes" if loud else "Thunder crashes", where)
    if d <= 10.0:
        return _with_where("Thunder rumbles", where)
    return _with_where("Distant thunder rumbles", where)


def caption_for(marker: EventMarker) -> str:
    """Accessible caption text for one marker, e.g. '[Distant thunder rumbles to the west]'."""
    data = marker.data or {}
    if data.get("caption"):
        return str(data["caption"])
    kind = str(marker.kind).lower()
    if kind == "thunder":
        text = _thunder_text(data)
    elif kind in ("lightning", "flash"):
        text = _with_where("Lightning flashes", _where(data))
    elif kind == "gust":
        speed = data.get("wind_10m_mps", data.get("speed_mps"))
        bft = data.get("beaufort")
        strong = (bft is not None and float(bft) >= 7.0) or (speed is not None and float(speed) >= 13.9)
        text = "A strong gust of wind" if strong else "Wind gusts"
        if data.get("from_deg") is not None:
            text += f" from the {_compass(float(data['from_deg']))}"
    elif kind == "drip":
        target = str(data.get("target") or "").replace("_", " ").strip().lower()
        if not target:
            text = "Water drips nearby"
        elif target in _WATERY:
            text = f"Water drips into the {target}"
        else:
            text = f"Water drips onto the {target}"
    elif kind == "gutter_overflow":
        text = "Water spills over the gutter"
    elif kind == "hail":
        species = str(data.get("type") or "hail").lower()
        text = {"graupel": "Graupel patters", "sleet": "Sleet patters", "ice_pellets": "Ice pellets tick",
                "snow": "Snow hisses softly"}.get(species, "Hail clatters")
    elif kind == "rain_start":
        text = "Rain begins"
    elif kind == "rain_stop":
        text = "Rain stops"
    else:
        name = str(data.get("label") or kind.replace("_", " ")).strip()
        text = name[:1].upper() + name[1:]
    return "[" + text + "]"


def _salience(marker: EventMarker) -> float:
    """Which of several merged events names the cue: closest / loudest first."""
    data = marker.data or {}
    if data.get("peak_db_spl") is not None:
        return float(data["peak_db_spl"])
    d = _distance_km(data)
    if d is not None:
        return -d
    return float(data.get("intensity", data.get("gust", 0.0)) or 0.0)


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


def _wrap(text: str) -> list[str]:
    """Lines of <= 42 characters, never more than two (an over-long custom caption widens line 2)."""
    lines = textwrap.wrap(text, MAX_LINE_CHARS, break_long_words=False, break_on_hyphens=False) or [text]
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + [" ".join(lines[MAX_LINES - 1:])]
    return lines


def _cues(markers: Iterable[EventMarker], include_visual: bool = False, merge_gap_s: float = MERGE_WINDOW_S,
          min_dur_s: float = MIN_CUE_S):
    """[start, end, text, marker] sorted, non-overlapping, >= min_dur_s each, <= 2 lines of <= 42 chars."""
    items = []
    for m in markers:
        kind = str(m.kind).lower()
        if not include_visual and kind in VISUAL_KINDS:
            continue
        s = max(0.0, _start(m))
        items.append((s, s + _duration(m), kind, m))
    items.sort(key=lambda it: (it[0], it[2]))

    # 1. events of one kind within merge_gap_s of the cue's first event become one cue
    cues: list[dict] = []
    open_by_kind: dict[str, dict] = {}
    for s, e, kind, m in items:
        cue = open_by_kind.get(kind)
        if cue is not None and s - cue["start"] <= merge_gap_s:
            cue["end"] = max(cue["end"], e)
            if _salience(m) > _salience(cue["marker"]):
                cue["marker"] = m
            continue
        cue = {"start": s, "end": e, "marker": m}
        open_by_kind[kind] = cue
        cues.append(cue)
    for c in cues:
        c["lines"] = _wrap(caption_for(c["marker"]))
        c["end"] = max(c["end"], c["start"] + min_dur_s)
    cues.sort(key=lambda c: c["start"])

    # 2. no overlapping display: trim, stack two one-line captions, or delay the later cue
    out: list[dict] = []
    for c in cues:
        if out and c["start"] < out[-1]["end"]:
            prev = out[-1]
            if c["lines"] == prev["lines"] or all(line in prev["lines"] for line in c["lines"]):
                prev["end"] = max(prev["end"], c["end"])
                continue
            if c["start"] - prev["start"] >= min_dur_s:
                prev["end"] = c["start"]
            elif len(prev["lines"]) + len(c["lines"]) <= MAX_LINES:
                prev["lines"] = prev["lines"] + c["lines"]
                prev["end"] = max(prev["end"], c["end"])
                continue
            else:
                c["start"] = prev["start"] + min_dur_s
                prev["end"] = c["start"]
                c["end"] = max(c["end"], c["start"] + min_dur_s)
        out.append(c)
    return [[c["start"], c["end"], "\n".join(c["lines"]), c["marker"]] for c in out]


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


def _df_label(frame: int, nominal: int) -> tuple[int, int, int, int]:
    """Frame count -> SMPTE drop-frame HH, MM, SS, FF (29.97 / 59.94)."""
    drop = nominal // 15
    per_10min = nominal * 600 - drop * 9
    per_min = nominal * 60 - drop
    tens, rem = divmod(frame, per_10min)
    frame += drop * 9 * tens + (drop * ((rem - drop) // per_min) if rem > drop else 0)
    return (frame // (nominal * 3600), (frame // (nominal * 60)) % 60, (frame // nominal) % 60, frame % nominal)


def seconds_to_timecode(t: float, fps: float = 24, start_tc: str = "00:00:00:00") -> str:
    """Seconds -> timecode HH:MM:SS:FF offset by start_tc (drop-frame HH:MM:SS;FF if start_tc uses ';')."""
    nominal = int(round(float(fps)))
    drop = ";" in str(start_tc)
    total = timecode_to_frames(start_tc, fps) + int(math.floor(max(0.0, float(t)) * exact_fps(fps) + 1e-6))
    if drop:
        total %= 24 * 6 * (nominal * 600 - (nominal // 15) * 9)
        hh, mm, ss, ff = _df_label(total, nominal)
        return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"
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
            dist = _distance_km(d)
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
    """CMX3600 EDL with one marker event (+ '* LOC:' locator) per weather event; returns events written.

    CMX3600 event numbers have three digits, so they run 001..999 and then wrap to 001.
    """
    rows = sorted(list(markers), key=lambda m: _start(m) if str(m.kind).lower() == "thunder" else float(m.t))
    nominal = int(round(float(fps)))
    drop = ";" in str(start_tc)
    src_zero = "00:00:00;00" if drop else "00:00:00:00"
    with open(path, "w", encoding="ascii", errors="replace", newline="\r\n") as fh:
        fh.write(f"TITLE: {title}\n")
        fh.write(f"FCM: {'DROP FRAME' if drop else 'NON-DROP FRAME'}\n\n")
        for i, m in enumerate(rows, start=1):
            t0 = _start(m) if str(m.kind).lower() == "thunder" else float(m.t)
            t1 = t0 + max(1.0 / nominal, _duration(m))
            rec_in = seconds_to_timecode(t0, fps, start_tc)
            rec_out = seconds_to_timecode(t1, fps, start_tc)
            src_in = seconds_to_timecode(t0, fps, src_zero)
            src_out = seconds_to_timecode(t1, fps, src_zero)
            cap = caption_for(m).strip("[]")
            fh.write(f"{(i - 1) % 999 + 1:03d}  AX       AA    C        {src_in} {src_out} {rec_in} {rec_out}\n")
            fh.write(f"* FROM CLIP NAME: {cap}\n")
            fh.write(f"* LOC: {rec_in} {_EDL_COLORS.get(str(m.kind).lower(), 'GREEN'):<7} {cap}\n\n")
    return len(rows)
