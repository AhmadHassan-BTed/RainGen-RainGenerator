"""Streaming Broadcast-WAV / RF64 writer (and FLAC via python-soundfile).

Layout written up front (never holding the audio in memory):

    RIFF <size> WAVE
      JUNK (28-byte payload)   <- reserved so the file can become RF64 in place
      fmt                     WAVE_FORMAT_PCM, or WAVE_FORMAT_EXTENSIBLE for > 2 channels,
                              24-bit or float, with dwChannelMask and the KSDATAFORMAT GUID
      fact                    (float only) sample frames
      bext                    EBU Tech 3285 v2: description, originator, date/time,
                              TimeReference (samples since midnight), UMID, loudness x100
                              (integrated, LRA, max true peak, max momentary, max short-term;
                              0x7FFF = not set, also used for -inf/NaN), CodingHistory
      iXML                    iXML 2.x with UCS fields and the full scene in <USER>
      data                    interleaved little-endian int16 / int24 / float32

On close, when the RIFF size exceeds 2 GiB (or `rf64_threshold`; 0 forces RF64) the header
is rewritten in place to RF64 (EBU Tech 3306): 'RIFF' -> 'RF64' (or 'BW64', ITU-R BS.2088),
the JUNK chunk becomes 'ds64' carrying the 64-bit RIFF/data sizes and sample count, and the
32-bit size fields are set to 0xFFFFFFFF.

The writer itself writes in place; callers that need an atomic result write to a temporary
name and move it with os.replace() (which, unlike os.rename, overwrites on Windows too).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import struct
from typing import Any, Optional
from xml.sax.saxutils import escape

import numpy as np

RF64_THRESHOLD = 2 ** 31 - 1          # bytes of RIFF payload above which we switch to RF64
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_GUID_TAIL = b"\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_BEXT_FMT = "<256s32s32s10s8sIIH64s5h180s"      # 602 bytes
_BEXT_LOUDNESS_OFFSET = 256 + 32 + 32 + 10 + 8 + 4 + 4 + 2 + 64
_LOUDNESS_UNKNOWN = 0x7FFF

_DEFAULT_MASKS = {1: 0x4, 2: 0x3, 4: 0x33, 6: 0x60F, 8: 0x63F, 12: 0x2D63F}
_LAYOUT_MASKS = {"mono": 0x4, "stereo": 0x3, "binaural": 0x3, "5.1": 0x60F, "7.1": 0x63F,
                 "7.1.4": 0x2D63F, "foa": 0, "ambix": 0}


def layout_channel_mask(layout: Optional[str], channels: int) -> int:
    """dwChannelMask for a layout name (ambisonics -> 0, i.e. no speaker assignment)."""
    if layout:
        lay = str(layout).strip().lower()
        if lay.startswith("hoa") or lay in ("foa", "ambix", "ambisonic"):
            return 0
        if lay in _LAYOUT_MASKS:
            return _LAYOUT_MASKS[lay]
    return _DEFAULT_MASKS.get(int(channels), 0)


def exact_fps(fps: float) -> float:
    """Frame rate as used for real time: 29.97 -> 30000/1001, 23.976 -> 24000/1001, 25 -> 25."""
    rate = float(fps)
    if abs(rate - round(rate)) > 1e-6:
        r1001 = rate * 1001.0 / 1000.0
        if abs(r1001 - round(r1001)) < 0.01:
            return round(r1001) * 1000.0 / 1001.0
    return rate


def timecode_to_frames(tc: str, fps: float) -> int:
    """'HH:MM:SS:FF' -> frame count since 00:00:00:00.

    A ';' before the frames marks SMPTE drop-frame (29.97 / 59.94): frame numbers 0-1 (0-3
    at 59.94) are skipped every minute except every tenth (SMPTE ST 12-1).
    """
    text = str(tc).strip()
    drop = ";" in text
    parts = text.replace(";", ":").replace(".", ":").split(":")
    if len(parts) != 4:
        raise ValueError(f"bad timecode {tc!r}")
    h, m, s, f = (int(p) for p in parts)
    nominal = int(round(float(fps)))
    frames = ((h * 60 + m) * 60 + s) * nominal + f
    if drop:
        if nominal not in (30, 60):
            raise ValueError(f"drop-frame timecode needs 29.97 or 59.94 fps, got {fps}")
        per_min = nominal // 15                       # 2 at 29.97, 4 at 59.94
        minutes = h * 60 + m
        frames -= per_min * (minutes - minutes // 10)
    return frames


def timecode_to_seconds(tc: str, fps: float) -> float:
    """'HH:MM:SS:FF' -> seconds of real time (frames / fps, so 29.97 NDF runs 0.1 % slow).

    ';' marks drop-frame. 01:00:00:00 at 25 fps -> 3600 s; at 29.97 NDF -> 3603.6 s; at
    29.97 DF ('01:00:00;00') -> 3599.9964 s.
    """
    return timecode_to_frames(tc, fps) / exact_fps(fps)


def _ascii(text: Any, size: int) -> bytes:
    raw = str(text if text is not None else "").encode("ascii", "replace")
    return raw[:size]


def _loudness_x100(v: Optional[float]) -> int:
    """Tech 3285 v2 loudness field: round(100 x value) as int16; 0x7FFF = not set.

    Non-finite readings (-inf for silence, NaN) have no representable value and are written as
    'not set', never as a fake -327.68 LUFS.
    """
    if v is None:
        return _LOUDNESS_UNKNOWN
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _LOUDNESS_UNKNOWN
    if not math.isfinite(f):
        return _LOUDNESS_UNKNOWN
    return int(max(-32768, min(_LOUDNESS_UNKNOWN - 1, round(f * 100.0))))


def build_bext(bext: dict, sr: int, channels: int, bit_depth: int) -> bytes:
    """EBU Tech 3285 v2 bext payload from a dict of fields (all optional)."""
    now = _dt.datetime.now()
    tref = bext.get("time_reference")
    if tref is None and bext.get("start_tc"):
        tref = int(round(timecode_to_seconds(bext["start_tc"], bext.get("fps", 24)) * sr))
    tref = int(tref or 0)
    umid = bext.get("umid", b"")
    if isinstance(umid, str):
        umid = bytes.fromhex(umid) if all(c in "0123456789abcdefABCDEF" for c in umid) and len(umid) % 2 == 0 \
            else umid.encode("ascii", "replace")
    umid = bytes(umid)[:64].ljust(64, b"\0")
    mode = {1: "mono", 2: "stereo"}.get(channels, f"{channels}ch")
    fmt = "PCM" if bit_depth < 32 else "FLOAT"
    history = bext.get("coding_history")
    if history is None:
        tail = bext.get("coding_history_text", "raingen")
        history = f"A={fmt},F={sr},W={bit_depth},M={mode},T={tail}\r\n"
    history = str(history)
    if not history.endswith("\r\n"):
        history += "\r\n"
    payload = struct.pack(
        _BEXT_FMT,
        _ascii(bext.get("description", ""), 256),
        _ascii(bext.get("originator", "raingen"), 32),
        _ascii(bext.get("originator_reference", ""), 32),
        _ascii(bext.get("origination_date", now.strftime("%Y-%m-%d")), 10),
        _ascii(bext.get("origination_time", now.strftime("%H:%M:%S")), 8),
        tref & 0xFFFFFFFF,
        (tref >> 32) & 0xFFFFFFFF,
        2,
        umid,
        _loudness_x100(bext.get("loudness_value")),
        _loudness_x100(bext.get("loudness_range")),
        _loudness_x100(bext.get("max_true_peak_level")),
        _loudness_x100(bext.get("max_momentary_loudness")),
        _loudness_x100(bext.get("max_short_term_loudness")),
        b"\0" * 180,
    ) + history.encode("ascii", "replace")
    return payload


_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _xml(tag: str, value: Any, indent: str = "    ") -> str:
    text = _XML_ILLEGAL.sub(" ", str(value))                  # XML 1.0 forbids most C0 controls
    return f"{indent}<{tag}>{escape(text)}</{tag}>\n"


def _xml_tag(name: Any) -> str:
    tag = "".join(ch for ch in str(name).upper() if ch.isascii() and (ch.isalnum() or ch == "_")) or "FIELD"
    if tag[0].isdigit() or tag.startswith("XML"):
        tag = "F_" + tag                                      # names may not start with a digit or 'xml'
    return tag


def _timecode_rate(fps: float) -> str:
    """iXML TIMECODE_RATE as a ratio: 25 -> '25/1', 29.97 -> '30000/1001', 23.976 -> '24000/1001'."""
    f = float(fps)
    if abs(f - round(f)) < 1e-6:
        return f"{int(round(f))}/1"
    return f"{int(round(f * 1001.0 / 1000.0)) * 1000}/1001"


def build_ixml(ucs: Optional[dict] = None, scene: Any = None, channel_names: Optional[list] = None,
               project: str = "raingen", note: str = "", extra: Optional[dict] = None,
               fps: Optional[float] = None, sr: Optional[int] = None, drop_frame: bool = False) -> str:
    """iXML document with UCS fields and the scene JSON inside <USER>."""
    from .. import __version__ as engine_version

    ucs = dict(ucs or {})
    scene_json = None
    scene_name = ucs.get("FXName", "")
    seed = None
    if scene is not None:
        if hasattr(scene, "to_json"):
            scene_json = scene.to_json(indent=None)
            scene_name = getattr(scene, "name", scene_name)
            seed = getattr(scene, "seed", None)
        elif isinstance(scene, dict):
            scene_json = json.dumps(scene)
            seed = scene.get("seed")
            scene_name = scene.get("name", scene_name)
        else:
            scene_json = str(scene)
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<BWFXML>\n']
    out.append(_xml("IXML_VERSION", "2.10", "  "))
    out.append(_xml("PROJECT", project, "  "))
    out.append(_xml("SCENE", scene_name or "raingen", "  "))
    out.append(_xml("TAKE", "1", "  "))
    out.append(_xml("NOTE", note or ucs.get("Description", ""), "  "))
    if fps is not None or sr is not None:
        out.append("  <SPEED>\n")
        if fps is not None:
            out.append(_xml("TIMECODE_RATE", _timecode_rate(fps)))
            out.append(_xml("TIMECODE_FLAG", "DF" if drop_frame else "NDF"))
        if sr is not None:
            out.append(_xml("FILE_SAMPLE_RATE", int(sr)))
            out.append(_xml("DIGITIZER_SAMPLE_RATE", int(sr)))
        out.append("  </SPEED>\n")
    if channel_names:
        out.append("  <TRACK_LIST>\n")
        out.append(_xml("TRACK_COUNT", len(channel_names)))
        for i, nm in enumerate(channel_names, start=1):
            out.append("    <TRACK>\n")
            out.append(_xml("CHANNEL_INDEX", i, "      "))
            out.append(_xml("INTERLEAVE_INDEX", i, "      "))
            out.append(_xml("NAME", nm, "      "))
            out.append("    </TRACK>\n")
        out.append("  </TRACK_LIST>\n")
    out.append("  <USER>\n")
    field_map = (("CATID", "CatID"), ("CATEGORY", "Category"), ("SUBCATEGORY", "SubCategory"),
                 ("CATEGORYFULL", "CategoryFull"), ("FXNAME", "FXName"), ("DESCRIPTION", "Description"),
                 ("KEYWORDS", "Keywords"), ("LIBRARY", "Library"), ("DESIGNER", "Designer"),
                 ("SHOW", "Source"), ("TRACKTITLE", "TrackTitle"), ("MANUFACTURER", "Manufacturer"),
                 ("USERCATEGORY", "UserCategory"), ("VENDORCATEGORY", "VendorCategory"), ("NOTES", "Notes"))
    for tag, key in field_map:
        if key in ucs and ucs[key] not in (None, ""):
            val = ucs[key]
            if isinstance(val, (list, tuple)):
                val = ", ".join(str(v) for v in val)
            out.append(_xml(tag, val))
    out.append(_xml("EMBEDDER", "raingen"))
    out.append(_xml("RAINGEN_VERSION", engine_version))
    if seed is not None:
        out.append(_xml("RAINGEN_SEED", f"0x{int(seed):X}"))
    for k, v in (extra or {}).items():
        out.append(_xml(_xml_tag(k), v if not isinstance(v, (dict, list)) else json.dumps(v)))
    if scene_json is not None:
        out.append(_xml("RAINGEN_SCENE", scene_json))
    out.append("  </USER>\n</BWFXML>\n")
    return "".join(out)


def _chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\0" if len(payload) % 2 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


class StreamingWavWriter:
    """Write BWF/RF64 incrementally: .write(x (C, n)) many times, then .close()."""

    def __init__(self, path: str, sr: int, channels: int, bit_depth: int = 24, bext: Optional[dict] = None,
                 ixml: Optional[str] = None, channel_mask: Optional[int] = None,
                 rf64_threshold: int = RF64_THRESHOLD, bw64: bool = False):
        if int(bit_depth) not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24 or 32 (float)")
        self.path = str(path)
        self.sr = int(sr)
        self.channels = int(channels)
        self.bit_depth = int(bit_depth)
        self.is_float = self.bit_depth == 32
        self.bytes_per_sample = self.bit_depth // 8
        self.block_align = self.channels * self.bytes_per_sample
        self.rf64_threshold = int(rf64_threshold)
        self.bw64 = bool(bw64)
        self.frames = 0
        self._data_bytes = 0
        self._closed = False
        self._loudness: Optional[dict] = None
        self.channel_mask = int(channel_mask) if channel_mask is not None else \
            _DEFAULT_MASKS.get(self.channels, 0)

        self._f = open(self.path, "wb")
        f = self._f
        f.write(b"RIFF" + struct.pack("<I", 0) + b"WAVE")
        self._junk_pos = f.tell()
        f.write(_chunk(b"JUNK", b"\0" * 28))
        f.write(_chunk(b"fmt ", self._fmt_payload()))
        self._fact_pos = None
        if self.is_float:
            self._fact_pos = f.tell() + 8
            f.write(_chunk(b"fact", struct.pack("<I", 0)))
        # Metadata chunks are kept even-sized so no RIFF pad byte follows them: libsndfile's
        # RF64 reader does not skip pad bytes and would lose the 'data' chunk.
        self._bext_loudness_pos = None
        if bext is not None:
            payload = build_bext(bext, self.sr, self.channels, self.bit_depth)
            if len(payload) % 2:
                payload += b"\0"                               # NUL after the CodingHistory text
            self._bext_loudness_pos = f.tell() + 8 + _BEXT_LOUDNESS_OFFSET
            f.write(_chunk(b"bext", payload))
        if ixml:
            doc = ixml.encode("utf-8")
            if len(doc) % 2:
                doc += b"\n"                                   # trailing whitespace is legal XML
            f.write(_chunk(b"iXML", doc))
        f.write(b"data")
        self._data_size_pos = f.tell()
        f.write(struct.pack("<I", 0))
        self._data_start = f.tell()

    # -- format -------------------------------------------------------------
    def _fmt_payload(self) -> bytes:
        extensible = self.channels > 2 or self.bit_depth in (24, 32)
        tag = WAVE_FORMAT_IEEE_FLOAT if self.is_float else WAVE_FORMAT_PCM
        byte_rate = self.sr * self.block_align
        if not extensible:
            return struct.pack("<HHIIHH", tag, self.channels, self.sr, byte_rate, self.block_align, self.bit_depth)
        return struct.pack("<HHIIHHHHI", WAVE_FORMAT_EXTENSIBLE, self.channels, self.sr, byte_rate,
                           self.block_align, self.bit_depth, 22, self.bit_depth, self.channel_mask) + \
            struct.pack("<I", tag) + _GUID_TAIL

    def _encode(self, x: np.ndarray) -> bytes:
        inter = np.ascontiguousarray(x.T)                       # (n, C) interleaved
        if self.is_float:
            return inter.astype("<f4").tobytes()
        if self.bit_depth == 16:
            q = np.clip(np.round(inter * 32768.0), -32768, 32767).astype("<i2")
            return q.tobytes()
        q = np.clip(np.round(inter * 8388608.0), -8388608, 8388607).astype("<i4")
        return np.ascontiguousarray(q.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3]).tobytes()

    # -- streaming ------------------------------------------------------------
    def write(self, x: np.ndarray) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[0] != self.channels:
            if a.shape[1] == self.channels and a.shape[0] != self.channels:
                a = a.T
            else:
                raise ValueError(f"expected {self.channels} channels, got shape {a.shape}")
        if a.shape[1] == 0:
            return
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        data = self._encode(a)
        self._f.write(data)
        self._data_bytes += len(data)
        self.frames += a.shape[1]

    def set_loudness(self, integrated: Optional[float] = None, lra: Optional[float] = None,
                     true_peak: Optional[float] = None, max_momentary: Optional[float] = None,
                     max_short_term: Optional[float] = None) -> None:
        """Loudness values to store in bext (patched in place on close). Accepts a meter summary too."""
        self._loudness = dict(integrated=integrated, lra=lra, true_peak=true_peak,
                              max_momentary=max_momentary, max_short_term=max_short_term)

    def set_loudness_from_meter(self, meter) -> None:
        self.set_loudness(meter.integrated(), meter.lra(), meter.true_peak_db(),
                          meter.max_momentary(), meter.max_short_term())

    def flush(self) -> None:
        """Write the current sizes into the header (crash safety for long renders)."""
        self._f.flush()
        pos = self._f.tell()
        riff = self._data_start - 8 + self._data_bytes
        if riff <= 0xFFFFFFFF:
            self._f.seek(4)
            self._f.write(struct.pack("<I", riff))
            self._f.seek(self._data_size_pos)
            self._f.write(struct.pack("<I", min(self._data_bytes, 0xFFFFFFFF)))
        self._f.seek(pos)

    def close(self) -> None:
        if self._closed:
            return
        f = self._f
        if self._data_bytes % 2:
            f.write(b"\0")
        end = f.tell()
        riff_size = end - 8
        if riff_size > self.rf64_threshold:
            f.seek(0)
            f.write((b"BW64" if self.bw64 else b"RF64") + struct.pack("<I", 0xFFFFFFFF))
            f.seek(self._junk_pos)
            f.write(b"ds64" + struct.pack("<I", 28) +
                    struct.pack("<QQQI", riff_size, self._data_bytes, self.frames, 0))
            f.seek(self._data_size_pos)
            f.write(struct.pack("<I", 0xFFFFFFFF))
            if self._fact_pos is not None:
                f.seek(self._fact_pos)
                f.write(struct.pack("<I", 0xFFFFFFFF))
        else:
            f.seek(4)
            f.write(struct.pack("<I", riff_size))
            f.seek(self._data_size_pos)
            f.write(struct.pack("<I", self._data_bytes))
            if self._fact_pos is not None:
                f.seek(self._fact_pos)
                f.write(struct.pack("<I", self.frames & 0xFFFFFFFF))
        if self._bext_loudness_pos is not None and self._loudness is not None:
            L = self._loudness
            f.seek(self._bext_loudness_pos)
            f.write(struct.pack("<5h", _loudness_x100(L["integrated"]), _loudness_x100(L["lra"]),
                                _loudness_x100(L["true_peak"]), _loudness_x100(L["max_momentary"]),
                                _loudness_x100(L["max_short_term"])))
        f.close()
        self._closed = True

    @property
    def duration_s(self) -> float:
        return self.frames / self.sr

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass


class FlacWriter:
    """FLAC output through python-soundfile (libsndfile); 16- or 24-bit, up to 8 channels."""

    def __init__(self, path: str, sr: int, channels: int, bit_depth: int = 24, metadata: Optional[dict] = None):
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            raise ImportError("FLAC export needs python-soundfile: pip install soundfile") from exc
        if int(channels) > 8:
            raise ValueError("FLAC supports at most 8 channels; use WAV/RF64 for this layout")
        self.path = str(path)
        self.sr = int(sr)
        self.channels = int(channels)
        self.bit_depth = 16 if int(bit_depth) == 16 else 24
        self.frames = 0
        self._f = sf.SoundFile(self.path, "w", samplerate=self.sr, channels=self.channels, format="FLAC",
                               subtype="PCM_16" if self.bit_depth == 16 else "PCM_24")
        md = metadata or {}
        ucs = md.get("ucs") or {}
        comment = md.get("comment") or ucs.get("Description") or ""
        scene = md.get("scene")
        if scene is not None:
            # The full scene travels in the Vorbis COMMENT so a FLAC render is reproducible too.
            if hasattr(scene, "to_json"):
                scene_json = scene.to_json(indent=None)
            elif isinstance(scene, dict):
                scene_json = json.dumps(scene)
            else:
                scene_json = str(scene)
            comment =(comment + "\n" if comment else "") + "RAINGEN_SCENE=" + scene_json
        tags = {
            "title": md.get("title") or ucs.get("FXName"),
            "artist": md.get("artist") or ucs.get("Designer"),
            "album": md.get("album") or ucs.get("Library"),
            "comment": comment,
            "software": "raingen",
            "genre": ucs.get("Category"),
        }
        for k, v in tags.items():
            if v:
                try:
                    setattr(self._f, k, str(v))
                except Exception:
                    pass
        self._closed = False

    def write(self, x: np.ndarray) -> None:
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] == 0:
            return
        self._f.write(np.ascontiguousarray(np.clip(a, -1.0, 1.0).T))
        self.frames += a.shape[1]

    def set_loudness(self, *args, **kw) -> None:
        pass

    def set_loudness_from_meter(self, meter) -> None:
        pass

    def close(self) -> None:
        if not self._closed:
            self._f.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _target_ext(path: str) -> str:
    """Extension of the final file, ignoring a temporary suffix ('x.flac.part' -> '.flac')."""
    p = str(path).lower()
    for tmp in (".part", ".tmp", ".partial"):
        if p.endswith(tmp):
            p = p[: -len(tmp)]
    dot = p.rfind(".")
    return p[dot:] if dot >= 0 else ""


def open_writer(path: str, sr: int, channels: int, bit_depth: int = 24, metadata: Optional[dict] = None):
    """Open a streaming writer chosen by extension (.flac -> FLAC, else BWF/RF64 WAV).

    A temporary suffix (.part/.tmp) is ignored when choosing, so 'x.flac.part' is FLAC.

    metadata keys (all optional): bext (dict), ixml (str), ucs (dict), scene (Scene|dict|str),
    layout (str), channel_mask (int), channel_names (list), bw64 (bool), rf64 (bool: force the
    RF64/ds64 header even for a small file), fps, start_tc, title/artist/comment (FLAC).
    """
    md = dict(metadata or {})
    if _target_ext(path) == ".flac":
        return FlacWriter(path, sr, channels, bit_depth, md)
    mask = md.get("channel_mask")
    if mask is None:
        mask = layout_channel_mask(md.get("layout"), channels)
    ixml = md.get("ixml")
    if ixml is None and (md.get("ucs") or md.get("scene") is not None):
        ixml = build_ixml(md.get("ucs"), md.get("scene"), md.get("channel_names"),
                          fps=md.get("fps"), sr=sr, extra=md.get("ixml_extra"),
                          drop_frame=";" in str(md.get("start_tc") or ""))
    bext = md.get("bext")
    if bext is None and (md.get("ucs") or md.get("scene") is not None or md.get("start_tc")):
        ucs = md.get("ucs") or {}
        scene = md.get("scene")
        if isinstance(scene, dict):
            seed = scene.get("seed")
        else:
            seed = getattr(scene, "seed", None)
        from .. import __version__ as engine_version
        bext = {
            "description": ucs.get("Description") or ucs.get("FXName", ""),
            "originator": "raingen",
            "originator_reference": (f"RGN{int(seed):016X}" if seed is not None else "RGN")[:32],
            "coding_history_text": f"raingen {engine_version}" + (f" seed=0x{int(seed):X}" if seed is not None else ""),
        }
        if md.get("start_tc"):
            bext["start_tc"] = md["start_tc"]
            bext["fps"] = md.get("fps", 24)
    return StreamingWavWriter(path, sr, channels, bit_depth, bext=bext, ixml=ixml, channel_mask=mask,
                              rf64_threshold=0 if md.get("rf64") else RF64_THRESHOLD,
                              bw64=bool(md.get("bw64", False)))
