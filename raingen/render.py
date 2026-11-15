"""Offline renderer: Scene -> streamed, mastered, metadata-rich audio files.

Signal flow per block of BLOCK frames at SAMPLE_RATE (MASTER_PLAN 9.1):

    timeline.evaluate(scene, t) [+ Auto/Offset/Pinned overrides]
      -> World.step()                  WeatherState (DSD, gusts, wetness, storm cells)
      -> BlockContext -> every Generator.render()       list[Emission]
      -> Encoder per stem              3rd-order AmbiX (ACN/SN3D) bus per stem
      -> ShelterTransfer               envelope transmission loss on exterior stems
      -> Room send                     SH-domain FDN reverb of the enclosure (added to 'interior')
      -> sum stems -> Decoder          stereo | binaural | 5.1 | 7.1.4 | foa | hoa3 | mono
      -> CalmGovernor                  wellness modes only ("Keep it soothing", 8.2)
      -> static gain -> TruePeakLimiter (>= 8x true-peak detection, destination ceiling)
      -> [rational polyphase resampler when the output rate is not 48 kHz]
      -> BS.1770-5 meter -> TPDF dither -> streaming BWF/RF64 or FLAC writer

Loudness normalisation is two-pass without temporary files (9.4): every layer is
counter-based and seeded, so pass 1 renders the programme once and only measures its
integrated loudness (BS.1770-5 gating, ITU-R BS.1770-5 / EBU R128), and pass 2 renders it
again bit-identically and applies the exact gain target - measured before the limiter. The
governor sits before the gain so its level-dependent decisions are the same in both passes.
Nothing is ever held in memory beyond one block.

Latency of the decoder, governor and limiter is removed per output chain by discarding
their first `latency` samples and rendering on past the end, so sample 0 of every file is
scene time 0 and captions/markers line up with the audio exactly. The scene (JSON), seed
and engine version are embedded in every file (bext CodingHistory + iXML <USER>, or the
FLAC Vorbis comment), so any render can be reproduced.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from . import __version__
from .core.constants import AMBI_ORDER, BLOCK, SAMPLE_RATE, STEMS
from .core.rng import RngFactory
from .core.types import BlockContext, EventMarker, WeatherState
from .scene.schema import Scene

log = logging.getLogger("raingen.render")

__all__ = [
    "BlockRenderer", "BlockOutput", "Renderer", "RenderReport", "RenderCancelled", "StreamingResampler",
    "render", "render_array", "thunder_one_shots", "resolve_layout", "stem_gains", "STEM_ALIASES",
    "SAFETY_CEILING_DBTP", "CAPTION_FORMATS",
]

SAFETY_CEILING_DBTP = -0.3          # integer exports of un-normalised renders are limited here instead of clipped
CAPTION_FORMATS = ("srt", "vtt", "csv", "edl")

# Mix keys accepted in `mix=` (scene-file style: rain, drips, nature ...) -> engine stems.
STEM_ALIASES: dict[str, tuple[str, ...]] = {
    "rain": ("rain_near", "rain_far"),
    "near": ("rain_near",),
    "far": ("rain_far",),
    "surfaces": ("rain_near",),
    "drips": ("aftermath",),
    "runoff": ("aftermath",),
    "gutters": ("aftermath",),
    "nature": ("ambience",),
    "biophony": ("ambience",),
    "traffic": ("ambience",),
    "foliage": ("wind",),
    "shelter": ("interior",),
}


class RenderCancelled(RuntimeError):
    """Raised when a progress callback returns False."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _db_value(v: Any) -> Optional[float]:
    """A mix entry -> gain in dB (None = muted)."""
    if v is None or v is False:
        return None
    if v is True:
        return 0.0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("off", "mute", "muted", "none", ""):
            return None
        if s in ("on", "unity"):
            return 0.0
        return float(s.replace("db", "").strip())
    return float(v)


def stem_gains(mix: Optional[dict]) -> dict[str, float]:
    """Linear gain per engine stem from a mix dict ({'rain': -3, 'thunder': 'off', 'wind': 2.5} ...)."""
    gains = {s: 1.0 for s in STEMS}
    for key, val in (mix or {}).items():
        k = str(key).strip().lower()
        targets = STEM_ALIASES.get(k, (k,) if k in gains else ())
        if not targets:
            raise ValueError(f"unknown mix key {key!r}; stems are {', '.join(STEMS)} "
                             f"(aliases: {', '.join(sorted(STEM_ALIASES))})")
        db = _db_value(val)
        g = 0.0 if db is None else 10.0 ** (db / 20.0)
        for t in targets:
            gains[t] = g
    return gains


def resolve_layout(scene: Scene, layout: Optional[str], destination: Optional[dict] = None) -> str:
    """Explicit layout > the scene's listener.render (when not plain stereo) > destination hint > stereo."""
    from .spatial.decoders import normalize_layout

    if layout:
        return normalize_layout(layout)
    lr = normalize_layout(getattr(scene.listener, "render", None) or "stereo")
    if lr != "stereo":
        return lr
    hint = (destination or {}).get("layout_hint")
    return normalize_layout(hint) if hint else "stereo"


def _fmt_duration(s: float) -> str:
    s = max(0.0, float(s))
    h, rem = divmod(int(round(s)), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _fmt_db(v: Optional[float], unit: str) -> str:
    if v is None:
        return "n/a"
    if not math.isfinite(v):
        return f"-inf {unit}"
    return f"{v:.1f} {unit}"


# ---------------------------------------------------------------------------
# streaming rational resampler
# ---------------------------------------------------------------------------
class StreamingResampler:
    """Rational L/M polyphase resampler with carried state and no added delay.

    A Kaiser-windowed sinc low-pass (cut-off `rolloff` x the lower Nyquist, `taps_per_phase`
    taps per polyphase branch, unity passband gain) is evaluated only at the output instants
    (Crochiere & Rabiner, *Multirate Digital Signal Processing*, 1983): output k sits at
    upsampled index u = kM + D, where D is the filter's centre, so the output is aligned with
    the input. History is carried between blocks, so blockwise output equals one-shot output.
    """

    def __init__(self, sr_in: int, sr_out: int, channels: int, taps_per_phase: int = 32,
                 beta: float = 8.6, rolloff: float = 0.92):
        from scipy.signal.windows import kaiser

        g = math.gcd(int(sr_in), int(sr_out))
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.L = self.sr_out // g
        self.M = self.sr_in // g
        self.channels = int(channels)
        self.T = max(4, int(taps_per_phase))
        N = self.L * self.T
        fc = float(rolloff) * 0.5 / max(self.L, self.M)            # cycles per upsampled sample
        n = np.arange(N) - (N - 1) / 2.0
        h = 2.0 * fc * np.sinc(2.0 * fc * n) * kaiser(N, beta)
        h *= self.L / np.sum(h)                                      # each branch sums to ~1
        self.poly = np.ascontiguousarray(h.reshape(self.T, self.L).T)   # poly[p, j] = h[j L + p]
        self.D = (N - 1) // 2
        self._taps = np.arange(self.T)
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros((self.channels, self.T - 1))
        self._n_in = 0
        self._k = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mono = x.ndim == 1
        x2 = x[None, :] if mono else x
        n = x2.shape[1]
        buf = np.concatenate([self._hist, x2], axis=1)
        end = self._n_in + n
        k_last = (end * self.L - 1 - self.D) // self.M
        if k_last < self._k:
            y = np.zeros((x2.shape[0], 0))
        else:
            ks = np.arange(self._k, k_last + 1, dtype=np.int64)
            u = ks * self.M + self.D
            q = u // self.L
            p = u - q * self.L
            base = self._n_in - (self.T - 1)                       # absolute input index of buf[:, 0]
            idx = (q - base)[:, None] - self._taps[None, :]
            y = np.einsum("kj,ckj->ck", self.poly[p], buf[:, idx])
            self._k = int(k_last) + 1
        self._hist = buf[:, buf.shape[1] - (self.T - 1):].copy()
        self._n_in = end
        return y[0] if mono else y


# ---------------------------------------------------------------------------
# one scene, block by block
# ---------------------------------------------------------------------------
@dataclass
class BlockOutput:
    index: int
    t0: float
    mix: np.ndarray                                 # (C, n) decoded, pre-master
    stems: Optional[dict] = None                    # stem -> (C, n) decoded (stems mode)
    markers: list = field(default_factory=list)
    weather: Optional[WeatherState] = None


class BlockRenderer:
    """Deterministic scene -> decoded output, one BLOCK at a time (shared by Renderer and Engine).

    Build it from a prepared Scene (seed, mode guard-rails and quality already set). Each
    `render_block(index)` resolves the scene at t = index * BLOCK / SAMPLE_RATE (timeline +
    overrides), steps the World, runs every generator, encodes per stem, applies the shelter
    transfer and room, and decodes the mix (and every stem when `stems=True`; decoders are
    linear, so the decoded stems sum to the decoded mix). A generator that raises is disabled
    with a logged warning (deterministically, so both loudness passes agree) unless `strict`.
    """

    def __init__(self, scene: Scene, layout: str = "stereo", *, stems: bool = False, mix: Optional[dict] = None,
                 strict: bool = False, sr: int = SAMPLE_RATE, block: int = BLOCK, precip_types=None):
        from .physics.layout import build_layout
        from .physics.world import World
        from .spatial.decoders import make_decoder, normalize_layout
        from .spatial.encoder import Encoder
        from .synth.places import ShelterTransfer
        from .synth.registry import make_generators

        if int(sr) != SAMPLE_RATE:
            raise ValueError(f"the synthesis core runs at {SAMPLE_RATE} Hz; resample the output instead")
        self.sr = int(sr)
        self.block = int(block)
        self.strict = bool(strict)
        self.base_scene = scene
        self._graph = None
        if getattr(scene, "overrides", None):
            from .scene.graph import ParameterGraph
            self._graph = ParameterGraph(scene.copy())
        self.scene = self.scene_at(0.0)            # structure (place, listener) as resolved at t = 0
        self.rngf = RngFactory(int(scene.seed))
        self.patches = build_layout(self.scene)
        self.world = World(self.scene, self.rngf, sr=self.sr, block=self.block)
        self.generators = list(make_generators(self.scene, self.patches, self.rngf, precip_types=precip_types))
        self.encoder = Encoder(AMBI_ORDER, self.sr)
        self.shelter = ShelterTransfer(self.scene, self.sr)
        self._window_open = float(getattr(self.scene.place, "window_open", 0.0) or 0.0)
        self.layout = normalize_layout(layout)
        hrtf = getattr(self.scene.listener, "hrtf", None) or "builtin"
        width = float(getattr(self.scene.listener, "width", 1.0) or 1.0)
        self.decoder = make_decoder(self.layout, AMBI_ORDER, self.sr, hrtf=hrtf, width=width)
        self.stem_names = tuple(STEMS)
        self.stem_decoders = {s: make_decoder(self.layout, AMBI_ORDER, self.sr, hrtf=hrtf, width=width)
                              for s in self.stem_names} if stems else {}
        self.stem_gain = stem_gains(mix)
        self.warnings: list[str] = []
        self._unknown_stems: set = set()
        self.last_weather: Optional[WeatherState] = None

    # -- scene resolution -------------------------------------------------
    def scene_at(self, t: float) -> Scene:
        sc = self.base_scene
        if sc.timeline:
            from .scene.timeline import evaluate
            sc = evaluate(sc, float(t))
        if self._graph is not None:
            sc = self._graph.apply(sc)
        return sc

    @property
    def channels(self) -> int:
        return int(self.decoder.channels)

    @property
    def latency(self) -> int:
        return int(getattr(self.decoder, "latency", 0) or 0)

    def _stem_for(self, stem: str) -> str:
        if stem in self.stem_gain:
            return stem
        if stem not in self._unknown_stems:
            self._unknown_stems.add(stem)
            self.warnings.append(f"unknown stem {stem!r} routed to 'ambience'")
            log.warning("unknown stem %r routed to 'ambience'", stem)
        return "ambience"

    # -- one block ----------------------------------------------------------
    def render_block(self, index: int, scene_at_t: Optional[Scene] = None) -> BlockOutput:
        n = self.block
        t0 = index * n / self.sr
        sc = scene_at_t if scene_at_t is not None else self.scene_at(t0)
        weather = self.world.step(index, t0, n, sc)
        self.last_weather = weather
        ctx = BlockContext(sr=self.sr, n=n, index=int(index), t0=float(t0), weather=weather, rng=self.rngf,
                           listener_height_m=float(sc.listener.height_m), shelter=str(sc.place.shelter),
                           markers=[])
        N = self.encoder.channels
        buses = {s: np.zeros((N, n)) for s in self.stem_names}
        for gen in list(self.generators):
            try:
                emissions = gen.render(ctx)
            except Exception as exc:
                if self.strict:
                    raise
                name = getattr(gen, "name", type(gen).__name__)
                msg = f"generator {name} disabled after error at t={t0:.2f}s: {exc!r}"
                log.exception(msg)
                self.warnings.append(msg)
                self.generators.remove(gen)
                continue
            for em in emissions or ():
                stem = self._stem_for(em.stem)
                buses[stem] += self.encoder.encode(em, n)

        for s, g in self.stem_gain.items():
            if g != 1.0:
                buses[s] *= g

        wo = float(getattr(sc.place, "window_open", 0.0) or 0.0)
        if wo != self._window_open:
            self._window_open = wo
            self.shelter.set_window_open(wo)
        if self.shelter.active:
            for s in self.stem_names:
                buses[s] = self.shelter.process(s, buses[s])
        if self.shelter.room is not None and self.shelter.room_wet > 0.0:
            send = np.zeros((N, n))
            for s in self.stem_names:
                send += buses[s]
            wet = self.shelter.room_process(send)
            buses["interior"] = buses["interior"] + wet[:N]

        total = np.zeros((N, n))
        for s in self.stem_names:
            total += buses[s]
        mix = self.decoder.process(total)
        stems = None
        if self.stem_decoders:
            stems = {s: self.stem_decoders[s].process(buses[s]) for s in self.stem_names}
        return BlockOutput(index=int(index), t0=float(t0), mix=mix, stems=stems, markers=list(ctx.markers),
                           weather=weather)


# ---------------------------------------------------------------------------
# output chain: governor -> gain -> limiter -> latency trim -> resample -> meter -> dither -> sink
# ---------------------------------------------------------------------------
class _OutputChain:
    def __init__(self, name: str, source: str, channels: int, frames_out: int, *, sr_in: int, sr_out: int,
                 skip: int = 0, governor=None, gain_db: float = 0.0, ceiling_dbtp: Optional[float] = None,
                 layout: str = "stereo", meter: bool = True, true_peak: bool = True,
                 dither_bits: Optional[int] = None, rngf: Optional[RngFactory] = None,
                 sink: Optional[Callable[[np.ndarray], None]] = None):
        from .master.dither import Ditherer
        from .master.limiter import TruePeakLimiter
        from .master.loudness import LoudnessMeter

        self.name = name
        self.source = source
        self.channels = int(channels)
        self.frames_out = int(frames_out)
        self.governor = governor
        self.gain = 10.0 ** (float(gain_db) / 20.0)
        self.limiter = TruePeakLimiter(sr_in, self.channels, ceiling_dbtp) if ceiling_dbtp is not None else None
        self.skip = int(skip) + int(getattr(governor, "latency", 0) or 0) + \
            (int(self.limiter.latency) if self.limiter is not None else 0)
        self.resampler = StreamingResampler(sr_in, sr_out, self.channels) if int(sr_out) != int(sr_in) else None
        self.meter = LoudnessMeter(sr_out, self.channels, layout=layout, true_peak=true_peak) if meter else None
        self.ditherer = None
        if dither_bits is not None and int(dither_bits) < 32:
            self.ditherer = Ditherer(int(dither_bits), rngf if rngf is not None else RngFactory(0),
                                     layer=f"master.dither.{name}")
        self.sink = sink
        self.written = 0
        self._blk = 0

    @property
    def done(self) -> bool:
        return self.written >= self.frames_out

    @property
    def fraction(self) -> float:
        return min(1.0, self.written / max(self.frames_out, 1))

    def push(self, x: np.ndarray) -> None:
        if self.done:
            return
        y = x
        if self.governor is not None:
            y = self.governor.process(y)
        if self.gain != 1.0:
            y = y * self.gain
        if self.limiter is not None:
            y = self.limiter.process(y)
        if self.skip > 0:
            drop = min(self.skip, y.shape[1])
            y = y[:, drop:]
            self.skip -= drop
            if y.shape[1] == 0:
                return
        if self.resampler is not None:
            y = self.resampler.process(y)
        y = y[:, : self.frames_out - self.written]
        if y.shape[1] == 0:
            return
        if self.meter is not None:
            self.meter.process(y)
        if self.ditherer is not None:
            y = self.ditherer.process(y, self._blk)
        self._blk += 1
        if self.sink is not None:
            self.sink(y)
        self.written += y.shape[1]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
class RenderReport(dict):
    """The render summary (a plain dict) with a readable str()."""

    def __str__(self) -> str:
        lines = []
        target = self.get("path") or "(memory)"
        lines.append(f"Rendered {self.get('scene', 'scene')!r} -> {target}")
        dest = self.get("destination", "raw")
        tgt = self.get("target_lufs")
        dest_txt = dest + (f" ({tgt:g} LUFS" if tgt is not None else " (not normalised")
        ceil = self.get("ceiling_dbtp")
        dest_txt += (f", {ceil:g} dBTP)" if ceil is not None else ")")
        lines.append(f"  {_fmt_duration(self.get('seconds', 0.0))} · {self.get('layout')} · "
                     f"{self.get('sample_rate', 0) / 1000:g} kHz/{self.get('bit_depth')}-bit · {dest_txt}")
        lines.append(f"  Loudness {_fmt_db(self.get('integrated_lufs'), 'LUFS')} integrated · true peak "
                     f"{_fmt_db(self.get('true_peak_dbtp'), 'dBTP')} · LRA {_fmt_db(self.get('lra_lu'), 'LU')} · "
                     f"gain {self.get('gain_db', 0.0):+.1f} dB")
        kinds = self.get("marker_kinds") or {}
        ev = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        rt = self.get("render_time_s")
        speed = self.get("realtime_factor")
        timing = f" · rendered in {_fmt_duration(rt)}" if rt is not None else ""
        if speed:
            timing += f" ({speed:.1f}x real time)"
        lines.append(f"  Events: {self.get('markers', 0)} markers" + (f" ({ev})" if ev else "") + timing)
        if self.get("understood"):
            lines.append("  Understood: " + " · ".join(self["understood"]))
        if self.get("stems"):
            lines.append(f"  Stems: {len(self['stems'])} files in {os.path.dirname(target) or '.'}")
        if self.get("captions"):
            lines.append("  Captions: " + ", ".join(sorted(self["captions"])))
        for w in self.get("warnings") or ():
            lines.append(f"  Warning: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the offline renderer
# ---------------------------------------------------------------------------
class Renderer:
    """Render a Scene for `seconds` to files (or memory), mastered for a delivery destination.

    scene        Scene (or a dict in scene-file form)
    seconds      programme length
    layout       stereo|binaural|5.1|7.1|7.1.4|foa|hoa3|mono; None = the scene's listener.render,
                 else the destination's layout hint, else stereo
    destination  raingen.master.destinations key (youtube, spotify, sleep, broadcast_eu, post_stems,
                 ...); None = scene.render.destination
    stems        also write one file per stem (UCS names) next to the mix; None = scene.render.stems
    progress     callable(fraction 0..1, phase 'measure'|'render'); returning False cancels

    Keyword-only extras: sample_rate / bit_depth (default: the scene's render settings), seed
    (overrides scene.seed), mix ({'rain': -3, 'thunder': 'off', ...} per stem or alias, dB),
    target_lufs / ceiling_dbtp (override the destination), gain_db (fixed gain, no measurement),
    quality (mobile|game|studio|cinematic), mode (apply a wellness mode), honor_timer (let a
    sleep timer fade the render to silence; off by default for offline renders), strict
    (re-raise generator errors), start_tc / fps (bext TimeReference and caption timecode).
    """

    def __init__(self, scene, seconds: float, layout: Optional[str] = None, destination: Optional[str] = None,
                 stems: Optional[bool] = None, progress: Optional[Callable] = None, *,
                 sample_rate: Optional[int] = None, bit_depth: Optional[int] = None, seed: Optional[int] = None,
                 mix: Optional[dict] = None, target_lufs: Optional[float] = None,
                 ceiling_dbtp: Optional[float] = None, gain_db: Optional[float] = None,
                 quality: Optional[str] = None, mode: Optional[str] = None, honor_timer: bool = False,
                 strict: bool = False, start_tc: Optional[str] = None, fps: float = 24.0):
        from .master.destinations import get_destination
        from .wellness.modes import apply_mode, canonical_mode

        if isinstance(scene, dict):
            scene = Scene.from_dict(scene)
        if not isinstance(scene, Scene):
            raise TypeError("Renderer needs a Scene; use raingen.api.render() for presets, files or text")
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("seconds must be a positive number")

        sc = scene.copy()                                   # frozen snapshot: later edits never leak in
        if seed is not None:
            sc.seed = int(seed)
        if quality:
            sc.render.quality = str(quality).lower()
        if mode is not None:
            sc = apply_mode(sc, mode)
        elif canonical_mode(sc.mode) != "none" and "guard" not in (sc.vibe or {}):
            sc = apply_mode(sc, sc.mode)

        self.dest = get_destination(destination or getattr(sc.render, "destination", None) or "raw")
        self.layout = resolve_layout(sc, layout, self.dest)
        self.stems = bool(sc.render.stems if stems is None else stems)
        self.sr_out = int(sample_rate or getattr(sc.render, "sample_rate", None) or self.dest["sample_rate"]
                          or SAMPLE_RATE)
        bits = int(bit_depth or getattr(sc.render, "bit_depth", None) or self.dest["bit_depth"] or 24)
        if bits not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24 or 32 (float)")
        self.bit_depth = bits
        sc.render.destination = self.dest["name"]
        sc.render.stems = self.stems
        sc.render.sample_rate = self.sr_out
        sc.render.bit_depth = self.bit_depth
        sc.listener.render = self.layout
        self.scene = sc

        self.seconds = seconds
        self.frames_out = int(round(seconds * self.sr_out))
        self.frames_in = int(round(seconds * SAMPLE_RATE))
        self.progress = progress
        self.mix = dict(mix) if mix else None
        stem_gains(self.mix)                                # validate early
        self.honor_timer = bool(honor_timer)
        self.strict = bool(strict)
        self.start_tc = start_tc
        self.fps = float(fps)

        if target_lufs is not None:
            self.target_lufs: Optional[float] = float(target_lufs)
        elif self.dest["normalize"] and self.dest["target_lufs"] is not None:
            self.target_lufs = float(self.dest["target_lufs"])
        else:
            self.target_lufs = None
        if ceiling_dbtp is not None:
            self.ceiling_dbtp: Optional[float] = float(ceiling_dbtp)
        elif self.dest["tp_dbtp"] is not None:
            self.ceiling_dbtp = float(self.dest["tp_dbtp"])
        elif self.bit_depth < 32:
            self.ceiling_dbtp = SAFETY_CEILING_DBTP
        else:
            self.ceiling_dbtp = None
        self._gain_db: Optional[float] = float(gain_db) if gain_db is not None else None
        self._measured: Optional[float] = None
        self.warnings: list[str] = []
        self.summary: Optional[RenderReport] = None

    # -- pieces ---------------------------------------------------------------
    @property
    def two_pass(self) -> bool:
        return self._gain_db is None and self.target_lufs is not None

    def _block_renderer(self, stems: bool) -> BlockRenderer:
        return BlockRenderer(self.scene, self.layout, stems=stems, mix=self.mix, strict=self.strict)

    def _governor(self):
        from .wellness.modes import CalmGovernor, canonical_mode

        mode = canonical_mode(self.scene.mode)
        if mode == "none":
            return None
        vibe = self.scene.vibe or {}
        if self.honor_timer:
            timer = vibe.get("timer_min")
            descent = vibe.get("descent_min")
            timer = float(timer) if timer is not None else None
            descent = float(descent) if descent is not None else None
        else:
            timer, descent = 0.0, 0.0
        return CalmGovernor(SAMPLE_RATE, mode, timer_min=timer, descent_min=descent, tingle=vibe.get("tingle"))

    def _report_progress(self, fraction: float, phase: str) -> None:
        if self.progress is None:
            return
        if self.progress(float(min(max(fraction, 0.0), 1.0)), phase) is False:
            raise RenderCancelled("render cancelled")

    def _loop(self, br: BlockRenderer, chains: list[_OutputChain], phase: str, f0: float, f1: float,
              collect: bool = False) -> list:
        markers: list = []
        lead = chains[0]
        slack = max(c.skip for c in chains) + 8 * BLOCK
        limit = (max(self.frames_in, int(math.ceil(self.frames_out * SAMPLE_RATE / self.sr_out))) + slack) // BLOCK + 64
        idx = 0
        while not all(c.done for c in chains):
            if idx > limit:
                raise RuntimeError("render loop did not converge (latency bookkeeping error)")
            out = br.render_block(idx)
            for c in chains:
                if not c.done:
                    c.push(out.mix if c.source == "mix" else out.stems[c.source])
            if collect and out.markers:
                markers.extend(out.markers)
            idx += 1
            if idx % 8 == 0 or lead.done:
                self._report_progress(f0 + (f1 - f0) * lead.fraction, phase)
        for w in br.warnings:
            if w not in self.warnings:
                self.warnings.append(w)
        return markers

    def _final_markers(self, markers: Iterable[EventMarker]) -> list[EventMarker]:
        keep = [m for m in markers if 0.0 <= float(m.t) < self.seconds]
        return sorted(keep, key=lambda m: (float(m.t), str(m.kind)))

    # -- pass 1 ---------------------------------------------------------------
    def measure(self) -> float:
        """Pass 1: integrated loudness (LUFS) of the un-limited programme after the governor."""
        if self._measured is not None:
            return self._measured
        br = self._block_renderer(False)
        chain = _OutputChain("measure", "mix", br.channels, self.frames_in, sr_in=SAMPLE_RATE, sr_out=SAMPLE_RATE,
                             skip=br.latency, governor=self._governor(), layout=self.layout, true_peak=False)
        self._loop(br, [chain], "measure", 0.0, 0.4 if self.two_pass else 1.0)
        self._measured = float(chain.meter.integrated())
        log.info("pass 1: %.2f LUFS integrated", self._measured)
        return self._measured

    @property
    def gain_db(self) -> float:
        """Static gain applied before the limiter (measured when the destination normalises)."""
        if self._gain_db is None:
            if self.target_lufs is None:
                self._gain_db = 0.0
            else:
                m = self.measure()
                self._gain_db = float(np.clip(self.target_lufs - m, -60.0, 60.0)) if math.isfinite(m) else 0.0
                if not math.isfinite(m):
                    self.warnings.append("programme is silent; loudness normalisation skipped")
        return self._gain_db

    # -- outputs --------------------------------------------------------------
    def _output_path(self, out) -> str:
        from .io.ucs import ucs_filename

        fmt = str(self.dest.get("format") or "wav").lower()
        if out is None:
            path = os.path.join(os.getcwd(), ucs_filename(self.scene, ext=fmt))
        else:
            path = os.path.expanduser(os.fspath(out))
            if os.path.isdir(path) or path.endswith(("/", os.sep)):
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, ucs_filename(self.scene, ext=fmt))
            elif not os.path.splitext(path)[1]:
                path = f"{path}.{fmt}"
        root, ext = os.path.splitext(path)
        if ext.lower() == ".flac":
            if self._channels_for_layout() > 8:
                self.warnings.append(f"FLAC holds at most 8 channels; writing {self.layout} as RF64 WAV")
                path = root + ".wav"
            elif self.bit_depth == 32:
                self.warnings.append("FLAC is integer-only; writing 24-bit")
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        return os.path.abspath(path)

    def _channels_for_layout(self) -> int:
        from .spatial.decoders import SPEAKER_LAYOUTS

        lay = self.layout
        if lay in ("stereo", "binaural"):
            return 2
        if lay == "mono":
            return 1
        if lay == "foa":
            return 4
        if lay.startswith("hoa"):
            try:
                return (int(lay[3:]) + 1) ** 2
            except ValueError:
                return 16
        return len(SPEAKER_LAYOUTS.get(lay, ())) or 2

    def _file_bits(self, path: str) -> int:
        if path.lower().endswith(".flac"):
            return 16 if self.bit_depth == 16 else 24
        return self.bit_depth

    def _open_writer(self, path: str, channels: int, names: list, mask: int, stem: Optional[str], gain_db: float):
        from .io.ucs import ucs_metadata
        from .io.wav import FlacWriter, StreamingWavWriter, build_ixml

        tmp = path + ".part"
        ucs = ucs_metadata(self.scene, stem)
        seed = int(self.scene.seed)
        history = (f"raingen {__version__} seed=0x{seed:X} layout={self.layout} destination={self.dest['name']} "
                   f"gain={gain_db:+.2f}dB" + (f" stem={stem}" if stem else ""))
        if path.lower().endswith(".flac"):
            md = {"ucs": ucs, "scene": self.scene, "title": ucs.get("FXName"),
                  "comment": f"{ucs.get('Description', '')}\n{history}".strip()}
            return FlacWriter(tmp, self.sr_out, channels, self._file_bits(path), md), tmp
        bext = {
            "description": str(ucs.get("Description", ""))[:256],
            "originator": "raingen",
            "originator_reference": f"RGN{seed & 0xFFFFFFFFFFFFFFFF:016X}"[:32],
            "coding_history_text": history,
        }
        if self.start_tc:
            bext["start_tc"] = self.start_tc
            bext["fps"] = self.fps
        extra = {"RAINGEN_LAYOUT": self.layout, "RAINGEN_DESTINATION": self.dest["name"],
                 "RAINGEN_GAIN_DB": f"{gain_db:.3f}", "RAINGEN_STEM": stem or "mix",
                 "RAINGEN_SECONDS": f"{self.seconds:.6f}"}
        ixml = build_ixml(ucs, self.scene, names, extra=extra, fps=self.fps if self.start_tc else None, sr=self.sr_out)
        return StreamingWavWriter(tmp, self.sr_out, channels, self._file_bits(path), bext=bext, ixml=ixml,
                                  channel_mask=mask), tmp

    def _write_captions(self, markers: list, base: str, captions) -> dict:
        from .io import captions as cap

        if captions is True:
            formats = ("srt", "csv")
        elif isinstance(captions, str):
            formats = tuple(f.strip().lower() for f in captions.replace(",", " ").split() if f.strip())
        else:
            formats = tuple(str(f).lower() for f in captions)
        written = {}
        for fmt in formats:
            if fmt not in CAPTION_FORMATS:
                raise ValueError(f"unknown caption format {fmt!r}; choose from {CAPTION_FORMATS}")
            p = f"{base}.{fmt}"
            if fmt == "srt":
                cap.write_srt(markers, p)
            elif fmt == "vtt":
                cap.write_vtt(markers, p)
            elif fmt == "csv":
                cap.write_csv(markers, p, fps=self.fps, start_tc=self.start_tc or "00:00:00:00")
            else:
                cap.write_edl(markers, p, fps=self.fps, start_tc=self.start_tc or "01:00:00:00")
            written[fmt] = os.path.abspath(p)
        return written

    def _summary(self, path: Optional[str], chain: _OutputChain, markers: list, wall: float,
                 stems: Optional[dict] = None, captions: Optional[dict] = None, channels: int = 2) -> RenderReport:
        meter = chain.meter
        kinds: dict[str, int] = {}
        for m in markers:
            kinds[str(m.kind)] = kinds.get(str(m.kind), 0) + 1
        rep = RenderReport(
            path=path,
            paths=[p for p in [path, *(stems or {}).values()] if p],
            stems=dict(stems or {}),
            captions=dict(captions or {}),
            scene=self.scene.name,
            seed=int(self.scene.seed),
            engine_version=__version__,
            seconds=self.seconds,
            frames=chain.written,
            sample_rate=self.sr_out,
            bit_depth=self.bit_depth,
            channels=int(channels),
            layout=self.layout,
            destination=self.dest["name"],
            target_lufs=self.target_lufs,
            ceiling_dbtp=self.ceiling_dbtp,
            measured_lufs=self._measured,
            gain_db=float(self.gain_db),
            integrated_lufs=float(meter.integrated()) if meter is not None else None,
            true_peak_dbtp=float(meter.true_peak_db()) if meter is not None else None,
            lra_lu=float(meter.lra()) if meter is not None else None,
            markers=len(markers),
            marker_kinds=kinds,
            mode=self.scene.mode,
            render_time_s=wall,
            realtime_factor=(self.seconds / wall) if wall > 0 else None,
            warnings=list(self.warnings),
        )
        self.summary = rep
        return rep

    # -- pass 2 ---------------------------------------------------------------
    def run(self, out=None, captions=False) -> RenderReport:
        """Render to `out` (file path or directory; None = a UCS-named file in the working dir)."""
        t_wall = time.perf_counter()
        gain_db = self.gain_db
        f0 = 0.4 if self.two_pass else 0.0
        mix_path = self._output_path(out)
        br = self._block_renderer(self.stems)
        C = br.channels
        names = list(getattr(br.decoder, "channel_names", []) or [])
        mask = int(getattr(br.decoder, "channel_mask", 0) or 0)
        ext = os.path.splitext(mix_path)[1]
        targets: list[tuple[str, str, str]] = [("mix", "mix", mix_path)]
        if self.stems:
            from .io.ucs import ucs_filename
            folder = os.path.dirname(mix_path)
            for s in br.stem_names:
                targets.append((s, s, os.path.join(folder, ucs_filename(self.scene, s, ext=ext))))
        rngf = RngFactory(int(self.scene.seed))
        writers: list = []
        tmps: list[str] = []
        chains: list[_OutputChain] = []
        try:
            for name, source, path in targets:
                w, tmp = self._open_writer(path, C, names, mask, None if name == "mix" else name, gain_db)
                writers.append(w)
                tmps.append(tmp)
                chains.append(_OutputChain(
                    name, source, C, self.frames_out, sr_in=SAMPLE_RATE, sr_out=self.sr_out, skip=br.latency,
                    governor=self._governor() if name == "mix" else None, gain_db=gain_db,
                    ceiling_dbtp=self.ceiling_dbtp, layout=self.layout, dither_bits=self._file_bits(path),
                    rngf=rngf, sink=w.write))
            markers = self._loop(br, chains, "render", f0, 1.0, collect=True)
            for chain, w, tmp, (_, _, path) in zip(chains, writers, tmps, targets):
                w.set_loudness_from_meter(chain.meter)
                w.close()
                os.replace(tmp, path)
        except BaseException:
            for w in writers:
                try:
                    w.close()
                except Exception:
                    pass
            for tmp in tmps:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

        markers = self._final_markers(markers)
        cap_paths = self._write_captions(markers, os.path.splitext(mix_path)[0], captions) if captions else {}
        stem_paths = {t[0]: t[2] for t in targets[1:]}
        rep = self._summary(mix_path, chains[0], markers, time.perf_counter() - t_wall, stem_paths, cap_paths, C)
        if self.progress is not None:
            self.progress(1.0, "done")
        return rep

    def stream(self, sink: Callable[[np.ndarray], None], dither: bool = False) -> tuple[RenderReport, list]:
        """Render the mix into `sink(block (C, n) float)` instead of a file. Returns (report, markers)."""
        t_wall = time.perf_counter()
        gain_db = self.gain_db
        br = self._block_renderer(False)
        chain = _OutputChain("mix", "mix", br.channels, self.frames_out, sr_in=SAMPLE_RATE, sr_out=self.sr_out,
                             skip=br.latency, governor=self._governor(), gain_db=gain_db,
                             ceiling_dbtp=self.ceiling_dbtp, layout=self.layout,
                             dither_bits=self.bit_depth if dither else None,
                             rngf=RngFactory(int(self.scene.seed)), sink=sink)
        markers = self._final_markers(self._loop(br, [chain], "render", 0.4 if self.two_pass else 0.0, 1.0,
                                                 collect=True))
        rep = self._summary(None, chain, markers, time.perf_counter() - t_wall, channels=br.channels)
        return rep, markers

    def to_array(self) -> tuple[np.ndarray, list]:
        """The mastered mix as one float array (C, n) plus its markers. Short renders only."""
        parts: list[np.ndarray] = []
        _, markers = self.stream(parts.append)
        C = self._channels_for_layout()
        y = np.concatenate(parts, axis=1) if parts else np.zeros((C, 0))
        return y, markers


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------
def render(scene, seconds: float, out=None, layout: Optional[str] = None, destination: Optional[str] = None,
           stems: Optional[bool] = None, captions=False, progress: Optional[Callable] = None, **kw) -> RenderReport:
    """Render `scene` for `seconds` to `out`; returns a RenderReport (a dict: path(s), LUFS, true peak,
    marker count ...). Extra keywords go to Renderer (sample_rate, bit_depth, seed, mix, quality ...).

    layout / destination / stems default to the scene's own settings (stereo, raw, no stems).
    captions: False, True (srt + csv), or formats such as ('srt', 'vtt', 'csv', 'edl').
    """
    r = Renderer(scene, seconds, layout=layout, destination=destination, stems=stems, progress=progress, **kw)
    return r.run(out, captions=captions)


def render_array(scene, seconds: float, layout: Optional[str] = None, destination: Optional[str] = None,
                 **kw) -> tuple[np.ndarray, list]:
    """Render to memory: (audio (C, n) float64, markers). Keep it short; files stream, arrays do not."""
    return Renderer(scene, seconds, layout=layout, destination=destination, stems=False, **kw).to_array()


def thunder_one_shots(count: int = 3, distance_km: float = 3.0, sample_rate: int = SAMPLE_RATE,
                      seed: Optional[int] = None, max_s: float = 10.0, layout: str = "stereo") -> list[np.ndarray]:
    """Synthesized thunder one-shots (C, n) peak-normalised to 0.9, cut at each 'thunder' marker.

    A dry, calm, open-meadow scene with a steady thunder cell at `distance_km` is rendered just
    long enough for `count` flashes at 12 per minute, and each roll is cut from its first
    arrival (the marker time) for up to `max_s` seconds with a short fade-out.
    """
    count = max(1, int(count))
    sc = Scene(name="Thunder one-shots", seed=int(seed) if seed is not None else 0x7D0D3E)
    sc.weather.precipitation.type = "none"
    sc.weather.precipitation.rate_mm_h = 0.0
    sc.weather.wind.speed_mps = 1.0
    sc.weather.storm_arc = "steady"
    th = sc.weather.thunder
    th.enabled = True
    th.distance_km = float(max(distance_km, 0.3))
    th.bearing_deg = 270.0
    th.approach_kmh = 0.0
    th.flashes_per_min = 12.0
    th.stroke_intensity = 0.7
    th.cg_fraction = 0.6
    sc.place.preset = "meadow"
    sc.place.surfaces = {"grass": 100.0}
    sc.place.vegetation = {"deciduous": 0, "conifer": 0, "distance_m": 30.0}
    sc.place.emitters = []
    sc.place.biophony = {"birds": "off", "insects": "off", "frogs": "off", "traffic": 0.0,
                         "region": "temperate_europe"}
    sc.render.quality = "mobile"
    flash_to_bang = th.distance_km / 0.343
    seconds = min(150.0, flash_to_bang + max_s + 7.5 * count + 5.0)
    y, markers = Renderer(sc, seconds, layout=layout, destination="raw", sample_rate=sample_rate,
                          bit_depth=32, ceiling_dbtp=None).to_array()
    starts = sorted(float(m.t) for m in markers if m.kind == "thunder")
    clips: list[np.ndarray] = []
    sr = int(sample_rate)
    fade = int(0.3 * sr)
    for i, t in enumerate(starts):
        s = max(0, int((t - 0.05) * sr))
        e = s + int(max_s * sr)
        if i + 1 < len(starts):
            e = min(e, max(s + int(1.5 * sr), int((starts[i + 1] - 0.05) * sr)))
        e = min(e, y.shape[1])
        if e - s < int(0.5 * sr):
            continue
        clip = y[:, s:e].copy()
        f = min(fade, clip.shape[1] // 4)
        if f > 0:
            clip[:, -f:] *= np.cos(np.linspace(0.0, 0.5 * math.pi, f)) ** 2
        peak = float(np.max(np.abs(clip)))
        if peak <= 1e-9:
            continue
        clips.append(clip * (0.9 / peak))
        if len(clips) >= count:
            break
    return clips
