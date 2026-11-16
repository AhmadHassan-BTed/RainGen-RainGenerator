"""raingen command line (MASTER_PLAN 3.3).

    raingen render "heavy rain on a car roof, wipers on" --minutes 30 --out car.flac --destination youtube
    raingen render my_tent.json --hours 10 --layout binaural --destination sleep
    raingen render tent-in-a-storm --minutes 5 --stems --captions
    raingen variations tent-in-a-storm --count 20 --seconds 90 --out-dir library/
    raingen presets --tag sleep
    raingen metar "KJFK 121755Z 21016G24KT 1SM +TSRA BR BKN015CB 18/17 A2990" --minutes 10 --out jfk.wav
    raingen live --place "Bergen, Norway" --play
    raingen play "rain on a tent, gusty"
    raingen manifest --format osc
    raingen lint-claims README.md site/index.html
    raingen scene "drizzle in a pine forest at dawn" --out forest.json

Scene arguments accept a preset name, a .json/.yaml/.wxscene file, a METAR or plain words
(several words need no quotes: `raingen render heavy rain on a tent --minutes 5`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Optional

from . import __version__

LAYOUTS = ("stereo", "binaural", "5.1", "7.1", "7.1.4", "foa", "hoa3", "mono")
MODES = ("none", "sleep", "focus", "calm", "relax", "asmr")
CAPTIONS = ("srt", "vtt", "csv", "edl")
MANIFEST_FORMATS = ("json-schema", "osc", "wwise", "fmod", "markdown", "json")


class CLIError(Exception):
    """A user-facing error (printed without a traceback)."""


# BLAS / OpenMP pools read these once, when numpy (or scipy) first loads them.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_blas_threads(n: int = 1) -> bool:
    """Ask the BLAS / OpenMP runtimes for `n` threads; only effective before numpy is imported.

    raingen renders many small per-block array operations, where a pooled BLAS spins its
    worker threads (a multi-threaded render used ~8x more CPU time than wall time) for no
    speed-up. Variables the user has set are left alone. Returns True when numpy was not yet
    loaded, i.e. when the setting takes effect.
    """
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(int(n)))
    return "numpy" not in sys.modules


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _seed(text: str) -> int:
    try:
        return int(str(text).replace("_", ""), 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"seed must be an integer (decimal or 0x hex), got {text!r}") from None


def destination_names() -> tuple[str, ...]:
    """The --destination choices: every profile in raingen.master.destinations.DESTINATIONS."""
    from .master.destinations import DESTINATIONS

    return tuple(DESTINATIONS)


def _destination(text: str) -> str:
    """A destination name or alias (ebu, r128, atsc, ott ...) -> its canonical key."""
    from .master import destinations as dst

    key = dst._key(text)
    if key not in dst.DESTINATIONS:
        raise argparse.ArgumentTypeError(f"unknown destination {text!r}; choose from {', '.join(dst.DESTINATIONS)}")
    return key


# METAR present-weather groups with an intensity sign ('-RA', '-SHRA', '-TSRAGR') look like
# options to argparse; they are protected before parsing and restored in _join().
_METAR_WX = re.compile(r"^-(VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?((DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|"
                       r"SQ|FC|SS|DS)+)$")
_PROTECT = "\u2212"                        # U+2212 MINUS SIGN stands in for '-' while parsing


def _protect_metar_groups(argv: list[str]) -> list[str]:
    return [_PROTECT + t[1:] if _METAR_WX.match(t) else t for t in argv]


def _join(words) -> str:
    if isinstance(words, (list, tuple)):
        return " ".join(("-" + str(w)[1:]) if str(w).startswith(_PROTECT) else str(w) for w in words)
    return str(words)


def _progress(quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None
    t0 = time.monotonic()
    state = {"last": -1}

    def cb(fraction: float, phase: str) -> None:
        pct = int(fraction * 1000)
        if pct == state["last"] and phase != "done":
            return
        state["last"] = pct
        width = 30
        fill = int(round(fraction * width))
        elapsed = time.monotonic() - t0
        eta = ""
        if 0.02 < fraction < 1.0:
            remain = elapsed * (1.0 - fraction) / fraction
            eta = f" ~{int(remain // 60)}:{int(remain % 60):02d} left"
        label = {"measure": "measuring", "render": "rendering", "done": "done"}.get(phase, phase)
        sys.stderr.write(f"\r{label:<10}[{'#' * fill}{'.' * (width - fill)}] {fraction * 100:5.1f}%{eta}   ")
        if phase == "done":
            sys.stderr.write("\n")
        sys.stderr.flush()

    return cb


def _add_duration(p: argparse.ArgumentParser, default_note: str = "default 60 s") -> None:
    g = p.add_argument_group("length (summed)")
    g.add_argument("--seconds", "-s", type=float, help=f"length in seconds ({default_note})")
    g.add_argument("--minutes", "-m", type=float, help="length in minutes")
    g.add_argument("--hours", type=float, help="length in hours")
    g.add_argument("--duration", help="length as text: '90 s', '5 min', '1.5 h', '01:30:00'")


def _add_render_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", "-o", help="output file (.wav/.flac) or directory (UCS-named file)")
    p.add_argument("--layout", "-l", choices=LAYOUTS, help="output layout (default: the scene's, else stereo)")
    p.add_argument("--destination", "-d", type=_destination, metavar="DEST",
                   help="delivery profile: loudness, true peak, format (default: raw); one of "
                        + ", ".join(destination_names()) + " (aliases such as ebu, atsc, ott work too)")
    p.add_argument("--seed", type=_seed, help="random seed (overrides the scene's)")
    p.add_argument("--stems", action="store_true", help="also write one file per stem")
    p.add_argument("--captions", nargs="*", choices=CAPTIONS, metavar="FMT",
                   help="write captions/cue sheets next to the output (default srt + csv)")
    p.add_argument("--mode", choices=MODES, help="apply a wellness mode before rendering")
    p.add_argument("--quality", choices=("mobile", "game", "studio", "cinematic"), help="level of detail")
    p.add_argument("--bit-depth", type=int, choices=(16, 24, 32), help="16/24-bit PCM or 32-bit float")
    p.add_argument("--sample-rate", type=int, help="output sample rate (resampled from 48 kHz)")
    p.add_argument("--timecode", help="start timecode HH:MM:SS:FF for bext and cue sheets")
    p.add_argument("--fps", type=float, default=24.0, help="timecode frame rate (default 24)")
    p.add_argument("--honor-timer", action="store_true", help="let a sleep timer fade the render to silence")
    p.add_argument("--strict", action="store_true", help="fail when free text names nothing known")


def _render_kwargs(a: argparse.Namespace) -> dict:
    kw = dict(out=a.out, layout=a.layout, destination=a.destination, seed=a.seed, stems=bool(a.stems),
              mode=a.mode, quality=a.quality, bit_depth=a.bit_depth, sample_rate=a.sample_rate,
              start_tc=a.timecode, fps=a.fps, honor_timer=bool(a.honor_timer), strict=bool(a.strict))
    if a.captions is not None:
        kw["captions"] = tuple(a.captions) if a.captions else True
    return {k: v for k, v in kw.items() if v is not None}


def _length(a: argparse.Namespace, default: float = 60.0) -> float:
    from .api import duration_seconds
    return duration_seconds(a.seconds, a.minutes, a.hours, a.duration, default=default)


def _print_report(rep, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(rep), indent=2, default=str))
    else:
        print(str(rep))


def _write_scene(scene, path: Optional[str]) -> None:
    if path:
        scene.save(path)
        print(f"Saved {os.path.abspath(path)}")
    else:
        print(scene.to_json())


def _has_length(a: argparse.Namespace) -> bool:
    return any(getattr(a, k, None) is not None for k in ("seconds", "minutes", "hours", "duration"))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_render(a: argparse.Namespace) -> int:
    from .api import render_file

    what = _join(a.what)
    rep = render_file(what, seconds=_length(a), progress=_progress(a.quiet or a.json), **_render_kwargs(a))
    _print_report(rep, a.json)
    return 0


def cmd_variations(a: argparse.Namespace) -> int:
    from .api import load_scene, render_file as render, variations
    from .io.ucs import ucs_filename

    base = load_scene(_join(a.what))
    vs = variations(base, a.count, jitter=a.jitter)
    out_dir = a.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    if a.scenes_only:
        for i, v in enumerate(vs, start=1):
            p = os.path.join(out_dir, f"{ucs_filename(v)}_v{i:02d}.json")
            v.save(p)
            print(p)
        return 0
    secs = _length(a, default=90.0)
    kw = _render_kwargs(a)
    kw.pop("out", None)
    kw.pop("seed", None)
    from .master.destinations import get_destination
    ext = "." + str(get_destination(a.destination or base.render.destination).get("format") or "wav")
    reports = []
    for i, v in enumerate(vs, start=1):
        path = os.path.join(out_dir, f"{ucs_filename(v)}_v{i:02d}{ext}")
        if not a.quiet:
            print(f"[{i}/{len(vs)}] {os.path.basename(path)}", file=sys.stderr)
        rep = render(v, seconds=secs, out=path, progress=_progress(a.quiet or a.json), **kw)
        reports.append(dict(rep))
        if not a.json:
            print(str(rep))
    if a.json:
        print(json.dumps(reports, indent=2, default=str))
    return 0


def cmd_presets(a: argparse.Namespace) -> int:
    from .api import list_presets

    cards = list_presets(a.tag)
    if a.json:
        print(json.dumps(cards, indent=2))
        return 0
    if not cards:
        print(f"No presets tagged {a.tag!r}.")
        return 1
    w = max(len(c["name"]) for c in cards)
    for c in cards:
        tags = " ".join(t.split(":", 1)[-1] for t in c["tags"])
        print(f"{c['name']:<{w}}  {c['title']}")
        if a.verbose:
            print(f"{'':<{w}}  {c['description']}")
            print(f"{'':<{w}}  tags: {tags}" + (f" · mode: {c['mode']}" if c.get("mode") not in (None, "none") else ""))
    return 0


def cmd_metar(a: argparse.Namespace) -> int:
    from .api import play, render_file as render, scene_from_metar

    scene = scene_from_metar(_join(a.raw), utc_offset_h=a.utc_offset)
    if a.play:
        play(scene, seconds=_length(a) if _has_length(a) else None, layout=a.layout)
        return 0
    if a.out or _has_length(a):
        rep = render(scene, seconds=_length(a), progress=_progress(a.quiet or a.json), **_render_kwargs(a))
        _print_report(rep, a.json)
        return 0
    print(scene.to_json())
    return 0


def cmd_live(a: argparse.Namespace) -> int:
    from .api import play, render_file as render, scene_from_live

    if a.station:
        location, provider = a.station, "metar"
    elif a.place:
        location, provider = a.place, a.provider
    elif a.lat is not None and a.lon is not None:
        location, provider = (a.lat, a.lon), a.provider
    else:
        raise CLIError("give --place NAME, --lat/--lon or --station ICAO")
    scene = scene_from_live(location, provider=provider)
    if a.play:
        play(scene, seconds=_length(a) if _has_length(a) else None, layout=a.layout)
        return 0
    if a.out or _has_length(a):
        rep = render(scene, seconds=_length(a), progress=_progress(a.quiet or a.json), **_render_kwargs(a))
        _print_report(rep, a.json)
        return 0
    from .api import describe
    print(describe(scene))
    if a.json:
        print(scene.to_json())
    return 0


def cmd_play(a: argparse.Namespace) -> int:
    from .api import Engine, load_scene

    scene = load_scene(_join(a.what))
    if a.mode:
        from .wellness.modes import apply_mode
        scene = apply_mode(scene, a.mode)

    def show(m) -> None:
        if a.quiet:
            return
        from .io.captions import caption_for
        print(f"{m.t:8.2f}s  {caption_for(m)}", flush=True)

    osc = a.osc if a.osc is None else (a.osc_host, a.osc)
    eng = Engine(scene, layout=a.layout or "stereo", device=a.device, quality=a.quality or "game",
                 volume_db=a.volume, on_marker=show, osc=osc)
    if not a.quiet:
        print(f"Playing {scene.name!r} ({eng.layout}). Ctrl+C to stop.", file=sys.stderr)
        if osc is not None:
            print(f"OSC control on {osc[0]}:{osc[1]} (e.g. /rg/weather/precipitation/rate_mm_h 20)",
                  file=sys.stderr)
    eng.run(_length(a) if _has_length(a) else None)
    if not a.quiet and eng.underruns:
        print(f"{eng.underruns} buffer underruns (try --quality mobile)", file=sys.stderr)
    return 0


def cmd_manifest(a: argparse.Namespace) -> int:
    from .api import manifest

    data = manifest(a.format)
    text = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        print(f"Saved {os.path.abspath(a.out)}", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_lint_claims(a: argparse.Namespace) -> int:
    from .wellness.claims import main as lint_main

    missing = [f for f in a.files if not os.path.isfile(f)]
    if missing:
        raise CLIError("not found: " + ", ".join(missing))
    code = lint_main(list(a.files))
    if code == 0 and not a.quiet:
        print("No claim errors found.", file=sys.stderr)
    return code


def cmd_scene(a: argparse.Namespace) -> int:
    from .api import describe, load_scene

    scene = load_scene(_join(a.what))
    if a.seed is not None:
        scene.seed = a.seed
    if a.mode:
        from .wellness.modes import apply_mode
        scene = apply_mode(scene, a.mode)
    if a.describe:
        print(describe(scene))
        if not a.out:
            return 0
    _write_scene(scene, a.out)
    return 0


def cmd_destinations(a: argparse.Namespace) -> int:
    from .api import destinations

    rows = destinations()
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    for d in rows:
        tgt = f"{d['target_lufs']:g} LUFS" if d["target_lufs"] is not None else "not normalised"
        tp = f"{d['tp_dbtp']:g} dBTP" if d["tp_dbtp"] is not None else "-"
        print(f"{d['name']:<22} {tgt:<15} {tp:<10} {d['format']}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="raingen", description="Physically based weather and environment sound engine.",
                                epilog="Scenes: a preset name, a .json/.yaml file, a METAR, or plain words.")
    p.add_argument("--version", action="version", version=f"raingen {__version__}")
    p.add_argument("--debug", action="store_true", help="show tracebacks and debug logging")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    r = sub.add_parser("render", help="render a scene to a file")
    r.add_argument("what", nargs="+", help="preset, scene file, METAR or plain words")
    _add_duration(r)
    _add_render_opts(r)
    r.add_argument("--quiet", "-q", action="store_true", help="no progress bar")
    r.add_argument("--json", action="store_true", help="print the report as JSON")
    r.set_defaults(func=cmd_render)

    v = sub.add_parser("variations", help="render (or save) N derived variations of a scene")
    v.add_argument("what", nargs="+", help="preset, scene file, METAR or plain words")
    v.add_argument("--count", "-n", type=int, default=10, help="number of variations (default 10)")
    v.add_argument("--jitter", type=float, default=1.0, help="size of the parameter jitter (0 = seeds only)")
    v.add_argument("--out-dir", help="output directory (default: current)")
    v.add_argument("--scenes-only", action="store_true", help="write scene JSON files instead of audio")
    _add_duration(v, "default 90 s")
    _add_render_opts(v)
    v.add_argument("--quiet", "-q", action="store_true")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_variations)

    ps = sub.add_parser("presets", help="list scene presets")
    ps.add_argument("--tag", "-t", help="filter by tag (sleep, film, forest, thunderstorm, ...)")
    ps.add_argument("--verbose", "-v", action="store_true", help="show descriptions and tags")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_presets)

    m = sub.add_parser("metar", help="scene from a METAR report: print, render or play it")
    m.add_argument("raw", nargs="+", help="the METAR text")
    m.add_argument("--utc-offset", type=float, help="local time offset from UTC in hours")
    m.add_argument("--play", action="store_true", help="play instead of rendering")
    _add_duration(m)
    _add_render_opts(m)
    m.add_argument("--quiet", "-q", action="store_true")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_metar)

    lv = sub.add_parser("live", help="scene from the current weather somewhere (network)")
    loc = lv.add_argument_group("location (coordinates are rounded to 0.1 deg before any request)")
    loc.add_argument("--place", help="place name, e.g. 'Bergen, Norway'")
    loc.add_argument("--lat", type=float)
    loc.add_argument("--lon", type=float)
    loc.add_argument("--station", help="ICAO station for METAR, e.g. EGLL")
    lv.add_argument("--provider", default="open-meteo", choices=("open-meteo", "metar"))
    lv.add_argument("--play", action="store_true", help="play it now")
    _add_duration(lv)
    _add_render_opts(lv)
    lv.add_argument("--quiet", "-q", action="store_true")
    lv.add_argument("--json", action="store_true")
    lv.set_defaults(func=cmd_live)

    pl = sub.add_parser("play", help="play a scene in real time (needs sounddevice)")
    pl.add_argument("what", nargs="+", help="preset, scene file, METAR or plain words")
    _add_duration(pl, "default: until Ctrl+C")
    pl.add_argument("--layout", "-l", choices=LAYOUTS, help="output layout (default stereo)")
    pl.add_argument("--device", help="output device (index or name)")
    pl.add_argument("--mode", choices=MODES)
    pl.add_argument("--quality", choices=("mobile", "game", "studio", "cinematic"))
    pl.add_argument("--volume", type=float, default=0.0, help="output volume in dB (default 0)")
    pl.add_argument("--osc", type=int, help="serve OSC control on this UDP port")
    pl.add_argument("--osc-host", default="127.0.0.1", help="OSC bind address (default localhost)")
    pl.add_argument("--quiet", "-q", action="store_true", help="do not print events")
    pl.set_defaults(func=cmd_play)

    mf = sub.add_parser("manifest", help="export the parameter manifest")
    mf.add_argument("--format", "-f", default="json-schema", choices=MANIFEST_FORMATS)
    mf.add_argument("--out", "-o", help="write to a file instead of stdout")
    mf.set_defaults(func=cmd_manifest)

    lc = sub.add_parser("lint-claims", help="check copy for health claims (exit 1 on errors)")
    lc.add_argument("files", nargs="+")
    lc.add_argument("--quiet", "-q", action="store_true")
    lc.set_defaults(func=cmd_lint_claims)

    sc = sub.add_parser("scene", help="convert text / preset / METAR to a scene file")
    sc.add_argument("what", nargs="+", help="preset, scene file, METAR or plain words")
    sc.add_argument("--out", "-o", help="write .json or .yaml (default: print JSON)")
    sc.add_argument("--seed", type=_seed)
    sc.add_argument("--mode", choices=MODES)
    sc.add_argument("--describe", action="store_true", help="print 'What am I hearing?'")
    sc.set_defaults(func=cmd_scene)

    ds = sub.add_parser("destinations", help="list delivery destinations")
    ds.add_argument("--json", action="store_true")
    ds.set_defaults(func=cmd_destinations)
    return p


def _tolerant_std_streams() -> None:
    """Never crash on a label: a Windows pipe (cp1252) cannot encode every character of a scene name."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    limit_blas_threads(1)
    _tolerant_std_streams()
    parser = build_parser()
    args = parser.parse_args(_protect_metar_groups(list(sys.argv[1:] if argv is None else argv)))
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        from .render import RenderCancelled
        if isinstance(exc, RenderCancelled):
            print("cancelled", file=sys.stderr)
            return 130
        known = (CLIError, ValueError, KeyError, TypeError, FileNotFoundError, ImportError, OSError, RuntimeError)
        if isinstance(exc, known):
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            print(f"raingen: error: {msg}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
