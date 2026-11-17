"""The website's claims and code examples hold for the raingen that ships today.

The Python, CLI, scene-file and browser-engine examples are read straight out of
site/index.html:

* the Python and CLI examples are executed as written, in a temporary directory, with the
  expensive parts replaced by stand-ins that keep the real argument checking: the
  Renderer is constructed for real (destination, layout, bit depth, mode, timecode and
  output path are all validated) but does not synthesize, the real-time Engine is built
  and accepts set() commands but opens no audio device, and live weather and thunder
  one-shots return canned data;
* the scene-file example loads with Scene.from_json and uses only known fields;
* the browser example only imports and calls what site/engine/weather-engine.js provides
  (and runs under Node when it is installed).

The loudness table is compared with the engine's destinations, the scene-file size
quoted on the page is measured, and the page copy is run through the claims linter.
Everything here is fast; the one real render is marked slow and lasts 0.25 s.
"""

from __future__ import annotations

import ast
import html
import inspect
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

import raingen
import raingen.api as api
import raingen.cli as cli
from raingen.engine import Engine
from raingen.render import CAPTION_FORMATS, Renderer, RenderReport
from raingen.scene.schema import Keyframe, Listener, Place, Precipitation, Scene, SceneObject, Thunder, Weather, Wind

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
INDEX = SITE / "index.html"
ENGINE_JS = SITE / "engine" / "weather-engine.js"


# ---------------------------------------------------------------------------
# reading the page
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def page() -> str:
    return INDEX.read_text(encoding="utf-8")


def snippet(page: str, code_id: str) -> str:
    """The text of <code id=...> exactly as a visitor copies it."""
    m = re.search(rf'<code id="{code_id}"[^>]*>(.*?)</code>', page, re.S)
    assert m, f"no <code id={code_id!r}> on the page"
    return html.unescape(m.group(1))


def cli_lines(page: str) -> list[list[str]]:
    out = []
    for line in snippet(page, "code-cli").splitlines():
        if not line.strip():
            continue
        argv = shlex.split(line, comments=True)
        assert argv and argv[0] == "raingen", f"CLI example does not start with 'raingen': {line!r}"
        out.append(argv)
    return out


# ---------------------------------------------------------------------------
# stand-ins: real validation, no synthesis, no devices, no network
# ---------------------------------------------------------------------------
class FakeRenderer(Renderer):
    """The real Renderer constructor and output-path logic; run() skips the synthesis."""

    calls: list["FakeRenderer"] = []

    def run(self, out=None, captions=False) -> RenderReport:
        path = self._output_path(out)
        if captions and captions is not True:
            formats = captions.replace(",", " ").split() if isinstance(captions, str) else list(captions)
            unknown = [f for f in formats if str(f).lower() not in CAPTION_FORMATS]
            assert not unknown, f"unknown caption formats {unknown}"
        self.out_path = path
        self.captions = captions
        FakeRenderer.calls.append(self)
        return RenderReport(path=path, scene=self.scene.name, seconds=self.seconds, layout=self.layout,
                            sample_rate=self.sr_out, bit_depth=self.bit_depth, destination=self.dest["name"],
                            target_lufs=self.target_lufs, ceiling_dbtp=self.ceiling_dbtp, markers=0)


class FakeEngine(Engine):
    """The real Engine (scene, layout and set() validation) without an audio stream or OSC socket."""

    instances: list["FakeEngine"] = []

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.ran_for = "not run"
        FakeEngine.instances.append(self)

    def start(self, audio: bool = True) -> "FakeEngine":
        self._running = True
        return self

    def stop(self) -> None:
        self._running = False

    def run(self, seconds=None, poll_s: float = 0.05) -> None:
        self.ran_for = seconds

    def queued(self) -> list[tuple]:
        cmds = []
        while not self._cmd.empty():
            cmds.append(self._cmd.get_nowait())
        return cmds


@pytest.fixture
def stand_ins(monkeypatch, tmp_path):
    import raingen.live.providers as providers

    FakeRenderer.calls = []
    FakeEngine.instances = []
    live_calls = []

    def fake_live(location, base=None, provider="open-meteo", **kw):
        live_calls.append((location, provider))
        return raingen.preset("steady-rain-leaves-night")

    def fake_one_shots(count=3, distance_km=3.0, sample_rate=48000, seed=None, **kw):
        return [np.zeros((2, 480)) for _ in range(int(count))]

    monkeypatch.setattr(api, "Renderer", FakeRenderer)
    monkeypatch.setattr(api, "Engine", FakeEngine)
    monkeypatch.setattr(raingen, "Engine", FakeEngine, raising=False)
    monkeypatch.setattr(api, "thunder_one_shots", fake_one_shots)
    monkeypatch.setattr(providers, "scene_from_live", fake_live)
    monkeypatch.chdir(tmp_path)
    return {"live": live_calls, "dir": tmp_path}


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
def test_top_level_render_is_the_function():
    """`rg.render(...)` is the render function, also after raingen.api (and with it the
    raingen.render submodule) has been imported, as the page's examples do."""
    import importlib

    rg = raingen
    rg.preset("tent-in-a-storm")
    importlib.import_module("raingen.render")       # binding the submodule must not shadow the function
    assert callable(rg.render) and rg.render is api.render, f"rg.render is {rg.render!r}"


def test_python_example_names_exist(page):
    """Every rg.<name> exists, and keyword arguments are accepted by the function they reach."""
    code = snippet(page, "code-py")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "rg":
            assert hasattr(raingen, node.attr), f"raingen has no attribute {node.attr!r}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "rg":
            fn = getattr(raingen, node.func.attr)
            assert callable(fn), f"rg.{node.func.attr} is not callable: it is {fn!r}"
            target = fn.__init__ if inspect.isclass(fn) else fn
            params = inspect.signature(target).parameters
            has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
            # render() forwards extra keywords to the Renderer
            extra = inspect.signature(Renderer.__init__).parameters if node.func.attr == "render" else {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                ok = kw.arg in params or kw.arg in extra or (has_var_kw and node.func.attr != "render")
                assert ok, f"rg.{node.func.attr}() does not take {kw.arg}="
    for name in re.findall(r'rg\.preset\("([^"]+)"\)', code):
        assert name in raingen.PRESETS, f"unknown preset {name!r}"


def test_python_example_plain_words_are_understood(page):
    from raingen.scene.textparse import parse_text

    code = snippet(page, "code-py")
    phrases = re.findall(r'rg\.(?:play|render)\("([^"]+)"', code)
    assert phrases
    for text in phrases:
        assert parse_text(text).matched, f"the text parser understood nothing in {text!r}"


def test_python_example_runs(page, stand_ins):
    code = snippet(page, "code-py")
    exec(compile(code, "site/index.html#code-py", "exec"), {"__name__": "site_example"})
    tmp = stand_ins["dir"]

    renders = FakeRenderer.calls
    by_out = {Path(r.out_path).name: r for r in renders}
    assert "cabin_storm.wav" in by_out and by_out["cabin_storm.wav"].dest["name"] == "youtube"
    ten_h = by_out["tent_sleep.wav"]
    assert ten_h.seconds == 10 * 3600 and ten_h.scene.mode == "sleep" and ten_h.dest["name"] == "sleep"
    atmos = by_out["storm_714.wav"]
    assert atmos.layout == "7.1.4" and atmos.stems and atmos._channels_for_layout() == 12
    assert atmos.start_tc == "01:00:00:00" and atmos.dest["name"] == "post_stems"
    assert [k.path for k in atmos.scene.timeline].count("weather.precipitation.rate_mm_h") >= 3
    lib = [r for r in renders if Path(r.out_path).parent.name == "library"]
    assert len(lib) == 20 and all(r.seconds == 90 for r in lib)
    assert len({r.scene.seed for r in lib}) == 20

    saved = tmp / "my_tent.json"
    assert saved.is_file()
    scene = raingen.Scene.from_json(saved.read_text(encoding="utf-8"))
    assert scene.weather.precipitation.rate_mm_h == 8 and scene.weather.wind.from_deg == 270
    assert saved.stat().st_size < 3 * 1024, "the page says the saved storm is about 2 KB"
    assert len(list((tmp / "thunder_pack").glob("*.wav"))) == 12

    played, live = FakeEngine.instances
    assert played.ran_for is None                    # rg.play blocks until Ctrl+C
    cmds = {c[1]: c[2] for c in live.queued() if c[0] == "set"}
    assert cmds["weather.wind.speed_mps"] == 14
    assert cmds["weather.thunder.distance_km"] == pytest.approx(2.0)
    assert cmds["place.shelter"] == "car"
    assert live.layout == "binaural" and live.ran_for == 60


@pytest.mark.slow
def test_python_plain_words_render_for_real(tmp_path):
    """A 0.25 s real render of the page's plain-words example."""
    out = tmp_path / "cabin_storm.wav"
    report = raingen.render("rain on a tin roof, wind 30 km/h from the west, distant thunder",
                            seconds=0.25, out=str(out), destination="youtube")
    assert out.is_file() and out.stat().st_size > 1000
    assert report["understood"] and "Rendered" in str(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_example_flags_are_documented(page):
    """Each flag used on the page appears in that subcommand's --help."""
    parser = cli.build_parser()
    subs = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction").choices
    for argv in cli_lines(page):
        sub = argv[1]
        assert sub in subs, f"unknown command 'raingen {sub}'"
        help_text = subs[sub].format_help()
        for tok in argv[2:]:
            if tok.startswith("--"):
                assert re.search(rf"(?<![\w-]){re.escape(tok)}(?![\w-])", help_text), \
                    f"'raingen {sub} --help' does not list {tok}"


def test_cli_example_plain_words_are_understood(page):
    from raingen.scene.textparse import parse_text

    found = 0
    for argv in cli_lines(page):
        if argv[1] in ("render", "play") and " " in argv[2]:
            found += 1
            assert parse_text(argv[2]).matched, f"the text parser understood nothing in {argv[2]!r}"
    assert found


def test_cli_example_runs(page, stand_ins, capsys):
    for argv in cli_lines(page):
        code = cli.main(argv[1:])
        assert code == 0, f"{shlex.join(argv)} exited with {code}: {capsys.readouterr().err}"
    tmp = stand_ins["dir"]
    outs = {Path(r.out_path).name: r for r in FakeRenderer.calls}
    assert outs["car.flac"].dest["name"] == "youtube" and outs["car.flac"].seconds == 30 * 60
    ten_h = [r for r in FakeRenderer.calls if r.seconds == 10 * 3600]
    assert len(ten_h) == 1 and ten_h[0].layout == "binaural" and ten_h[0].scene.mode == "sleep"
    assert outs["jfk.wav"].seconds == 600 and outs["bergen.wav"].seconds == 300
    assert len([r for r in FakeRenderer.calls if Path(r.out_path).parent.name == "library"]) == 20
    assert (tmp / "my_tent.json").is_file() and (tmp / "raingen_wwise.json").is_file()
    assert stand_ins["live"] == [("Bergen, Norway", "open-meteo")]
    assert any(e._osc_cfg == ("127.0.0.1", 9000) for e in FakeEngine.instances)


# ---------------------------------------------------------------------------
# scene file
# ---------------------------------------------------------------------------
_NESTED = {
    (Scene, "weather"): Weather, (Scene, "listener"): Listener, (Scene, "place"): Place,
    (Scene, "timeline"): Keyframe, (Weather, "precipitation"): Precipitation, (Weather, "wind"): Wind,
    (Weather, "thunder"): Thunder, (Place, "objects"): SceneObject,
}


def _unknown_keys(tp, data, where="scene") -> list[str]:
    names = {f.name for f in fields(tp)}
    bad = []
    for key, value in data.items():
        if tp is Scene and key == "rainscene":
            continue
        if key not in names:
            bad.append(f"{where}.{key}")
            continue
        sub = _NESTED.get((tp, key))
        if sub is not None and is_dataclass(sub):
            for i, item in enumerate(value if isinstance(value, list) else [value]):
                bad += _unknown_keys(sub, item, f"{where}.{key}" + (f"[{i}]" if isinstance(value, list) else ""))
    return bad


def test_scene_file_example_loads(page):
    from raingen.scene.presets import place_template
    from raingen.scene.timeline import validate
    from raingen.synth.materials import get_material

    text = snippet(page, "code-json")
    data = json.loads(text)
    assert data["rainscene"] == 1
    assert _unknown_keys(Scene, data) == []
    scene = Scene.from_json(text)
    assert validate(scene) == []
    place_template(scene.place.preset)
    for name in scene.place.surfaces:
        get_material(name)
    for obj in scene.place.objects:
        if obj.material:
            get_material(obj.material)
    assert Scene.from_dict(scene.to_dict()).to_dict() == scene.to_dict()


def test_scene_files_are_about_2_kb():
    """The hero badge: 'A storm is a ~2 KB scene file' (compact JSON of the built-in presets)."""
    sizes = sorted(len(json.dumps(raingen.preset(n).to_dict(), separators=(",", ":"))) for n in raingen.PRESETS)
    median = sizes[len(sizes) // 2]
    assert 1000 <= median <= 2600, f"median preset scene is {median} bytes"


# ---------------------------------------------------------------------------
# browser engine
# ---------------------------------------------------------------------------
def test_browser_example_uses_real_exports(page):
    code = snippet(page, "code-js")
    js = ENGINE_JS.read_text(encoding="utf-8")
    m = re.search(r"import \{([^}]+)\} from '\./engine/weather-engine\.js';", code)
    assert m, "the browser example must import from ./engine/weather-engine.js"
    exported = set(re.findall(r"^export (?:async )?(?:class|function|const) (\w+)", js, re.M))
    for name in (n.strip() for n in m.group(1).split(",")):
        assert name in exported, f"weather-engine.js does not export {name}"
    body = js[js.index("export class WeatherEngine"):]
    for method in set(re.findall(r"\bengine\.(\w+)\(", code)):
        assert re.search(rf"^\s+(?:async\s+)?{method}\(", body, re.M), f"WeatherEngine has no method {method}()"
    for event in re.findall(r"engine\.on\('(\w+)'", code):
        assert f"_emit('{event}'" in js, f"the engine never emits '{event}'"
    for fields_txt in re.findall(r"\(\{([^}]+)\}\) =>", code):
        for field in (f.strip() for f in fields_txt.split(",")):
            assert re.search(rf"\b{field}\b", body), f"event payload field {field!r} not found"
    for path in re.findall(r"setParam\('([\w.]+)'", code) + re.findall(r"\bscene\.((?:\w+\.)+\w+) =", code):
        Scene().get(path)                             # the shared rainscene format has this path


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_browser_example_runs_in_node(page, tmp_path):
    """Everything in the example except the audio-only calls (start, renderOffline) runs in Node."""
    code = snippet(page, "code-js")
    code = code.replace("'./engine/weather-engine.js'", repr((ENGINE_JS).as_uri()))
    code = re.sub(r"^await engine\.start\(\);.*$", "", code, flags=re.M)
    code = re.sub(r"^const wav = await engine\.renderOffline\(.*$",
                  "if (typeof engine.renderOffline !== 'function') throw new Error('no renderOffline');", code, flags=re.M)
    code += ("\nengine.strikeLightning({ distanceKm: 3 });\n"
             "if (engine.scene.place.shelter !== 'tent') throw new Error('scene not applied');\n"
             "console.log('ok');\n")
    script = tmp_path / "example.mjs"
    script.write_text(code, encoding="utf-8")
    res = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert "bolt of" in res.stdout and res.stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------
# numbers and copy
# ---------------------------------------------------------------------------
def _first_number(cell: str):
    m = re.search(r"[-−]?\d+(?:\.\d+)?", html.unescape(re.sub(r"<[^>]+>", "", cell)))
    return None if m is None else float(m.group(0).replace("−", "-"))


def test_loudness_table_matches_destinations(page):
    rows = re.findall(r'<tr[^>]*data-dest="([^"]+)"[^>]*>(.*?)</tr>', page, re.S)
    assert len(rows) >= 10
    dests = {d["name"]: d for d in raingen.destinations()}
    seen = set()
    for keys, row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        loud, peak = _first_number(cells[0]), _first_number(cells[1])
        for key in keys.split():
            d = dests[key]
            seen.add(key)
            if d["normalize"] and d["target_lufs"] is not None:
                assert loud == d["target_lufs"], f"{key}: page says {loud}, engine targets {d['target_lufs']}"
            else:
                assert loud is None, f"{key} is not normalised but the page gives a loudness"
            assert peak == d["tp_dbtp"], f"{key}: page says {peak} dBTP, engine uses {d['tp_dbtp']}"
    assert seen >= set(dests) - {"raw"}


def test_ucs_name_on_the_page_is_the_engines(page):
    from raingen.io.ucs import ucs_filename

    shown = re.search(r"<code>(RAIN\w+_[^<]+_RGN_raingen)</code>", page)
    assert shown, "no UCS file name example on the page"
    assert html.unescape(shown.group(1)) == ucs_filename(raingen.load_scene("rain on a tent, gusty"))


def test_page_copy_passes_the_claims_linter():
    from raingen.wellness.claims import lint_text

    for path in (INDEX, SITE / "404.html"):
        findings = lint_text(path.read_text(encoding="utf-8"))
        assert findings == [], [f"{path.name}:{f['line']}: {f['rule']} {f['match']!r}" for f in findings]


def test_no_internal_planning_material_on_the_site():
    assert not (SITE / "plan.html").exists()
    assert not list(SITE.rglob("MASTER_PLAN*"))
    pattern = re.compile(r"master ?plan|plan\.html|roadmap|coming soon|placeholder name", re.I)
    for path in (INDEX, SITE / "404.html", SITE / "assets" / "site.css"):
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.relative_to(ROOT)}: {hits}"
