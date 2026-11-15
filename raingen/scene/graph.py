"""The parameter graph: Auto / Offset / Pinned resolution (MASTER_PLAN 2.7).

Every parameter is in one of three states:

* **auto**   - follows its driver (scene default, macro, live data, automation);
* **offset** - driver plus a delta stored in the *perceptual* domain (log2 for
  rain rate and distances, Beaufort for wind, plain units otherwise), the CLAP
  "plain value delta" model: "real weather, but cozier";
* **pinned** - ignores the driver.

    resolved = pinned ? value : clamp(driver + offset)

Driver priority: automation lane -> pinned value -> (live data or macro) + offset
-> scene default. State is stored in `scene.overrides` as
``path -> {"state": "offset"|"pinned", "offset": float, "value": Any}`` so it
round-trips with the scene file. One gesture is one undo step.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from . import manifest as mf
from .schema import Scene

STATES = ("auto", "offset", "pinned")
_MISSING = object()


def _get(scene: Scene, path: str, default: Any = _MISSING) -> Any:
    try:
        return scene.get(path)
    except (KeyError, AttributeError, TypeError):
        if default is _MISSING:
            spec = mf.find_param(path)
            return spec.default if spec is not None else None
        return default


def _set(scene: Scene, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = scene
    for part in parts[:-1]:
        if isinstance(obj, dict):
            obj = obj.setdefault(part, {})
        else:
            obj = getattr(obj, part)
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    else:
        setattr(obj, parts[-1], value)


class ParameterGraph:
    """Auto/Offset/Pinned state over one scene; `apply()` resolves a driven scene."""

    def __init__(self, scene: Scene, max_undo: int = 200):
        self.scene = scene
        if not isinstance(getattr(scene, "overrides", None), dict):
            scene.overrides = {}
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._max_undo = int(max_undo)
        self._gesture_depth = 0
        self._gesture_recorded = False
        self._last_driven: Optional[Scene] = None
        self._sources: dict[str, str] = {}

    # ------------------------------------------------------------------ state
    @property
    def overrides(self) -> dict:
        return self.scene.overrides

    def state(self, path: str) -> str:
        entry = self.overrides.get(path)
        if not entry:
            return "auto"
        st = entry.get("state", "auto")
        return st if st in STATES else "auto"

    def entry(self, path: str) -> dict:
        return dict(self.overrides.get(path) or {"state": "auto"})

    def set_manual(self, path: str, value: Any) -> Any:
        """Pin `path` to `value` (strings may carry units, e.g. '30 km/h'). Returns the stored value."""
        spec = mf.find_param(path)
        if spec is not None:
            value = mf.coerce(spec, value, temp_c=self._temp())
            value = mf.clamp(spec, value)
        self._record()
        self.overrides[path] = {"state": "pinned", "value": list(value) if isinstance(value, tuple) else value}
        return value

    def set_offset(self, path: str, delta: float, perceptual: bool = True, driver: Any = None) -> float:
        """Offset `path` from its driver.

        perceptual=True: `delta` is already in the perceptual domain (doublings for log
        parameters, Beaufort steps for wind, plain units otherwise).
        perceptual=False: `delta` is a plain-unit nudge (e.g. +3 mm/h) relative to the
        current driver; it is converted and stored perceptually so it scales with the driver.
        Returns the stored perceptual offset.
        """
        spec = mf.get_param(path)
        if not spec.is_numeric:
            raise ValueError(f"{path} is not numeric; pin it instead")
        if perceptual:
            p_off = float(delta)
        else:
            ref = driver if driver is not None else self.auto_value(path)
            if ref is None:
                ref = spec.default if spec.default is not None else spec.min
            ref = float(ref)
            p_off = mf.to_perceptual(spec, mf.clamp(spec, ref + float(delta))) - mf.to_perceptual(spec, ref)
        self._record()
        if p_off == 0.0:
            self.overrides.pop(path, None)
        else:
            self.overrides[path] = {"state": "offset", "offset": p_off}
        return p_off

    def nudge(self, path: str, delta: float, perceptual: bool = True) -> float:
        """Add to the existing offset (or start one). Pinned parameters move their pinned value."""
        spec = mf.get_param(path)
        if self.state(path) == "pinned":
            cur = self.overrides[path]["value"]
            if perceptual:
                new = mf.from_perceptual(spec, mf.to_perceptual(spec, cur) + float(delta))
            else:
                new = float(cur) + float(delta)
            self.set_manual(path, new)
            return float(delta)
        cur_off = float(self.overrides.get(path, {}).get("offset", 0.0)) if self.state(path) == "offset" else 0.0
        if perceptual:
            add = float(delta)
        else:
            ref = self.resolved_value(path)
            if ref is None:
                ref = spec.default
            ref = float(ref)
            add = mf.to_perceptual(spec, mf.clamp(spec, ref + float(delta))) - mf.to_perceptual(spec, ref)
        return self.set_offset(path, cur_off + add, perceptual=True)

    def relink(self, path: str) -> None:
        """Return `path` to Auto (Alt-click on the ghost marker)."""
        if path in self.overrides:
            self._record()
            del self.overrides[path]

    def relink_all(self, prefix: str = "") -> None:
        keys = [k for k in self.overrides if k.startswith(prefix)]
        if keys:
            self._record()
            for k in keys:
                del self.overrides[k]

    # ------------------------------------------------------------ resolution
    def resolve(self, path: str, driver: Any) -> Any:
        """Resolved value for one path given its driver value."""
        entry = self.overrides.get(path)
        if not entry:
            return driver
        state = entry.get("state", "auto")
        spec = mf.find_param(path)
        if state == "pinned":
            value = entry.get("value")
            if spec is not None and spec.type == "vec3" and isinstance(value, list):
                value = tuple(value)
            return value
        if state == "offset":
            if spec is None or not spec.is_numeric or driver is None:
                return driver
            off = float(entry.get("offset", 0.0))
            if spec.scale == "log" and float(driver) <= 0.0:
                return driver            # a multiplicative offset keeps zero at zero (no rain stays no rain)
            p = mf.to_perceptual(spec, driver) + off
            return mf.clamp(spec, mf.from_perceptual(spec, p))
        return driver

    def apply(self, driven: Scene, automated: Optional[Iterable[str]] = None, respect_automation: bool = True,
              source: str = "Scene") -> Scene:
        """Resolve every override on top of `driven` and return a new Scene.

        `driven` is the scene after drivers ran (timeline evaluation, vibe macros, live data).
        Paths with an automation lane (keyframes in driven.timeline, or listed in `automated`)
        win over pins and offsets, per the driver priority of MASTER_PLAN 2.7.
        `source` names the driver for provenance badges ('Scene', 'Macro' or 'Live').
        """
        out = driven.copy()
        if automated is None:
            automated = {kf.path for kf in (driven.timeline or []) if hasattr(kf, "path")}
        automated = set(automated) if respect_automation else set()
        self._last_driven = driven
        self._sources = {"*": source}
        for path in automated:
            self._sources[path] = "Automation"
        for path, entry in list(self.overrides.items()):
            if path in automated:
                continue
            driver = _get(driven, path)
            value = self.resolve(path, driver)
            try:
                _set(out, path, value)
            except (AttributeError, KeyError, TypeError):
                continue
        out.overrides = copy.deepcopy(self.overrides)
        return out

    def auto_value(self, path: str) -> Any:
        """Where Auto would put this control (the ghost marker)."""
        src = self._last_driven if self._last_driven is not None else self.scene
        return _get(src, path)

    def resolved_value(self, path: str) -> Any:
        return self.resolve(path, self.auto_value(path))

    def at_limit(self, path: str) -> bool:
        """True while a stored (unclamped) adjustment is being clamped by the range: '(at limit)'."""
        entry = self.overrides.get(path)
        spec = mf.find_param(path)
        if not entry or spec is None or not spec.is_numeric or entry.get("state") != "offset":
            return False
        driver = self.auto_value(path)
        if driver is None or (spec.scale == "log" and float(driver) <= 0.0):
            return False
        p = mf.to_perceptual(spec, driver) + float(entry.get("offset", 0.0))
        raw = mf.from_perceptual(spec, p)
        return mf.clamp(spec, raw) != raw and not spec.wrap

    def label(self, path: str) -> str:
        """Plain-words state for the UI: 'Following scene', 'Adjusted' or 'Locked' (never auto/offset/pinned)."""
        st = self.state(path)
        if st == "pinned":
            return "Locked"
        if st == "offset":
            return "Adjusted (at limit)" if self.at_limit(path) else "Adjusted"
        src = self.provenance(path)
        return "Automated" if src == "Automation" else f"Following {src.lower() if src != 'Live' else 'live weather'}"

    def provenance(self, path: str) -> str:
        """Badge text: Scene, Macro, Live, Automation or Manual (with '+ offset' when offset)."""
        if self._sources.get(path) == "Automation":
            return "Automation"
        st = self.state(path)
        if st == "pinned":
            return "Manual"
        base = self._sources.get("*", "Scene")
        if path.startswith("vibe.") or path == "mode":
            base = "Scene"
        return f"{base} + offset" if st == "offset" else base

    def summary(self) -> list[dict]:
        rows = []
        for path in sorted(self.overrides):
            e = self.overrides[path]
            rows.append({"path": path, "state": e.get("state"), "offset": e.get("offset"), "value": e.get("value"),
                         "auto": self.auto_value(path), "resolved": self.resolved_value(path),
                         "provenance": self.provenance(path)})
        return rows

    # ------------------------------------------------------------------- undo
    def _record(self) -> None:
        if self._gesture_depth > 0:
            if self._gesture_recorded:
                return
            self._gesture_recorded = True
        self._undo.append(copy.deepcopy(self.overrides))
        if len(self._undo) > self._max_undo:
            self._undo.pop(0)
        self._redo.clear()

    def begin_gesture(self) -> None:
        if self._gesture_depth == 0:
            self._gesture_recorded = False
        self._gesture_depth += 1

    def end_gesture(self) -> None:
        self._gesture_depth = max(0, self._gesture_depth - 1)
        if self._gesture_depth == 0:
            self._gesture_recorded = False

    @contextmanager
    def gesture(self):
        """Group many edits (a macro drag) into one undo step."""
        self.begin_gesture()
        try:
            yield self
        finally:
            self.end_gesture()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self.overrides))
        self._replace_overrides(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self.overrides))
        self._replace_overrides(self._redo.pop())
        return True

    def _replace_overrides(self, state: dict) -> None:
        self.overrides.clear()
        self.overrides.update(state)

    def _temp(self) -> float:
        try:
            return float(self.scene.weather.temperature_c)
        except (AttributeError, TypeError, ValueError):
            return 20.0
