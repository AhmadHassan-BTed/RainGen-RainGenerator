"""OSC control surface (MASTER_PLAN 2.6 Integration, 2.12).

Every manifest parameter is reachable at its OSC address (e.g.
``/rg/weather/precipitation/rate_mm_h f``), plus ``<address>/norm f`` (0..1 in
the perceptual domain, for faders) and commands ``/rg/preset s``, ``/rg/text s``,
``/rg/metar s``, ``/rg/pin s [value]``, ``/rg/offset s f``, ``/rg/relink s``.
Values are coerced to stored units (strings may carry units: "30 km/h") and
clamped to the manifest range, then forwarded to ``engine.set(path, value)``.
An optional second float argument is passed as ``ramp=`` seconds when the engine
supports it. Bundle timetags are honoured by python-osc's dispatcher.

python-osc is an optional dependency, imported only when the server starts.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from ..scene import manifest as mf

log = logging.getLogger("raingen.osc")


def _call_set(engine: Any, path: str, value: Any, ramp: Optional[float]) -> None:
    if ramp is not None:
        try:
            engine.set(path, value, ramp=ramp)
            return
        except TypeError:
            pass
    engine.set(path, value)


def _scene_of(engine: Any):
    scene = getattr(engine, "scene", None)
    return scene() if callable(scene) else scene


def _apply_scene(engine: Any, scene: Any) -> None:
    for name in ("set_scene", "load_scene", "load"):
        fn = getattr(engine, name, None)
        if callable(fn):
            fn(scene)
            return
    raise AttributeError("engine has no set_scene/load_scene method")


def make_param_handler(engine: Any, spec: mf.ParamSpec, normalized: bool = False,
                       on_error: Optional[Callable[[str, Exception], None]] = None) -> Callable:
    def handler(address: str, *args: Any) -> None:
        try:
            if not args:
                raise ValueError("missing value")
            ramp = None
            if spec.type == "vec3":
                if len(args) < 3:
                    raise ValueError("position needs x y z")
                value = mf.coerce(spec, args[:3])
                if len(args) > 3:
                    ramp = float(args[3])
            else:
                raw = args[0]
                if len(args) > 1 and isinstance(args[1], (int, float)) and not isinstance(args[1], bool):
                    ramp = float(args[1])
                if normalized:
                    value = mf.denormalize(spec, float(raw))
                else:
                    value = mf.coerce(spec, raw)
                    value = mf.clamp(spec, value)
            _call_set(engine, spec.id, value, ramp)
        except Exception as exc:  # never let a bad packet kill the server thread
            log.warning("OSC %s %r rejected: %s", address, args, exc)
            if on_error is not None:
                on_error(address, exc)
    return handler


def build_dispatcher(engine: Any, on_error: Optional[Callable[[str, Exception], None]] = None):
    """A python-osc Dispatcher wired to `engine` for every manifest address and command."""
    try:
        from pythonosc.dispatcher import Dispatcher
    except ImportError as e:
        raise ImportError("OSC control needs the optional 'python-osc' package: pip install python-osc") from e

    disp = Dispatcher()
    for spec in mf.PARAMS:
        disp.map(spec.osc, make_param_handler(engine, spec, False, on_error))
        if spec.is_numeric and spec.min is not None and spec.max is not None:
            disp.map(spec.osc + "/norm", make_param_handler(engine, spec, True, on_error))

    def _guard(fn: Callable) -> Callable:
        def wrapped(address: str, *args: Any) -> None:
            try:
                fn(*args)
            except Exception as exc:
                log.warning("OSC %s %r failed: %s", address, args, exc)
                if on_error is not None:
                    on_error(address, exc)
        return wrapped

    def cmd_preset(name: str, *_: Any) -> None:
        from ..scene.presets import preset
        _apply_scene(engine, preset(str(name)))

    def cmd_text(text: str, *_: Any) -> None:
        from ..scene.textparse import scene_from_text
        _apply_scene(engine, scene_from_text(str(text), base=_scene_of(engine)))

    def cmd_metar(text: str, *_: Any) -> None:
        from ..scene.metar import scene_from_metar
        _apply_scene(engine, scene_from_metar(str(text), base=_scene_of(engine)))

    def _graph():
        g = getattr(engine, "graph", None)
        g = g() if callable(g) and not hasattr(g, "set_manual") else g
        if g is None:
            raise AttributeError("engine has no parameter graph")
        return g

    def cmd_pin(path: str, *args: Any) -> None:
        g = _graph()
        path = str(path)
        value = args[0] if args else g.resolved_value(path)
        g.set_manual(path, value)

    def cmd_offset(path: str, delta: float, *_: Any) -> None:
        _graph().set_offset(str(path), float(delta), perceptual=True)

    def cmd_relink(path: str, *_: Any) -> None:
        _graph().relink(str(path))

    disp.map("/rg/preset", _guard(cmd_preset))
    disp.map("/rg/text", _guard(cmd_text))
    disp.map("/rg/metar", _guard(cmd_metar))
    disp.map("/rg/pin", _guard(cmd_pin))
    disp.map("/rg/offset", _guard(cmd_offset))
    disp.map("/rg/relink", _guard(cmd_relink))
    disp.set_default_handler(lambda address, *args: log.debug("OSC unmapped %s %r", address, args))
    return disp


class OSCServerHandle:
    """A running OSC server; call stop() to shut it down."""

    def __init__(self, server: Any, thread: Optional[threading.Thread]):
        self.server = server
        self.thread = thread

    @property
    def address(self) -> tuple:
        return self.server.server_address

    def stop(self, timeout_s: float = 2.0) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout_s)

    def __enter__(self) -> "OSCServerHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve_osc(engine: Any, host: str = "127.0.0.1", port: int = 9000, blocking: bool = False,
              on_error: Optional[Callable[[str, Exception], None]] = None) -> OSCServerHandle:
    """Serve the raingen OSC namespace for `engine` (anything with .set(path, value)).

    Binds to localhost by default; pass host='0.0.0.0' deliberately to accept LAN control.
    Non-blocking by default (daemon thread); returns a handle with .stop().
    """
    try:
        from pythonosc.osc_server import ThreadingOSCUDPServer
    except ImportError as e:
        raise ImportError("OSC control needs the optional 'python-osc' package: pip install python-osc") from e
    disp = build_dispatcher(engine, on_error)
    server = ThreadingOSCUDPServer((host, int(port)), disp)
    log.info("raingen OSC listening on %s:%d (%d addresses)", host, port, len(mf.PARAMS))
    if blocking:
        handle = OSCServerHandle(server, None)
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return handle
    thread = threading.Thread(target=server.serve_forever, name="raingen-osc", daemon=True)
    thread.start()
    return OSCServerHandle(server, thread)
