# RainGeneratorPy · raingen

**Weather you can direct.**

Describe the weather the way you would experience it (heavy rain, a gusty westerly, a storm 3 km away, you're in a tent) and get endless, never-repeating, studio-grade sound that you fully own: as a file, as separate stems, as surround or ambisonics, or live.

`raingen` is a physics-driven weather and environment sound engine. Rain is built from drop-size distributions striking real materials, wind is heard through the trees, wires and gaps it excites, thunder comes from a lightning-shaped source that is heard at the speed of sound for the air temperature, and shelters, rooms and the aftermath (drips, gutters, wet ground) change everything the way they do outside. Every render is a pure function of **scene + seed + engine version**, so a 10-hour storm can be shared as a 400-byte scene file and reproduced exactly.

- **Live demo (browser prototype):** <https://ahmadhassan-bted.github.io/RainGeneratorPy/>
- **4D Environment Composer (studio prototype):** <https://ahmadhassan-bted.github.io/RainGeneratorPy/studio/>
- **The plan, the science and the roadmap:** [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md)

> **Project status: alpha.** This repository holds the Python reference implementation of the engine (`raingen/`) and the classic pygame simulator (`rain_generator/`). The engine is new prototype code: treat its sound and numbers as unverified until the Phase 0/1 tests and listening checks in the plan are done. A Rust real-time core, DAW/game plugins and native apps are on the roadmap, not in this repository yet.

---

## Quick start

```bash
git clone https://github.com/AhmadHassan-BTed/RainGeneratorPy.git
cd RainGeneratorPy
pip install -e ".[audio]"          # numpy + scipy, plus sounddevice/soundfile for playback and FLAC

raingen render "rain on a tin roof, wind 30 km/h from the west, distant thunder" --minutes 2 --out cabin.wav
raingen play tent-in-a-storm       # real time; Ctrl+C stops
raingen presets --tag sleep
```

Python 3.10+ is required. The core needs only NumPy and SciPy; everything else is an optional extra:

| Extra | Installs | For |
|---|---|---|
| `audio` | sounddevice, soundfile | real-time playback, FLAC export |
| `yaml` | pyyaml | `.yaml` scene files |
| `osc` | python-osc | OSC control of the live engine |
| `sofa` | h5py | your own HRTFs from SOFA (AES69) files |
| `app` | pygame | the classic visual simulator |
| `all` | everything above | |

## Python API

```python
import raingen as rg

# Plain words -> file (streams to disk, any length). Returns a report dict with a readable str().
report = rg.render("heavy rain on a tent, gusty, thunder 3 km to the north-west",
                   minutes=60, out="tent.flac", destination="youtube")
print(report)          # path, integrated LUFS, true peak, markers, what the parser understood

# Presets and human units
scene = rg.preset("tent-in-a-storm")
scene.weather.precipitation.rate_mm_h = 8          # mm/h
scene.weather.wind.speed_mps = rg.beaufort(5)      # force 5, fresh breeze
scene.weather.wind.from_deg = 270                  # meteorological "from", degrees true
scene.weather.thunder.distance_km = 3              # about 8.7 s flash-to-bang at 20 °C
print(scene.describe())                            # "What am I hearing?"
scene.save("my_tent.json")

# Render, with stems (UCS names), captions and timecode
scene.render(minutes=10, out="tent_stems/", stems=True, captions=True,
             destination="post_stems", layout="5.1", start_tc="01:00:00:00")

# Variations for a sound library or a game: derived seeds + small realistic jitter
for v in scene.variations(20):
    v.render(seconds=90, out="library/")

# Real time (reduced level of detail); set() is thread-safe and glides like the real thing
with rg.Engine(scene, layout="binaural") as eng:
    eng.set("weather.wind.speed_mps", "force 7", ramp=2.0)
    eng.set("place.shelter", "car")                 # structural changes crossfade
    eng.run(seconds=30)

# Weather from elsewhere
rg.scene_from_metar("KJFK 121755Z 21016G24KT 1SM +TSRA BR BKN015CB 18/17 A2990")
rg.scene_from_live("Bergen, Norway")               # Open-Meteo; coordinates rounded to 0.1°
```

Anything that names a scene accepts a `Scene`, a `.json`/`.yaml` scene file, a preset name, a METAR report or plain words.

## Command line

| Command | What it does |
|---|---|
| `raingen render WHAT --minutes 10 --out x.flac --destination youtube` | Render to WAV/BWF (RF64 above 4 GB) or FLAC; `--stems`, `--captions [srt vtt csv edl]`, `--layout`, `--seed`, `--mode`, `--timecode` |
| `raingen variations WHAT --count 20 --out-dir lib/` | N derived variations, UCS-named (`--scenes-only` writes scene files) |
| `raingen presets [--tag sleep] [-v]` | List the scene presets |
| `raingen metar "RAW" [--minutes 5 --out x.wav \| --play]` | Scene from a METAR: print, render or play |
| `raingen live --place NAME \| --lat X --lon Y \| --station ICAO [--play \| --out x.wav]` | Scene from the current weather |
| `raingen play WHAT [--layout binaural] [--osc 9000]` | Real-time playback, events printed as captions |
| `raingen scene WHAT [--out scene.json] [--describe]` | Convert text / preset / METAR to a scene file |
| `raingen manifest --format json-schema\|osc\|wwise\|fmod\|markdown\|json` | Export the parameter manifest |
| `raingen lint-claims FILE...` | Check copy for health claims (exit 1 on errors) |
| `raingen destinations` | Loudness targets per delivery destination |

`python -m raingen ...` works too. Several words need no quotes: `raingen render heavy rain on a car roof --minutes 5`.

## Controls: one set of parameters, four tiers

Every parameter has one canonical id, a human unit, a range, a smoothing time and an OSC address, all generated from one manifest (`raingen manifest`).

| Tier | Who it is for | Parameters |
|---|---|---|
| 0 · Vibe | Everyone | `vibe.intensity`, `vibe.closeness`, `vibe.wildness`, volume, sleep timer, `mode` (sleep, focus, relax, ASMR) |
| 1 · Weather | Listeners and creators | precipitation type, amount (mm/h), drop character, showers, hail size; wind speed (Beaufort), direction, gusts; thunder distance, bearing, approach, flash rate; temperature, humidity, pressure, time of day, storm arc |
| 2 · Place & Listener | Sound designers | place, shelter (open, tree, umbrella, tent, porch, car, room, attic, cave, underpass), surfaces, vegetation, ground, biophony region, traffic, window opening; listener height, heading, position, output layout |
| 3 · Engineer | Studios and researchers | drop-size distribution (μ, Dm, Nw, size limits), water quality, turbulence, integral time, stroke energy, cloud-to-ground share, roughness length, latitude, HRTF, destination, stems |

Higher tiers never fight lower ones: each parameter is Auto, Offset ("real weather, but cozier") or Pinned, and automation lanes win over both.

## Presets

| Preset | Title |
|---|---|
| `steady-rain-leaves-night` | Steady rain on leaves, night (default) |
| `tin-roof-cabin-storm` | Tin-roof cabin in a storm |
| `tent-in-a-storm` | Tent in a storm |
| `car-in-a-thunderstorm` | Car in a thunderstorm |
| `umbrella-city-street` | Umbrella on a city street |
| `porch-summer-thunderstorm` | Porch, summer thunderstorm |
| `tokyo-night-alley` | Tokyo night alley |
| `tropical-monsoon-veranda` | Tropical monsoon veranda |
| `scottish-highland-drizzle` | Scottish highland drizzle |
| `conifer-forest-gale` | Conifer forest in a gale |
| `desert-flash-storm` | Desert flash storm |
| `coastal-squall` | Coastal squall |
| `attic-tin-roof` | Attic under a tin roof |
| `window-rain-bedroom` | Rain on the bedroom window (Sleep mode) |
| `focus-steady-roof` | Focus: steady rain on the roof |
| `asmr-umbrella-close` | ASMR: umbrella, close |
| `after-the-storm-drips` | After the storm: drips |
| `first-fat-drops` | First fat drops |
| `hailstorm-car` | Hailstorm on a car |
| `sleet-city` | Sleet in the city |
| `snow-hush-forest` | Snow hush in the forest |
| `freezing-rain-ice` | Freezing rain and ice |
| `wind-only-ridge` | Wind on the ridge |
| `cricket-warm-night-rain` | Crickets and warm night rain |
| `lake-rain-plinks` | Rain on a still lake |
| `farm-barn-downpour` | Barn door downpour |
| `underpass-shelter` | Under the underpass |
| `jungle-dawn-rain` | Jungle dawn rain |

Presets carry tags in four families (`weather:`, `place:`, `mood:`, `use:`); filter with `raingen presets --tag film`.

## Delivery destinations

| Destination | Integrated loudness | True peak | Notes |
|---|---|---|---|
| `raw` | not normalised | (safety limit −0.3 dBTP for integer PCM) | calibrated engine level, 1.0 = 94 dB SPL |
| `youtube` | −14 LUFS | −1 dBTP | YouTube's figure is unofficial |
| `spotify` | −14 LUFS | −1 dBTP | FLAC |
| `apple_music` | −16 LUFS | −1 dBTP | unofficial |
| `podcast` | −16 LKFS | −1 dBTP | Apple Podcasts |
| `sleep` | −27 LUFS | −1 dBTP | quiet bed, small loudness range |
| `broadcast_eu` | −23 LUFS | −1 dBTP | EBU R128, BWF with bext loudness |
| `streaming_broadcaster` | −18 LUFS | −1 dBTP | EBU R128 s2 |
| `broadcast_us` | −24 LKFS | −2 dBTP | ATSC A/85 |
| `post_stems` | not normalised | −2 dBTP | film/TV stems keep calibrated relative levels |
| `game` | not normalised | −1 dBTP | for middleware mixing |
| `vr` | not normalised | −1 dBTP | first-order AmbiX (W, Y, Z, X) |

Normalisation is two-pass without temporary files: the render is deterministic, so pass 1 only measures integrated loudness (ITU-R BS.1770-5) and pass 2 renders again with the exact gain, followed by an 8× oversampled true-peak limiter and TPDF dither. Output layouts: `stereo`, `binaural`, `5.1`, `7.1`, `7.1.4`, `foa`, `hoa3`, `mono`.

## Architecture

```
Scene (human units, JSON/YAML)  + timeline automation + Auto/Offset/Pinned overrides
  → World.step()          drop-size distribution, five acoustic drop classes, Kaimal gusts,
                          storm cells and arcs, wetness/puddle/canopy stores, day/night, stability
  → Generators            rain patches (modal + granular level of detail), diffuse bed, drips,
                          gutters, hail, snow, wind, vegetation, aeolian tones, thunder,
                          biophony, traffic, car wipers
  → Encoder               3rd-order ambisonics (AmbiX) bus per stem, spread and decorrelation
  → Shelter + room        envelope transmission loss (mass law + leaks), SH-domain FDN reverb
  → Decoder               stereo · binaural (HRTF) · 5.1 · 7.1.4 (AllRAD) · FOA/HOA · mono
  → Master                Keep-it-soothing governor (wellness modes) · gain · true-peak limiter
  → Writers               streaming BWF/RF64 with bext + iXML (scene, seed, engine version), FLAC,
                          UCS file names, SRT/VTT/CSV/EDL captions and cue sheets
```

| Package | Contents |
|---|---|
| `raingen.core` | constants, counter-based random streams (Philox keyed by layer name), shared data types |
| `raingen.scene` | scene schema, units, parameter manifest, parameter graph, vibe macros, presets, text and METAR parsers, timeline |
| `raingen.physics` | drop-size distributions and fall speeds, ISO 9613-1 air absorption, ground and refraction, wind and gusts, wetness and drips, storm system, world, layout |
| `raingen.synth` | materials, kernels (impact, bubble, modal), event processes, rain, wind, vegetation, aeolian, thunder, biophony, places and shelters |
| `raingen.spatial` | spherical harmonics, encoder, decoders, built-in spherical-head HRTF and SOFA loading |
| `raingen.master` | BS.1770-5 loudness meter, true-peak limiter, dither, destinations |
| `raingen.io` | streaming WAV/RF64/FLAC writers, UCS naming, captions |
| `raingen.live`, `raingen.control`, `raingen.wellness` | live weather providers, OSC, wellness modes, dose meter, claims linter |
| `raingen.render`, `raingen.engine`, `raingen.api`, `raingen.cli` | offline renderer, real-time engine, public API, command line |
| `site/` | the browser prototype and the 4D Composer (JavaScript / Web Audio) |

All randomness comes from `RngFactory.get(layer, block)`, so adding a sound layer never changes any other layer, and the same scene and seed render the same audio on the same build.

## The classic simulator (v2)

The original pygame visual simulator still works and now renders its audio exports through raingen:

```bash
pip install -e ".[app,audio]"
python -m rain_generator --preset thunderstorm --fullscreen
python thunderstorm_simulator.py --duration 120 --export storm.wav --preset thunderstorm
```

Presets `drizzle`, `steady`, `thunderstorm`, `gale` and `cozy` map onto raingen presets. Keys: `F`/`F11` fullscreen, `Space` pause, `M` mute, `G` audio export, `H` HUD, `L` lightning, `1`–`5` presets, `S`/`Tab` mixer, `?` help, `Esc` close/exit.

## Wellness, honestly

Sleep, Focus, Relax and ASMR modes shape the sound with a "Keep it soothing" governor (no sudden loud events, a darker tilt, gentle timers). They are made to help you relax, focus and keep a bedtime routine. raingen makes no health claims, and the repository ships a claims linter (`raingen lint-claims`) that keeps its copy that way.

## Licence

MIT. See [LICENSE](LICENSE).
