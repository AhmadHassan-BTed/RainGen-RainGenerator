"""Delivery destinations: loudness/true-peak targets and file formats.

Targets: EBU R128 (2023 rev.) -23 LUFS / -1 dBTP; R128 s2 streaming broadcasters -18 LUFS
(range -20..-16); ATSC A/85 -24 LKFS / -2 dBTP; music streaming -14 LUFS (YouTube's figure
is unofficial), Apple Music ~-16 LUFS; Apple Podcasts -16 LKFS +-1, <= -1 dBTP, stereo.
'sleep' is raingen's own quiet profile (-27 LUFS, LRA <= 4 LU). Post-production stems and
game assets are deliberately NOT loudness-normalised (calibrated relative levels).
"""

from __future__ import annotations

import copy
import math
from typing import Optional

DESTINATIONS: dict[str, dict] = {
    "raw": dict(target_lufs=None, tp_dbtp=None, lra_max=None, sample_rate=48000, bit_depth=24,
                format="wav", normalize=False, layout_hint=None,
                notes="No loudness processing; engine levels as rendered (float headroom clipped only at export)."),
    "youtube": dict(target_lufs=-14.0, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                    format="wav", normalize=True, layout_hint="stereo",
                    notes="YouTube normalises playback to about -14 LUFS (unofficial). Use -2 dBTP if the master is louder."),
    "spotify": dict(target_lufs=-14.0, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                    format="flac", normalize=True, layout_hint="stereo",
                    notes="Spotify 'normal' playback -14 LUFS; functional/noise tracks are paid only from 2:00."),
    "apple_music": dict(target_lufs=-16.0, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                        format="wav", normalize=True, layout_hint="stereo",
                        notes="Sound Check plays around -16 LUFS (unofficial)."),
    "podcast": dict(target_lufs=-16.0, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                    format="wav", normalize=True, layout_hint="stereo",
                    notes="Apple Podcasts: -16 LKFS +-1 LU, true peak <= -1 dBFS, stereo, >= 44.1 kHz."),
    "sleep": dict(target_lufs=-27.0, tp_dbtp=-1.0, lra_max=4.0, sample_rate=48000, bit_depth=24,
                  format="wav", normalize=True, layout_hint="stereo",
                  notes="Quiet bed for sleep: LRA <= 4 LU, no event more than ~3 dB over the bed, long fade-out."),
    "broadcast_eu": dict(target_lufs=-23.0, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                         format="wav", normalize=True, layout_hint=None,
                         notes="EBU R128: -23.0 LUFS integrated, -1 dBTP max; BWF with bext v2 loudness fields."),
    "streaming_broadcaster": dict(target_lufs=-18.0, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                                  format="wav", normalize=True, layout_hint="stereo",
                                  notes="EBU R128 s2 (streaming): -20..-16 LUFS, -1 dBTP; -18 LUFS chosen."),
    "broadcast_us": dict(target_lufs=-24.0, tp_dbtp=-2.0, lra_max=None, sample_rate=48000, bit_depth=24,
                         format="wav", normalize=True, layout_hint=None,
                         notes="ATSC A/85: -24 LKFS +-2 integrated, true peak below -2 dBTP."),
    "post_stems": dict(target_lufs=None, tp_dbtp=-2.0, lra_max=None, sample_rate=48000, bit_depth=24,
                       format="wav", normalize=False, layout_hint=None,
                       notes="Film/TV stems: calibrated, not normalised (dialogue-gated programme specs stay intact); "
                             "<= -2 dBTP headroom; BWF 48k/24 with timecode."),
    "game": dict(target_lufs=None, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
                 format="wav", normalize=False, layout_hint=None,
                 notes="Calibrated levels for middleware mixing; UCS names; loop points where requested."),
    "vr": dict(target_lufs=None, tp_dbtp=-1.0, lra_max=None, sample_rate=48000, bit_depth=24,
               format="wav", normalize=False, layout_hint="foa",
               notes="AmbiX FOA (ACN/SN3D: W, Y, Z, X) at 48 kHz for 360 video; HOA for engines."),
}


def get_destination(name: Optional[str]) -> dict:
    """A copy of the destination profile (unknown/None -> 'raw')."""
    key = (name or "raw").strip().lower().replace("-", "_").replace(" ", "_")
    return copy.deepcopy(DESTINATIONS.get(key, DESTINATIONS["raw"])) | {"name": key if key in DESTINATIONS else "raw"}


def normalization_gain_db(measured_lufs: float, destination: str) -> float:
    """Static gain (dB) that brings `measured_lufs` to the destination target (0 if not normalised)."""
    d = get_destination(destination)
    if not d["normalize"] or d["target_lufs"] is None or measured_lufs is None or not math.isfinite(measured_lufs):
        return 0.0
    return float(d["target_lufs"] - measured_lufs)


def list_destinations() -> list[dict]:
    return [dict(name=k, **v) for k, v in DESTINATIONS.items()]
