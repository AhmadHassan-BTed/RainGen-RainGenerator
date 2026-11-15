"""
Procedural audio synthesis and thunder sound playback.

Since v3 the file renderers (`generate_storm_audio`, `export_audio_file`) delegate to the
physically based raingen engine; the legacy presets map onto raingen presets (see
LEGACY_TO_RAINGEN) and the old synthesizer remains as a fallback. When no thunder WAV clips
are found, `SoundEngine.play_thunder` uses thunder synthesized by raingen.
"""

import math
import os
import time
import random
import logging
import threading
from typing import List, Optional, Callable
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
import pygame

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

logger = logging.getLogger(__name__)

def lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter to signal."""
    nyquist = 0.5 * fs
    normal_cutoff = max(0.01, min(0.99, cutoff / nyquist))
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 2) -> np.ndarray:
    """Apply Butterworth band-pass filter to signal."""
    nyquist = 0.5 * fs
    low = max(0.01, min(0.98, lowcut / nyquist))
    high = max(low + 0.01, min(0.99, highcut / nyquist))
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

class ProceduralRainSynthesizer:
    """Synthesizes rain noise, individual drop impacts, and wind sound.

    Every call continues the previous one: the filters keep their state between blocks
    (scipy.signal.sosfilt with zi stored on the instance), drop sounds that start near the
    end of a block ring on into the next, and the gust LFO keeps its phase. Restarting the
    filters from rest on every audio callback produced an audible flutter at the block rate.
    """

    _MAX_DROP_LEN = 500

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.rain_cutoff = 1900.0
        self.rain_scale = 1.0
        self.drop_rate = 1.0
        self.wind_level = 0.0
        self._wind_phase = 0.0
        self._rng = np.random.default_rng()
        self._rain_key = None
        self._rain_sos: Optional[np.ndarray] = None
        self._rain_zi: Optional[np.ndarray] = None
        self._wind_key = None
        self._wind_sos: Optional[np.ndarray] = None
        self._wind_zi: Optional[np.ndarray] = None
        self._drop_tail = np.zeros(0, dtype=np.float32)

    def _rain_filter(self) -> np.ndarray:
        key = (round(float(self.rain_cutoff), 3), int(self.sample_rate))
        if key != self._rain_key:
            nyquist = 0.5 * self.sample_rate
            wn = max(0.01, min(0.99, self.rain_cutoff / nyquist))
            sos = butter(4, wn, btype="low", output="sos")
            if self._rain_zi is None or self._rain_zi.shape[0] != sos.shape[0]:
                self._rain_zi = np.zeros((sos.shape[0], 2))
            self._rain_sos, self._rain_key = sos, key
        return self._rain_sos

    def _wind_filter(self, lowcut: float, highcut: float) -> np.ndarray:
        key = (round(lowcut, 1), round(highcut, 1), int(self.sample_rate))
        if key != self._wind_key:
            nyquist = 0.5 * self.sample_rate
            low = max(0.01, min(0.98, lowcut / nyquist))
            high = max(low + 0.01, min(0.99, highcut / nyquist))
            sos = butter(2, [low, high], btype="band", output="sos")
            if self._wind_zi is None or self._wind_zi.shape[0] != sos.shape[0]:
                self._wind_zi = np.zeros((sos.shape[0], 2))
            self._wind_sos, self._wind_key = sos, key
        return self._wind_sos

    def generate_rain_noise(self, length: int, intensity: float = 1.0) -> np.ndarray:
        noise = self._rng.normal(0.0, 0.22 * intensity, length)
        sos = self._rain_filter()
        filtered, self._rain_zi = sosfilt(sos, noise, zi=self._rain_zi)
        return filtered.astype(np.float32)

    def generate_drop_impacts(self, length: int, rate_multiplier: float = 1.0) -> np.ndarray:
        """Synthesizes raindrop impacts using decaying sinusoids (tails carry into the next block)."""
        max_len = self._MAX_DROP_LEN
        buffer = np.zeros(length + max_len, dtype=np.float32)
        tail = self._drop_tail
        if tail.size:
            buffer[:tail.size] += tail
        expected = max(0.0, 16.0 * rate_multiplier * (length / self.sample_rate))
        drop_count = int(self._rng.poisson(expected)) if expected > 0.0 and length > 0 else 0
        if drop_count > 0:
            positions = self._rng.integers(0, max(1, length), drop_count)
            for pos in positions:
                duration = int(self._rng.integers(180, max_len))
                t = np.arange(duration) / self.sample_rate
                freq = self._rng.uniform(1600, 3800)
                decay = np.exp(-t * self._rng.uniform(40, 100))
                amplitude = self._rng.uniform(0.04, 0.16)
                buffer[pos:pos + duration] += (np.sin(2 * np.pi * freq * t) * decay * amplitude).astype(np.float32)
        self._drop_tail = buffer[length:].copy()
        return buffer[:length].copy()

    def generate_wind_noise(self, length: int, wind_strength: float = 0.5) -> np.ndarray:
        """Generates wind noise using bandpassed noise and low-frequency modulation."""
        phase_step = 2 * np.pi * 0.4 * (length / self.sample_rate)
        if wind_strength <= 0.05:
            self._wind_phase = (self._wind_phase + phase_step) % (2 * np.pi)
            if self._wind_zi is not None:
                self._wind_zi[:] = 0.0          # the next gust starts from rest
            return np.zeros(length, dtype=np.float32)

        noise = self._rng.normal(0.0, 0.18, length)
        center_freq = 280.0 + (wind_strength * 220.0)
        lowcut = max(60.0, center_freq - 140.0)
        highcut = min(1200.0, center_freq + 180.0)

        sos = self._wind_filter(lowcut, highcut)
        wind_band, self._wind_zi = sosfilt(sos, noise, zi=self._wind_zi)

        t = np.arange(length) / self.sample_rate
        gust = 0.7 + 0.3 * np.sin(2 * np.pi * 0.4 * t + self._wind_phase)
        self._wind_phase = (self._wind_phase + phase_step) % (2 * np.pi)

        return (wind_band * gust * min(1.5, wind_strength)).astype(np.float32)

class SoundEngine:
    """Manages audio streams, thunder playback, and channel volume mixing."""

    def __init__(self, sounds_dir: Optional[str] = None):
        self.sample_rate = 44100
        self.block_size = 2048
        self.synthesizer = ProceduralRainSynthesizer(self.sample_rate)

        self._stream: Optional[sd.OutputStream] = None if HAS_SOUNDDEVICE else None
        self._stream_running = False
        self._audio_available = HAS_SOUNDDEVICE
        self._lock = threading.Lock()

        self.master_volume = 0.8
        self.rain_volume = 0.6
        self.drops_volume = 0.45
        self.wind_volume = 0.35
        self.thunder_volume = 0.7
        self.is_muted = False

        self.rain_intensity_scale = 1.0
        self.drop_rate_scale = 1.0
        self.wind_speed = 0.0

        self._delayed_thunders: List[dict] = []
        self._delayed_lock = threading.Lock()

        # raingen-synthesized thunder, used only when no WAV clips are found
        self._synth_thunder_state = "idle"          # idle | running | ready | failed
        self._pending_synth: List[tuple] = []

        self.thunder_sounds: List[pygame.mixer.Sound] = []
        self._init_mixer(sounds_dir)

    def _init_mixer(self, sounds_dir: Optional[str] = None):
        """Initializes pygame.mixer and loads available thunder sound clips."""
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            except Exception as e:
                logger.warning(f"Could not initialize pygame.mixer: {e}")
                return

        candidates = []
        if sounds_dir and os.path.isdir(sounds_dir):
            candidates.append(sounds_dir)
        candidates.extend([
            os.path.join(os.path.dirname(__file__), "..", "sounds"),
            os.path.join(os.getcwd(), "sounds"),
            "sounds"
        ])

        found_dir = None
        for candidate in candidates:
            norm = os.path.normpath(candidate)
            if os.path.isdir(norm):
                found_dir = norm
                break

        if found_dir:
            for i in range(1, 11):
                path = os.path.join(found_dir, f"thunder_{i}.wav")
                if os.path.exists(path):
                    try:
                        sound = pygame.mixer.Sound(path)
                        self.thunder_sounds.append(sound)
                    except Exception as err:
                        logger.warning(f"Failed to load thunder sound {path}: {err}")
        else:
            logger.warning("No sounds directory found for thunder clips.")

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Output stream callback."""
        if status:
            logger.debug(f"Audio stream status: {status}")

        if self.is_muted or self.master_volume <= 0.001:
            outdata.fill(0)
            return

        with self._lock:
            eff_rain = self.rain_volume * self.master_volume
            eff_drops = self.drops_volume * self.master_volume
            eff_wind = self.wind_volume * self.master_volume
            rain_scale = self.rain_intensity_scale
            drop_scale = self.drop_rate_scale
            wind_speed = abs(self.wind_speed)

        rain = self.synthesizer.generate_rain_noise(frames, rain_scale) * eff_rain
        drops = self.synthesizer.generate_drop_impacts(frames, drop_scale) * eff_drops
        wind = self.synthesizer.generate_wind_noise(frames, wind_speed) * eff_wind

        mixed = rain + drops + wind
        mixed = np.tanh(mixed)

        if outdata.shape[1] == 1:
            outdata[:] = mixed.reshape(-1, 1)
        elif outdata.shape[1] == 2:
            outdata[:, 0] = mixed
            outdata[:, 1] = mixed
        else:
            outdata[:] = np.repeat(mixed.reshape(-1, 1), outdata.shape[1], axis=1)

    def start(self):
        """Starts audio output stream."""
        if not HAS_SOUNDDEVICE or self._stream_running:
            return

        try:
            self._stream = sd.OutputStream(
                channels=2,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self._audio_callback,
                latency="high"
            )
            self._stream.start()
            self._stream_running = True
        except Exception as e:
            logger.warning(f"Could not open sounddevice stream: {e}")
            self._audio_available = False
            self._stream_running = False

    def stop(self):
        """Stops audio stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
                self._stream_running = False

    def update_parameters(self,
                          master_vol: float,
                          rain_vol: float,
                          drops_vol: float,
                          wind_vol: float,
                          thunder_vol: float,
                          is_muted: bool,
                          rain_intensity: float,
                          drop_rate: float,
                          wind_speed: float):
        """Updates volume and storm parameters."""
        with self._lock:
            self.master_volume = max(0.0, min(1.0, master_vol))
            self.rain_volume = max(0.0, min(1.0, rain_vol))
            self.drops_volume = max(0.0, min(1.0, drops_vol))
            self.wind_volume = max(0.0, min(1.0, wind_vol))
            self.thunder_volume = max(0.0, min(1.0, thunder_vol))
            self.is_muted = is_muted
            self.rain_intensity_scale = max(0.1, rain_intensity)
            self.drop_rate_scale = max(0.1, drop_rate)
            self.wind_speed = wind_speed

    def _ensure_synth_thunder(self):
        """Starts rendering raingen thunder one-shots in the background (once)."""
        if self._synth_thunder_state != "idle":
            return
        if not pygame.mixer.get_init():
            self._synth_thunder_state = "failed"
            return
        self._synth_thunder_state = "running"
        threading.Thread(target=self._synthesize_thunder, name="raingen-thunder", daemon=True).start()

    def _synthesize_thunder(self):
        try:
            from raingen.render import thunder_one_shots

            freq, _fmt, channels = pygame.mixer.get_init()
            clips = thunder_one_shots(count=3, distance_km=2.5, sample_rate=int(freq), layout="stereo")
            sounds = []
            for clip in clips:
                stereo = clip if clip.shape[0] >= 2 else np.vstack([clip[0], clip[0]])
                pcm = np.clip(np.round(stereo[:2].T * 32767.0), -32768, 32767).astype(np.int16)
                if channels == 1:
                    pcm = pcm.mean(axis=1).astype(np.int16)
                elif channels > 2:
                    pcm = np.column_stack([pcm[:, i % 2] for i in range(channels)])
                sounds.append(pygame.sndarray.make_sound(np.ascontiguousarray(pcm)))
            if not sounds:
                raise RuntimeError("no thunder was rendered")
            self.thunder_sounds.extend(sounds)
            self._synth_thunder_state = "ready"
            logger.info("Using %d raingen-synthesized thunder clips", len(sounds))
        except Exception as e:
            logger.warning(f"Synthesized thunder unavailable: {e}")
            self._synth_thunder_state = "failed"
            with self._delayed_lock:
                self._pending_synth = []

    def play_thunder(self, delay: float = 0.0, volume_scale: float = 1.0):
        """Plays a thunder clip, optionally queued with a delay.

        Without WAV clips, thunder synthesized by raingen is used; the first request starts
        the synthesis in the background and is played as soon as the clips are ready.
        """
        if not self.thunder_sounds:
            self._ensure_synth_thunder()
            if not self.thunder_sounds:
                if self._synth_thunder_state == "running":
                    with self._delayed_lock:
                        self._pending_synth.append((time.time() + max(0.0, delay), volume_scale))
                return

        sound = random.choice(self.thunder_sounds)
        eff_vol = 0.0 if self.is_muted else (self.thunder_volume * self.master_volume * volume_scale)
        eff_vol = max(0.0, min(1.0, eff_vol))

        if delay <= 0.05:
            sound.set_volume(eff_vol)
            sound.play()
        else:
            with self._delayed_lock:
                self._delayed_thunders.append({
                    "trigger_time": time.time() + delay,
                    "sound": sound,
                    "volume": eff_vol
                })

    def process_delayed_thunders(self):
        """Fires any queued thunder clips whose trigger time has elapsed."""
        if self._pending_synth and self.thunder_sounds:
            with self._delayed_lock:
                pending, self._pending_synth = self._pending_synth, []
            now = time.time()
            for trigger, volume_scale in pending:
                if now - trigger <= 5.0:         # drop requests that went stale while rendering
                    self.play_thunder(delay=max(0.0, trigger - now), volume_scale=volume_scale)

        if not self._delayed_thunders:
            return

        now = time.time()
        with self._delayed_lock:
            remaining = []
            for item in self._delayed_thunders:
                if now >= item["trigger_time"]:
                    sound = item["sound"]
                    vol = 0.0 if self.is_muted else min(1.0, item["volume"] * (self.master_volume / max(0.01, self.master_volume)))
                    sound.set_volume(vol)
                    sound.play()
                else:
                    remaining.append(item)
            self._delayed_thunders = remaining

# ---------------------------------------------------------------------------
# File rendering: delegated to raingen (physically based), legacy synth as fallback
# ---------------------------------------------------------------------------

# Legacy preset -> raingen preset (raingen preset names are accepted as they are).
LEGACY_TO_RAINGEN = {
    "drizzle": "scottish-highland-drizzle",
    "steady": "steady-rain-leaves-night",
    "thunderstorm": "tin-roof-cabin-storm",
    "gale": "conifer-forest-gale",
    "cozy": "window-rain-bedroom",
}
_DEFAULT_VOLUMES = {"master": 0.8, "rain": 0.6, "drops": 0.45, "wind": 0.35, "thunder": 0.7}
LEGACY_TARGET_LUFS = -18.0          # programme loudness at the default master volume


def _raingen_renderer(duration: float, preset_name: str, sample_rate: int, volumes: Optional[dict],
                      include_thunder: bool, bit_depth: int):
    """A raingen Renderer for a legacy (or raingen) preset name, or None for a muted master."""
    from raingen.render import Renderer
    from raingen.scene.presets import PRESETS as RAINGEN_PRESETS, preset as raingen_preset

    key = LEGACY_TO_RAINGEN.get(preset_name, str(preset_name).strip().lower().replace("_", "-"))
    if key not in RAINGEN_PRESETS:
        key = LEGACY_TO_RAINGEN["steady"]
    vol = dict(_DEFAULT_VOLUMES)
    vol.update(volumes or {})

    def rel_db(name: str) -> Optional[float]:
        v = float(vol.get(name, _DEFAULT_VOLUMES[name]))
        return None if v <= 1e-4 else 20.0 * math.log10(v / _DEFAULT_VOLUMES[name])

    master = float(vol.get("master", _DEFAULT_VOLUMES["master"]))
    if master <= 1e-4 or duration <= 0:
        return None
    mix = {
        "rain_far": rel_db("rain"),
        "rain_near": rel_db("drops"),
        "aftermath": rel_db("drops"),
        "wind": rel_db("wind"),
        "thunder": rel_db("thunder") if include_thunder else None,
    }
    target = LEGACY_TARGET_LUFS + 20.0 * math.log10(master / _DEFAULT_VOLUMES["master"])
    return Renderer(raingen_preset(key), float(duration), layout="stereo", destination="raw",
                    sample_rate=int(sample_rate), bit_depth=int(bit_depth), mix=mix,
                    target_lufs=target, ceiling_dbtp=-1.0, quality="game")


def generate_storm_audio(
    duration: float,
    preset_name: str = "steady",
    sample_rate: int = 44100,
    volumes: Optional[dict] = None,
    sounds_dir: Optional[str] = None,
    include_thunder: bool = True
) -> np.ndarray:
    """
    Generates a continuous storm audio track of specified duration in seconds.
    Returns stereo int16 audio array of shape (N, 2).

    Rendered by raingen from the matching physical preset (LEGACY_TO_RAINGEN), loudness
    normalised to about -18 LUFS at the default master volume. Falls back to the legacy
    synthesizer if raingen cannot render. `sounds_dir` is used by the fallback only.
    """
    total_samples = int(duration * sample_rate)
    if total_samples <= 0:
        return np.zeros((0, 2), dtype=np.int16)
    try:
        renderer = _raingen_renderer(duration, preset_name, sample_rate, volumes, include_thunder, 16)
        if renderer is None:
            return np.zeros((total_samples, 2), dtype=np.int16)
        audio, _markers = renderer.to_array()
        out = np.zeros((total_samples, 2), dtype=np.float64)
        m = min(total_samples, audio.shape[1])
        out[:m] = audio[:2, :m].T
        return np.clip(np.round(out * 32767.0), -32768, 32767).astype(np.int16)
    except Exception as err:
        logger.warning(f"raingen rendering unavailable ({err}); using the legacy synthesizer")
        return _legacy_generate_storm_audio(duration, preset_name, sample_rate, volumes, sounds_dir,
                                            include_thunder)


def _legacy_generate_storm_audio(
    duration: float,
    preset_name: str = "steady",
    sample_rate: int = 44100,
    volumes: Optional[dict] = None,
    sounds_dir: Optional[str] = None,
    include_thunder: bool = True
) -> np.ndarray:
    """
    The v2 renderer (filtered noise, sine drops, thunder WAV clips): stereo int16 (N, 2).
    """
    import scipy.io.wavfile as wavfile
    from .presets import PRESETS

    preset = PRESETS.get(preset_name, PRESETS["steady"])
    vol_dict = {
        "master": 0.8,
        "rain": 0.6,
        "drops": 0.45,
        "wind": 0.35,
        "thunder": 0.7
    }
    if volumes:
        vol_dict.update(volumes)

    total_samples = int(duration * sample_rate)
    synth = ProceduralRainSynthesizer(sample_rate=sample_rate)

    eff_rain = vol_dict["rain"] * vol_dict["master"]
    eff_drops = vol_dict["drops"] * vol_dict["master"]
    eff_wind = vol_dict["wind"] * vol_dict["master"]
    eff_thunder = vol_dict["thunder"] * vol_dict["master"]

    rain = synth.generate_rain_noise(total_samples, preset.rain_audio_intensity) * eff_rain
    drops = synth.generate_drop_impacts(total_samples, preset.drop_audio_rate) * eff_drops
    wind = synth.generate_wind_noise(total_samples, preset.wind_audio_intensity) * eff_wind

    mono_ambient = rain + drops + wind
    stereo = np.column_stack((mono_ambient, mono_ambient)).astype(np.float32)

    # Layer thunder clips
    if include_thunder and eff_thunder > 0.01:
        candidates = []
        if sounds_dir and os.path.isdir(sounds_dir):
            candidates.append(sounds_dir)
        candidates.extend([
            os.path.join(os.path.dirname(__file__), "..", "sounds"),
            os.path.join(os.getcwd(), "sounds"),
            "sounds"
        ])
        found_dir = None
        for c in candidates:
            norm = os.path.normpath(c)
            if os.path.isdir(norm):
                found_dir = norm
                break

        t_files = []
        if found_dir:
            for i in range(1, 11):
                p = os.path.join(found_dir, f"thunder_{i}.wav")
                if os.path.exists(p):
                    t_files.append(p)

        if t_files:
            current_time = random.uniform(*preset.thunder_interval_range)
            while current_time < duration:
                start_sample = int(current_time * sample_rate)
                clip_path = random.choice(t_files)
                try:
                    tsr, tdata = wavfile.read(clip_path)
                    tfloat = tdata.astype(np.float32) / 32768.0
                    if len(tfloat.shape) == 1:
                        tfloat = np.column_stack((tfloat, tfloat))
                    clip_len = min(len(tfloat), total_samples - start_sample)
                    if clip_len > 0:
                        stereo[start_sample:start_sample + clip_len] += tfloat[:clip_len] * eff_thunder
                except Exception as err:
                    logger.debug(f"Could not load thunder clip {clip_path}: {err}")

                current_time += random.uniform(*preset.thunder_interval_range)

    # Soft clipping / saturation
    stereo = np.tanh(stereo)
    int16_audio = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    return int16_audio

def export_audio_file(
    filepath: str,
    duration: float,
    preset_name: str = "steady",
    sample_rate: int = 44100,
    volumes: Optional[dict] = None,
    sounds_dir: Optional[str] = None
) -> str:
    """
    Generates storm audio of specified duration and saves to a WAV file.

    Streams a raingen render straight to disk (16-bit Broadcast WAV with the scene, seed and
    engine version embedded), so any length works without holding the audio in memory.
    Returns the absolute path written. Falls back to the legacy in-memory renderer.
    """
    import scipy.io.wavfile as wavfile

    try:
        renderer = _raingen_renderer(duration, preset_name, sample_rate, volumes, True, 16)
        if renderer is not None:
            report = renderer.run(filepath)
            return report["path"]
    except Exception as err:
        logger.warning(f"raingen export unavailable ({err}); using the legacy synthesizer")

    audio_data = _legacy_generate_storm_audio(
        duration=duration,
        preset_name=preset_name,
        sample_rate=sample_rate,
        volumes=volumes,
        sounds_dir=sounds_dir
    )
    wavfile.write(filepath, sample_rate, audio_data)
    return os.path.abspath(filepath)
