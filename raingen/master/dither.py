"""TPDF dither and requantisation for integer exports.

Triangular-PDF dither of +-1 LSB peak (sum of two independent uniform +-0.5 LSB variables)
makes the first two moments of the requantisation error independent of the signal
(Lipshitz, Wannamaker & Vanderkooy 1992; Wannamaker 1997). 'hp' uses the difference of
successive uniform draws (high-pass TPDF), which moves the dither power up in frequency
with the same total TPDF statistics. Dither has its own deterministic RNG stream
(`master.dither`), so a render's samples never depend on whether it was dithered.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.rng import RngFactory


def lsb(bits: int) -> float:
    """Size of one least-significant bit for full scale +-1.0."""
    return 2.0 ** (1 - int(bits))


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Round to the `bits` integer grid (as float, clipped to the representable range)."""
    q = lsb(bits)
    return np.clip(np.round(np.asarray(x, dtype=np.float64) / q) * q, -1.0, 1.0 - q)


def tpdf_dither(x: np.ndarray, bits: int, rng: np.random.Generator, quantize_output: bool = True) -> np.ndarray:
    """Add +-1 LSB TPDF dither for a `bits`-bit target and (by default) requantise.

    bits >= 32 (float output) returns the signal unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    if int(bits) >= 32:
        return x.copy()
    q = lsb(bits)
    noise = (rng.random(x.shape) - rng.random(x.shape)) * q
    y = x + noise
    if quantize_output:
        y = np.clip(np.round(y / q) * q, -1.0, 1.0 - q)
    return y


class Ditherer:
    """Block-wise TPDF ditherer drawing from RngFactory.get(layer, block_index)."""

    def __init__(self, bits: int, rngf: RngFactory, layer: str = "master.dither", shape: str = "flat",
                 quantize_output: bool = True):
        self.bits = int(bits)
        self.rngf = rngf
        self.layer = layer
        self.shape = shape
        self.quantize_output = quantize_output
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def process(self, x: np.ndarray, block_index: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.bits >= 32:
            return x.copy()
        rng = self.rngf.get(self.layer, block_index)
        if self.shape != "hp":
            return tpdf_dither(x, self.bits, rng, self.quantize_output)
        q = lsb(self.bits)
        x2 = np.atleast_2d(x)
        r = rng.random(x2.shape) - 0.5
        prev = self._prev if self._prev is not None and self._prev.shape[0] == x2.shape[0] else \
            np.zeros(x2.shape[0])
        r_shift = np.concatenate([prev[:, None], r[:, :-1]], axis=1)
        self._prev = r[:, -1].copy()
        y = x2 + (r - r_shift) * q
        if self.quantize_output:
            y = np.clip(np.round(y / q) * q, -1.0, 1.0 - q)
        return y[0] if x.ndim == 1 else y
