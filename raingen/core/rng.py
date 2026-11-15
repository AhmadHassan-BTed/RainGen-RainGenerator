"""Deterministic, counter-based random streams.

Every random draw in the engine comes from `RngFactory.get(layer, block)`.
Streams are keyed by a stable hash of the *layer name* (never creation order),
and indexed by block number through the Philox counter. Adding or removing a
layer therefore never changes any other layer, and the same (seed, layer,
block) always yields the same numbers on every platform.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _layer_key(root_seed: int, layer: str) -> np.ndarray:
    h = hashlib.blake2b(f"{root_seed & ((1 << 128) - 1):032x}|{layer}".encode(), digest_size=16).digest()
    return np.frombuffer(h, dtype=np.uint64).copy()  # 2 x uint64 Philox key


class RngFactory:
    """Factory of reproducible numpy Generators keyed by (seed, layer, block)."""

    def __init__(self, root_seed: int):
        self.root_seed = int(root_seed)
        self._keys: dict[str, np.ndarray] = {}

    def key(self, layer: str) -> np.ndarray:
        k = self._keys.get(layer)
        if k is None:
            k = _layer_key(self.root_seed, layer)
            self._keys[layer] = k
        return k

    def get(self, layer: str, block: int = 0) -> np.random.Generator:
        """Independent generator for `layer` at block index `block`."""
        counter = np.array([0, int(block) & 0xFFFFFFFFFFFFFFFF, 0, 0], dtype=np.uint64)
        return np.random.Generator(np.random.Philox(key=self.key(layer), counter=counter))

    def child(self, layer: str) -> "RngFactory":
        """A factory whose seed is derived from this one (e.g. for variations)."""
        k = self.key(layer)
        return RngFactory(int(k[0]) ^ (int(k[1]) << 64))


def new_seed() -> int:
    """A fresh random 64-bit seed (for 'surprise me' / new scenes)."""
    return int.from_bytes(np.random.default_rng().bytes(8), "little")
