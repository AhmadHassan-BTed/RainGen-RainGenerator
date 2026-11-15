"""Decoders: SN3D/ACN bus (N, n) -> output channels (C, n).

* mono      W (the pressure signal).
* stereo    virtual-microphone pair fitted (regularised least squares, i.e. mode matching on a
            dense grid) to a constant-power panning law whose image spans +-30 deg loudspeakers,
            with a `width` control and a gentle rear attenuation.
* 5.1, 7.1, 7.1.4
            AllRAD (Zotter & Frank 2012, JAES 60(10)): a max-rE sampling decoder onto a dense
            near-uniform virtual layout, then VBAP (Pulkki 1997) from every virtual point to
            the real loudspeakers (imaginary zenith/nadir speakers fill the hull; zenith energy
            is shared, nadir discarded). Energy-normalised so the direction-averaged sum of
            squared gains is 1. Speaker angles follow ITU-R BS.775 / BS.2051.
* foa, hoaN pass-through (truncate or zero-pad), AmbiX.
* binaural  virtual loudspeakers (26-point Lebedev for N <= 3, else a dense Fibonacci set)
            with HRIRs from `hrtf.py` (built-in Brown-Duda head or SOFA), folded into SH-domain
            filters with max-rE above ~2 kHz and the diffuse-field covariance constraint of
            Zaunschirm, Schoerkhuber & Hoeldrich (2018), JASA 143(6):3616, which restores the
            interaural coherence and diffuse-field timbre of the reference HRTF set
            (optimal 2x2 mixing per bin, Vilkamo, Baeckstroem & Kuntz 2013).
            Block FFT overlap-add carries the filter tails across blocks.

Channel order is the WAVEFORMATEXTENSIBLE bit order (so files are correct with their
channel mask): 7.1 = L R C LFE Lrs Rrs Lss Rss; pass `order_style='smpte'` for
L R C LFE Lss Rss Lrs Rrs.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.constants import SAMPLE_RATE
from .sh import (acn_order_index, dense_grid, fibonacci_sphere, lebedev26, maxre_energy_compensation,
                 maxre_weights, n_channels, sh_real, unit_vectors)

SPEAKER_LAYOUTS: dict[str, list[tuple[str, Optional[float], float]]] = {
    "5.1": [("L", 30.0, 0.0), ("R", -30.0, 0.0), ("C", 0.0, 0.0), ("LFE", None, 0.0),
            ("Ls", 110.0, 0.0), ("Rs", -110.0, 0.0)],
    "7.1": [("L", 30.0, 0.0), ("R", -30.0, 0.0), ("C", 0.0, 0.0), ("LFE", None, 0.0),
            ("Lrs", 135.0, 0.0), ("Rrs", -135.0, 0.0), ("Lss", 90.0, 0.0), ("Rss", -90.0, 0.0)],
    "7.1.4": [("L", 30.0, 0.0), ("R", -30.0, 0.0), ("C", 0.0, 0.0), ("LFE", None, 0.0),
              ("Lrs", 135.0, 0.0), ("Rrs", -135.0, 0.0), ("Lss", 90.0, 0.0), ("Rss", -90.0, 0.0),
              ("Ltf", 45.0, 45.0), ("Rtf", -45.0, 45.0), ("Ltr", 135.0, 45.0), ("Rtr", -135.0, 45.0)],
}

CHANNEL_MASKS = {
    "mono": 0x4,
    "stereo": 0x3,
    "binaural": 0x3,
    "5.1": 0x60F,
    "7.1": 0x63F,
    "7.1.4": 0x2D63F,
}

_SMPTE_ORDER = {
    "7.1": ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs"],
    "7.1.4": ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs", "Ltf", "Rtf", "Ltr", "Rtr"],
}

_ALIASES = {
    "2.0": "stereo", "lr": "stereo", "headphones": "binaural", "hrtf": "binaural",
    "ambix": "foa", "b-format": "foa", "bformat": "foa", "1oa": "foa", "hoa1": "foa",
    "5.1.0": "5.1", "7.1.0": "7.1", "1.0": "mono", "hoa": "hoa3",
}


def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


def normalize_layout(layout: str) -> str:
    lay = str(layout).strip().lower().replace(" ", "")
    return _ALIASES.get(lay, lay)


def channel_mask_for(layout: str, channels: Optional[int] = None) -> int:
    """WAVEFORMATEXTENSIBLE dwChannelMask for a layout (0 = no speaker mapping, e.g. ambisonics)."""
    lay = normalize_layout(layout)
    if lay in CHANNEL_MASKS:
        return CHANNEL_MASKS[lay]
    return 0


def _fit_bus(bus, N: int) -> np.ndarray:
    b = np.asarray(bus, dtype=np.float64)
    if b.ndim == 1:
        b = b[None, :]
    if b.shape[0] == N:
        return b
    out = np.zeros((N, b.shape[1]))
    c = min(N, b.shape[0])
    out[:c] = b[:c]
    return out


class Decoder:
    """Base class. Subclasses set channels/channel_names and implement process()."""

    layout: str = ""
    order: int = 1
    sr: int = SAMPLE_RATE
    channels: int = 1
    channel_names: list[str] = []
    channel_mask: int = 0
    latency: int = 0

    def process(self, bus: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.layout} order={self.order} channels={self.channels}>"


class MatrixDecoder(Decoder):
    """Static decoding matrix D (C, N); optional low-passed W feed into an LFE channel."""

    def __init__(self, layout: str, order: int, sr: int, matrix: np.ndarray, names: list[str],
                 mask: int = 0, lfe_index: Optional[int] = None, lfe_gain: float = 0.0,
                 lfe_cutoff_hz: float = 120.0):
        self.layout = layout
        self.order = int(order)
        self.sr = int(sr)
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.channels = self.matrix.shape[0]
        self.channel_names = list(names)
        self.channel_mask = int(mask)
        self._N = n_channels(self.order)
        self._lfe_index = lfe_index
        self._lfe_gain = float(lfe_gain)
        self._lfe_sos = None
        self._lfe_zi = None
        if lfe_index is not None and self._lfe_gain > 0.0:
            from scipy.signal import butter
            sos = butter(2, lfe_cutoff_hz, btype="lowpass", fs=self.sr, output="sos")
            self._lfe_sos = np.concatenate([sos, sos], axis=0)  # Linkwitz-Riley 4th order
            self._lfe_zi = np.zeros((self._lfe_sos.shape[0], 2))

    def process(self, bus: np.ndarray) -> np.ndarray:
        b = _fit_bus(bus, self._N)
        out = self.matrix @ b
        if self._lfe_sos is not None:
            from scipy.signal import sosfilt
            lfe, self._lfe_zi = sosfilt(self._lfe_sos, b[0], zi=self._lfe_zi)
            out[self._lfe_index] += self._lfe_gain * lfe
        return out

    def reset(self) -> None:
        if self._lfe_zi is not None:
            self._lfe_zi[:] = 0.0


class PassthroughDecoder(Decoder):
    """Ambisonic output (AmbiX ACN/SN3D), truncated or zero-padded to `out_order`."""

    def __init__(self, layout: str, order: int, sr: int, out_order: int):
        self.layout = layout
        self.order = int(order)
        self.sr = int(sr)
        self.out_order = int(out_order)
        self.channels = n_channels(self.out_order)
        if self.out_order == 1:
            self.channel_names = ["W", "Y", "Z", "X"]
        else:
            self.channel_names = [f"ACN{k}" for k in range(self.channels)]
        self.channel_mask = 0

    def process(self, bus: np.ndarray) -> np.ndarray:
        return _fit_bus(bus, self.channels).copy()


# ---------------------------------------------------------------------------
# design helpers
# ---------------------------------------------------------------------------
def _energy_normalize(D: np.ndarray, order: int) -> np.ndarray:
    az, el = dense_grid(order)
    G = D @ sh_real(order, az, el)
    e = float(np.mean(np.sum(G * G, axis=0)))
    return D / math.sqrt(e) if e > 0 else D


def stereo_matrix(order: int, width: float = 1.0, rear_atten_db: float = 3.0, reg: float = 1e-6) -> np.ndarray:
    """Least-squares virtual-microphone pair (2, N) matching a constant-power stereo panning law."""
    az, el = dense_grid(order)
    u = unit_vectors(az, el)
    q = np.clip(float(width) * u[:, 1], -1.0, 1.0)
    den = np.sqrt(2.0 * (1.0 + q * q))
    rear = 10.0 ** (-(rear_atten_db / 20.0) * (1.0 - u[:, 0]) / 2.0)
    G = np.vstack([(1.0 + q) / den, (1.0 - q) / den]) * rear[None, :]
    Y = sh_real(order, az, el)                       # (N, P)
    A = Y @ Y.T
    lam = reg * np.trace(A) / A.shape[0]
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), Y @ G.T).T


def vbap_gains(speakers_xyz: np.ndarray, dirs_xyz: np.ndarray) -> np.ndarray:
    """3-D VBAP gains (L, P), power-normalised, over the convex hull of the speaker directions."""
    from scipy.spatial import ConvexHull

    spk = np.asarray(speakers_xyz, dtype=np.float64)
    dirs = np.asarray(dirs_xyz, dtype=np.float64)
    hull = ConvexHull(spk)
    tris = []
    invs = []
    for simplex in hull.simplices:
        M = spk[simplex]                            # rows = speaker vectors
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        tris.append(simplex)
        invs.append(np.linalg.inv(M))
    tris = np.array(tris)
    invs = np.array(invs)                           # (T, 3, 3)
    g = np.einsum("pj,tjk->ptk", dirs, invs)        # p = g M  ->  g = p M^-1
    best = np.argmax(g.min(axis=2), axis=1)
    gsel = np.maximum(g[np.arange(dirs.shape[0]), best], 0.0)
    gsel /= np.maximum(np.linalg.norm(gsel, axis=1, keepdims=True), 1e-12)
    G = np.zeros((spk.shape[0], dirs.shape[0]))
    cols = np.repeat(np.arange(dirs.shape[0]), 3)
    np.add.at(G, (tris[best].reshape(-1), cols), gsel.reshape(-1))
    return G


def allrad_matrix(order: int, az_deg, el_deg) -> np.ndarray:
    """AllRAD decoder (L, N) for loudspeakers at the given angles (degrees)."""
    az = np.radians(np.asarray(az_deg, dtype=np.float64))
    el = np.radians(np.asarray(el_deg, dtype=np.float64))
    real = unit_vectors(az, el)
    L = real.shape[0]
    spk = [real]
    zenith_idx = None
    max_el, min_el = float(np.max(el_deg)), float(np.min(el_deg))
    if max_el < 20.0:
        spk.append(np.array([[0.0, 0.0, 1.0]]))
        zenith_idx = L
    if min_el > -20.0:
        spk.append(np.array([[0.0, 0.0, -1.0]]))
    allspk = np.concatenate(spk, axis=0)
    gaz, gel = dense_grid(order)
    grid = unit_vectors(gaz, gel)
    G = vbap_gains(allspk, grid)
    Gr = G[:L].copy()
    if zenith_idx is not None:
        Gr += G[zenith_idx][None, :] / math.sqrt(L)
    nidx = acn_order_index(order)
    w = ((2.0 * nidx + 1.0) * maxre_weights(order)[nidx]) / grid.shape[0]
    Ydec = sh_real(order, gaz, gel).T * w[None, :]   # (P, N)
    return _energy_normalize(Gr @ Ydec, order)


def _speaker_decoder(layout: str, order: int, sr: int, order_style: str = "wav",
                     lfe_gain: float = 0.0) -> MatrixDecoder:
    spk = SPEAKER_LAYOUTS[layout]
    names = [s[0] for s in spk]
    active = [i for i, s in enumerate(spk) if s[1] is not None]
    D_act = allrad_matrix(order, [spk[i][1] for i in active], [spk[i][2] for i in active])
    D = np.zeros((len(spk), n_channels(order)))
    D[active] = D_act
    lfe = names.index("LFE") if "LFE" in names else None
    mask = CHANNEL_MASKS.get(layout, 0)
    if order_style == "smpte" and layout in _SMPTE_ORDER:
        perm = [names.index(nm) for nm in _SMPTE_ORDER[layout]]
        D = D[perm]
        names = [names[i] for i in perm]
        lfe = names.index("LFE")
    return MatrixDecoder(layout, order, sr, D, names, mask=mask, lfe_index=lfe, lfe_gain=lfe_gain)


# ---------------------------------------------------------------------------
# binaural
# ---------------------------------------------------------------------------
def _smooth_band(freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """0 below f_lo, 1 above f_hi, raised cosine in log-frequency between."""
    f = np.maximum(freqs, 1e-3)
    t = np.clip(np.log(f / f_lo) / math.log(f_hi / f_lo), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * t)


def _smooth_matrix_bins(M: np.ndarray, freqs: np.ndarray, fraction: float) -> np.ndarray:
    """Fractional-octave moving average of (F, a, b) along frequency."""
    from .hrtf import smooth_fractional_octave
    Mt = np.moveaxis(M, 0, -1)
    return np.moveaxis(smooth_fractional_octave(Mt, freqs, fraction), -1, 0)


def covariance_constrain(Hsh: np.ndarray, Href: np.ndarray, wref: np.ndarray, order: int,
                         freqs: np.ndarray, f_lo: float = 500.0, f_hi: float = 1500.0,
                         smoothing_fraction: float = 6.0) -> np.ndarray:
    """Zaunschirm et al. (2018) diffuse-field covariance constraint on SH-domain binaural filters.

    Hsh (2, N, F) SH->ear filters; Href (P, 2, F) reference HRTFs with quadrature weights wref.
    Finds per bin the 2x2 M with M R_hat M^H = R_target closest to identity (Procrustes /
    optimal mixing) and returns M Hsh.
    """
    w = np.asarray(wref, dtype=np.float64)
    w = w / np.sum(w)
    nidx = acn_order_index(order)
    c = 1.0 / (2.0 * nidx + 1.0)                     # SN3D diffuse-field covariance
    Rt = np.einsum("p,paf,pbf->fab", w, Href, np.conj(Href))
    Rh = np.einsum("akf,k,bkf->fab", Hsh, c, np.conj(Hsh))
    I2 = np.eye(2)[None, :, :]
    tr_t = np.real(np.trace(Rt, axis1=1, axis2=2))[:, None, None]
    tr_h = np.real(np.trace(Rh, axis1=1, axis2=2))[:, None, None]
    Rt = Rt + (1e-6 * tr_t + 1e-14) * I2
    Rh = Rh + (1e-6 * tr_h + 1e-14) * I2
    X = np.linalg.cholesky(Rt)
    Xh = np.linalg.cholesky(Rh)
    C = np.conj(np.swapaxes(Xh, 1, 2)) @ X
    U, _, Vh = np.linalg.svd(C)
    P = np.conj(np.swapaxes(Vh, 1, 2)) @ np.conj(np.swapaxes(U, 1, 2))
    M = X @ P @ np.linalg.inv(Xh)
    b = _smooth_band(freqs, f_lo, f_hi)[:, None, None]
    M = (1.0 - b) * I2 + b * M
    if smoothing_fraction:
        M = _smooth_matrix_bins(M, freqs, smoothing_fraction)
    return np.einsum("fab,bkf->akf", M, Hsh)


class BinauralDecoder(Decoder):
    """Virtual-loudspeaker binaural decoder folded into SH-domain FIRs (2, N, Lf)."""

    def __init__(self, order: int, sr: int = SAMPLE_RATE, hrtf="builtin", covariance: bool = True,
                 maxre_hz: float = 2000.0, hrir_length: int = 256, pre_delay: int = 32):
        from .hrtf import get_hrtf

        self.layout = "binaural"
        self.order = int(order)
        self.sr = int(sr)
        self.channels = 2
        self.channel_names = ["L", "R"]
        self.channel_mask = CHANNEL_MASKS["binaural"]
        self.hrtf = get_hrtf(hrtf, self.sr, hrir_length)
        N = n_channels(self.order)
        Lh = int(self.hrtf.length)
        nfft = _next_pow2(4 * Lh)
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.sr)

        if self.order <= 3:
            vaz, vel, vw = lebedev26()
        else:
            vaz, vel = fibonacci_sphere(max(50, 2 * N))
            vw = np.full(vaz.shape[0], 1.0 / vaz.shape[0])
        self.virtual_az, self.virtual_el = vaz, vel
        Hs = np.fft.rfft(self.hrtf.hrirs(vaz, vel), nfft, axis=-1)        # (S, 2, F)
        nidx = acn_order_index(self.order)
        Dk = vw[:, None] * sh_real(self.order, vaz, vel).T * (2.0 * nidx + 1.0)[None, :]   # (S, N)
        g_re = maxre_weights(self.order) * maxre_energy_compensation(self.order)
        bnd = _smooth_band(freqs, maxre_hz / 1.5, maxre_hz * 1.5)
        gw = (1.0 - bnd)[None, :] + bnd[None, :] * g_re[:, None]          # (order+1, F)
        Hsh = np.einsum("sef,sk->ekf", Hs, Dk) * gw[nidx][None, :, :]
        if covariance:
            _, _, rw, rir = self.hrtf.reference()
            Href = np.fft.rfft(rir, nfft, axis=-1)
            Hsh = covariance_constrain(Hsh, Href, rw, self.order, freqs)
        Hsh = Hsh * np.exp(-2j * math.pi * freqs * (pre_delay / self.sr))[None, None, :]
        h = np.fft.irfft(Hsh, nfft, axis=-1)
        Lf = nfft // 2
        nf = max(8, Lf // 8)
        h = h[..., :Lf].copy()
        h[..., -nf:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, nf)))
        self.filters = h                                                  # (2, N, Lf)
        self._Lf = Lf
        self._N = N
        self._tail = np.zeros((2, Lf - 1))
        self._spec: dict[int, np.ndarray] = {}
        self.latency = int(pre_delay + round(1.0e-3 * self.sr))

    def _spectrum(self, nfft: int) -> np.ndarray:
        H = self._spec.get(nfft)
        if H is None:
            H = np.fft.rfft(self.filters, nfft, axis=-1)
            self._spec[nfft] = H
        return H

    def process(self, bus: np.ndarray) -> np.ndarray:
        b = _fit_bus(bus, self._N)
        n = b.shape[1]
        if n == 0:
            return np.zeros((2, 0))
        Lf = self._Lf
        nfft = _next_pow2(n + Lf - 1)
        X = np.fft.rfft(b, nfft, axis=1)
        Y = np.einsum("ekf,kf->ef", self._spectrum(nfft), X)
        y = np.fft.irfft(Y, nfft, axis=1)[:, : n + Lf - 1]
        y[:, : Lf - 1] += self._tail
        self._tail = y[:, n:].copy()
        return y[:, :n].copy()

    def reset(self) -> None:
        self._tail[:] = 0.0


# ---------------------------------------------------------------------------
def make_decoder(layout: str, order: int, sr: int = SAMPLE_RATE, hrtf="builtin", **kw) -> Decoder:
    """Build a decoder for `layout` (stereo|binaural|5.1|7.1|7.1.4|foa|hoa3|hoaN|mono).

    Extra keywords: width (stereo), rear_atten_db (stereo), lfe_gain (5.1/7.x),
    order_style 'wav'|'smpte' (7.x channel order), covariance / maxre_hz (binaural).
    """
    lay = normalize_layout(layout)
    order = int(order)
    N = n_channels(order)
    if lay == "mono":
        D = np.zeros((1, N))
        D[0, 0] = 1.0
        return MatrixDecoder("mono", order, sr, D, ["M"], mask=CHANNEL_MASKS["mono"])
    if lay == "stereo":
        D = stereo_matrix(order, width=kw.get("width", 1.0), rear_atten_db=kw.get("rear_atten_db", 3.0))
        return MatrixDecoder("stereo", order, sr, D, ["L", "R"], mask=CHANNEL_MASKS["stereo"])
    if lay in SPEAKER_LAYOUTS:
        return _speaker_decoder(lay, order, sr, order_style=kw.get("order_style", "wav"),
                                lfe_gain=kw.get("lfe_gain", 0.0))
    if lay == "foa":
        return PassthroughDecoder("foa", order, sr, 1)
    if lay.startswith("hoa"):
        try:
            out_order = int(lay[3:])
        except ValueError as exc:
            raise ValueError(f"unknown ambisonic layout {layout!r}") from exc
        return PassthroughDecoder(lay, order, sr, out_order)
    if lay == "binaural":
        return BinauralDecoder(order, sr, hrtf=hrtf, covariance=kw.get("covariance", True),
                               maxre_hz=kw.get("maxre_hz", 2000.0),
                               hrir_length=kw.get("hrir_length", 256))
    raise ValueError(f"unknown output layout {layout!r}")
