"""Decoders: SN3D/ACN bus (N, n) -> output channels (C, n).

* mono      W, the pressure signal at the listening position: unit gain for every direction, so
            a mono render has the loudness of the stereo / speaker renders (whose decoders are
            energy-normalised to a direction-averaged power of 1). AmbiX FOA/HOA carry W at the
            same level, so mono, foa and hoa3 files share their peak when W is the loudest channel.
* stereo    virtual-microphone pair fitted (regularised least squares, i.e. mode matching on a
            dense grid) to a constant-power panning law whose image spans +-30 deg loudspeakers,
            with a `width` control and a gentle rear attenuation.
* 5.1, 7.1, 7.1.4
            AllRAD (Zotter & Frank 2012, JAES 60(10)): a max-rE sampling decoder onto a dense
            near-uniform virtual layout, then VBAP (Pulkki 1997) from every virtual point to
            the real loudspeakers (imaginary zenith/nadir speakers close the hull and are
            shared by the loudspeakers around them), symmetrised left/right and refined for
            direction-independent energy (+-0.6 dB). Energy-normalised so the
            direction-averaged sum of squared gains is 1. Speaker angles follow ITU-R
            BS.775 / BS.2051.
* foa, hoaN pass-through (truncate or zero-pad), AmbiX.
* binaural  SH-domain filters fitted to the whole HRTF set (built-in Brown-Duda head or SOFA,
            `hrtf.py`) by magnitude least squares above N c / (2 pi r) (MagLS, Schoerkhuber,
            Zaunschirm & Hoeldrich 2018), with the diffuse-field covariance constraint of
            Zaunschirm, Schoerkhuber & Hoeldrich (2018), JASA 143(6):3616, which restores the
            interaural coherence and diffuse-field timbre of the reference HRTF set
            (optimal 2x2 mixing per bin, Vilkamo, Baeckstroem & Kuntz 2013). The older
            virtual-loudspeaker design (26-point Lebedev, max-rE above ~2 kHz) is method='vls'.
            Block FFT overlap-add carries the filter tails across blocks.

Channel order is the WAVEFORMATEXTENSIBLE bit order (so files are correct with their
channel mask): 5.1 = L R C LFE Ls Rs (mask 0x60F), 7.1 = L R C LFE Lrs Rrs Lss Rss (0x63F),
7.1.4 = 7.1 + Ltf Rtf Ltr Rtr (0x2D63F). Pass `order_style='smpte'` for L R C LFE Lss Rss Lrs
Rrs (the decoder then reports channel_mask 0, since that order is not the mask's bit order).
The LFE channel is empty unless `lfe_gain` > 0, when it carries W through a 120 Hz
Linkwitz-Riley low-pass (the mains are not high-passed: no bass management).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np

from ..core.constants import SAMPLE_RATE
from .sh import (acn_nm, acn_order_index, dense_grid, fibonacci_sphere, lebedev26, maxre_energy_compensation,
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
        if self._lfe_sos is not None and b.shape[1] > 0:     # sosfilt rejects zero-length input
            from scipy.signal import sosfilt
            lfe, self._lfe_zi = sosfilt(self._lfe_sos, b[0], zi=self._lfe_zi)
            if not np.any(np.abs(self._lfe_zi) > 1e-30):
                self._lfe_zi[:] = 0.0                # no denormal tail after silence
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
    """Least-squares virtual-microphone pair (2, N) matching a constant-power stereo panning law.

    Energy-normalised like the surround decoders. Diffuse-field inter-channel coherence ~0.57.
    """
    az, el = dense_grid(order)
    u = unit_vectors(az, el)
    q = np.clip(float(width) * u[:, 1], -1.0, 1.0)
    den = np.sqrt(2.0 * (1.0 + q * q))
    rear = 10.0 ** (-(rear_atten_db / 20.0) * (1.0 - u[:, 0]) / 2.0)
    G = np.vstack([(1.0 + q) / den, (1.0 - q) / den]) * rear[None, :]
    Y = sh_real(order, az, el)                       # (N, P)
    A = Y @ Y.T
    lam = reg * np.trace(A) / A.shape[0]
    D = np.linalg.solve(A + lam * np.eye(A.shape[0]), Y @ G.T).T
    # the Fibonacci grid is not mirror-symmetric: make L(az) = R(-az) exact
    _, m = acn_nm(order)
    D = 0.5 * (D + D[::-1] * np.where(m < 0, -1.0, 1.0)[None, :])
    # same calibration as the surround decoders (direction-averaged power 1), so a diffuse bed has
    # one level in every layout; the panning law itself gives front +1.3 dB, sides -0.4, rear -1.8
    return _energy_normalize(D, order)


def _vbap_hull(spk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-degenerate convex-hull triangles of the speaker directions and their inverse bases."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(spk)
    tris = []
    invs = []
    for simplex in hull.simplices:
        M = spk[simplex]                            # rows = speaker vectors
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        tris.append(simplex)
        invs.append(np.linalg.inv(M))
    return np.array(tris), np.array(invs)


def vbap_gains(speakers_xyz: np.ndarray, dirs_xyz: np.ndarray) -> np.ndarray:
    """3-D VBAP gains (L, P), power-normalised, over the convex hull of the speaker directions."""
    spk = np.asarray(speakers_xyz, dtype=np.float64)
    dirs = np.asarray(dirs_xyz, dtype=np.float64)
    tris, invs = _vbap_hull(spk)                    # (T, 3), (T, 3, 3)
    g = np.einsum("pj,tjk->ptk", dirs, invs)        # p = g M  ->  g = p M^-1
    best = np.argmax(g.min(axis=2), axis=1)
    gsel = np.maximum(g[np.arange(dirs.shape[0]), best], 0.0)
    gsel /= np.maximum(np.linalg.norm(gsel, axis=1, keepdims=True), 1e-12)
    G = np.zeros((spk.shape[0], dirs.shape[0]))
    cols = np.repeat(np.arange(dirs.shape[0]), 3)
    np.add.at(G, (tris[best].reshape(-1), cols), gsel.reshape(-1))
    return G


def _mirror_pairs(az_deg: np.ndarray, el_deg: np.ndarray, tol: float = 0.5) -> Optional[np.ndarray]:
    """Index of each speaker's left/right mirror image (az -> -az), or None if the layout is not symmetric."""
    perm = np.empty(az_deg.shape[0], dtype=np.int64)
    for i, (a, e) in enumerate(zip(az_deg, el_deg)):
        da = np.abs((az_deg + a + 180.0) % 360.0 - 180.0)
        j = int(np.argmin(da + np.abs(el_deg - e)))
        if da[j] + abs(el_deg[j] - e) > tol:
            return None
        perm[i] = j
    return perm


def allrad_matrix(order: int, az_deg, el_deg, imaginary_el_deg: float = 70.0,
                  energy_reg: Optional[float] = 1.0) -> np.ndarray:
    """AllRAD decoder (L, N) for loudspeakers at the given angles (degrees).

    Zotter & Frank (2012): max-rE sampling decoder onto a dense virtual grid, VBAP from each
    virtual point to the loudspeakers. When no loudspeaker is within 90 - `imaginary_el_deg`
    degrees of the zenith (or nadir), an imaginary loudspeaker closes the hull there; its
    signal is shared by the real loudspeakers it borders, with gain 1/sqrt(K) each, so energy
    from overhead (or from the ground) is kept rather than lost. Left/right-symmetric layouts
    get a symmetrised matrix, which removes the arbitrary choice of diagonal that Qhull makes
    on rectangular faces (e.g. the 7.1.4 top layer).

    AllRAD's total energy still varies by +-1 to +-1.7 dB with direction on these layouts (the
    max-rE lobe sums coherently where loudspeakers are sparse). With `energy_reg` set, the
    matrix is then refined by minimising mean_theta (|D y(theta)|^2 - 1)^2 + energy_reg
    |D - D_allrad|^2 / |D_allrad|^2 on a mirror-symmetric grid: the energy-preservation goal of
    EPAD (Zotter, Pomberger & Noisternig 2012) kept close to AllRAD's panning. At 1.0 this
    flattens the energy to about +-0.6 dB on 5.1 / 7.1 / 7.1.4 and turns the energy vector by
    ~1 deg (median; < 3.5 deg for 95 % of the upper hemisphere); smaller values flatten further
    but lean horizontal 7.1.4 sources towards the top layer.
    """
    az_deg = np.asarray(az_deg, dtype=np.float64).reshape(-1)
    el_deg = np.asarray(el_deg, dtype=np.float64).reshape(-1)
    real = unit_vectors(np.radians(az_deg), np.radians(el_deg))
    L = real.shape[0]
    spk = [real]
    if float(np.max(el_deg)) < imaginary_el_deg:
        spk.append(np.array([[0.0, 0.0, 1.0]]))
    if float(np.min(el_deg)) > -imaginary_el_deg:
        spk.append(np.array([[0.0, 0.0, -1.0]]))
    allspk = np.concatenate(spk, axis=0)
    gaz, gel = dense_grid(order)
    grid = unit_vectors(gaz, gel)
    G = vbap_gains(allspk, grid)
    Gr = G[:L].copy()
    if allspk.shape[0] > L:
        tris, _ = _vbap_hull(allspk)
        for i in range(L, allspk.shape[0]):
            nb = np.unique(tris[np.any(tris == i, axis=1)])
            nb = nb[nb < L]
            if nb.size:
                Gr[nb] += G[i][None, :] / math.sqrt(nb.size)
        # the shared share adds coherently to the triangle's own gains: renormalise every virtual
        # point back to unit panning power, as VBAP itself is
        Gr /= np.maximum(np.linalg.norm(Gr, axis=0, keepdims=True), 1e-12)
    nidx = acn_order_index(order)
    w = ((2.0 * nidx + 1.0) * maxre_weights(order)[nidx]) / grid.shape[0]
    Ydec = sh_real(order, gaz, gel).T * w[None, :]   # (P, N)
    D = Gr @ Ydec
    perm = _mirror_pairs(az_deg, el_deg)
    _, m = acn_nm(order)
    flip = np.where(m < 0, -1.0, 1.0)
    if perm is not None:
        D = 0.5 * (D + D[perm] * flip[None, :])
    D = _energy_normalize(D, order)
    if energy_reg:
        D = _flatten_energy(D, order, float(energy_reg))
        if perm is not None:
            D = 0.5 * (D + D[perm] * flip[None, :])
        D = _energy_normalize(D, order)
    return D


def _flatten_energy(D0: np.ndarray, order: int, reg: float, iters: int = 400) -> np.ndarray:
    """Regularised energy-flattening refinement of a decoding matrix (see allrad_matrix)."""
    from scipy.optimize import minimize

    az, el = fibonacci_sphere(max(300, 20 * n_channels(order)))
    Y = sh_real(order, np.concatenate([az, -az]), np.concatenate([el, el]))   # mirror-symmetric grid
    P = Y.shape[1]
    n0 = float(np.sum(D0 * D0))

    def cost(v):
        D = v.reshape(D0.shape)
        G = D @ Y
        r = np.sum(G * G, axis=0) - 1.0
        dD = D - D0
        val = float(np.mean(r * r) + reg * np.sum(dD * dD) / n0)
        grad = (4.0 / P) * (G * r) @ Y.T + (2.0 * reg / n0) * dD
        return val, grad.ravel()

    res = minimize(cost, D0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=iters))
    return res.x.reshape(D0.shape)


@lru_cache(maxsize=32)
def _layout_matrix(layout: str, order: int) -> np.ndarray:
    spk = SPEAKER_LAYOUTS[layout]
    active = [i for i, s in enumerate(spk) if s[1] is not None]
    D = np.zeros((len(spk), n_channels(order)))
    D[active] = allrad_matrix(order, [spk[i][1] for i in active], [spk[i][2] for i in active])
    D.setflags(write=False)
    return D


def _speaker_decoder(layout: str, order: int, sr: int, order_style: str = "wav",
                     lfe_gain: float = 0.0) -> MatrixDecoder:
    spk = SPEAKER_LAYOUTS[layout]
    names = [s[0] for s in spk]
    D = _layout_matrix(layout, int(order)).copy()
    lfe = names.index("LFE") if "LFE" in names else None
    mask = CHANNEL_MASKS.get(layout, 0)
    if order_style == "smpte" and layout in _SMPTE_ORDER:
        perm = [names.index(nm) for nm in _SMPTE_ORDER[layout]]
        D = D[perm]
        names = [names[i] for i in perm]
        lfe = names.index("LFE")
        mask = 0        # this order is not the mask's bit order; a WAV must not claim the mapping
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


def magls_filters(Href: np.ndarray, az: np.ndarray, el: np.ndarray, weights: np.ndarray, order: int,
                  freqs: np.ndarray, cutoff_hz: float, reg: float = 1e-3) -> np.ndarray:
    """Magnitude-least-squares SH-domain binaural filters (2, N, F).

    Schoerkhuber, Zaunschirm & Hoeldrich (2018), "Binaural rendering of Ambisonic signals via
    magnitude least squares", DAGA: below `cutoff_hz` a weighted, Tikhonov-regularised least-
    squares fit y(theta_p)^T h(f) ~ H(theta_p, f) to the reference HRTFs; above it only the
    magnitudes are fitted, the phase of each bin being taken from the previous bin's estimate
    (so the interaural phase, which order N cannot reproduce at high frequencies, is given up
    in favour of correct magnitude, i.e. ILD and timbre). Href (P, 2, F) on directions az, el.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    Y = sh_real(order, az, el).T                                       # (P, N)
    N = Y.shape[1]
    A = Y.T @ (w[:, None] * Y)
    lam = float(reg) * np.trace(A) / N
    pinv = np.linalg.solve(A + lam * np.eye(N), (Y * w[:, None]).T)    # (N, P)
    H = np.einsum("kp,pef->ekf", pinv, Href)                          # LS everywhere
    kc = int(np.searchsorted(freqs, float(cutoff_hz)))
    if 0 < kc < freqs.shape[0]:
        mag = np.abs(Href)
        prev = H[:, :, kc - 1].T                                       # (N, 2)
        for i in range(kc, freqs.shape[0]):
            est = Y @ prev                                             # (P, 2)
            prev = pinv @ (mag[:, :, i] * np.exp(1j * np.angle(est)))
            H[:, :, i] = prev.T
    return H


@lru_cache(maxsize=16)
def _binaural_filters(hrtf, order: int, sr: int, method: str, covariance: bool, cutoff_hz: float,
                      maxre_hz: float, pre_delay: int) -> np.ndarray:
    N = n_channels(order)
    Lh = int(hrtf.length)
    nfft = _next_pow2(4 * Lh)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    ra, re_, rw, rir = hrtf.reference()
    Href = np.fft.rfft(rir, nfft, axis=-1)                            # (P, 2, F)
    if method == "magls":
        # MagLS carries each bin's phase over from the previous one, i.e. zero group delay above the
        # cutoff: remove the set's bulk delay first and restore it after, so high frequencies arrive
        # with the head-centre delay instead of ~1 ms before the low band
        bulk = np.exp(2j * math.pi * freqs * (_hrir_onset(hrtf, sr) / sr))
        Hsh = magls_filters(Href * bulk[None, None, :], ra, re_, rw, order, freqs, cutoff_hz)
        Hsh = Hsh * np.conj(bulk)[None, None, :]
    elif method in ("vls", "virtual"):
        if order <= 3:
            vaz, vel, vw = lebedev26()
        else:
            vaz, vel = fibonacci_sphere(max(50, 2 * N))
            vw = np.full(vaz.shape[0], 1.0 / vaz.shape[0])
        Hs = np.fft.rfft(hrtf.hrirs(vaz, vel), nfft, axis=-1)        # (S, 2, F)
        nidx = acn_order_index(order)
        Dk = vw[:, None] * sh_real(order, vaz, vel).T * (2.0 * nidx + 1.0)[None, :]   # (S, N)
        g_re = maxre_weights(order) * maxre_energy_compensation(order)
        bnd = _smooth_band(freqs, maxre_hz / 1.5, maxre_hz * 1.5)
        gw = (1.0 - bnd)[None, :] + bnd[None, :] * g_re[:, None]      # (order+1, F)
        Hsh = np.einsum("sef,sk->ekf", Hs, Dk) * gw[nidx][None, :, :]
    else:
        raise ValueError(f"unknown binaural method {method!r} (magls|vls)")
    if covariance:
        Hsh = covariance_constrain(Hsh, Href, rw, order, freqs)
    Hsh = Hsh * np.exp(-2j * math.pi * freqs * (pre_delay / sr))[None, None, :]
    h = np.fft.irfft(Hsh, nfft, axis=-1)
    Lf = nfft // 2
    nf = max(8, Lf // 8)
    h = h[..., :Lf].copy()
    h[..., -nf:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, nf)))
    h.setflags(write=False)
    return h


def _hrir_onset(hrtf, sr: int) -> int:
    """Bulk delay of an HRTF set in samples (first arrival of the frontal HRIR)."""
    from .hrtf import BULK_DELAY_S, BuiltinHRTF
    if isinstance(hrtf, BuiltinHRTF):
        return int(round(BULK_DELAY_S * sr))
    ir = np.asarray(hrtf.hrirs(np.zeros(1), np.zeros(1)))[0]          # (2, L)
    env = np.max(np.abs(ir), axis=0)
    peak = float(np.max(env))
    return int(np.argmax(env >= 0.1 * peak)) if peak > 0 else 0


class BinauralDecoder(Decoder):
    """SH-domain binaural decoder: FIRs (2, N, Lf) applied by FFT overlap-add.

    method 'magls' (default): magnitude least squares above `cutoff_hz` (default
    N c / (2 pi r_head), 1.87 kHz at third order, where kr = N) plus the diffuse-field
    covariance constraint. 'vls': virtual loudspeakers (26-point Lebedev) with max-rE above
    `maxre_hz`. Filters are designed once per (HRTF set, parameters) and shared.
    """

    def __init__(self, order: int, sr: int = SAMPLE_RATE, hrtf="builtin", covariance: bool = True,
                 maxre_hz: float = 2000.0, hrir_length: Optional[int] = None, pre_delay: int = 32,
                 method: str = "magls", cutoff_hz: Optional[float] = None):
        from .hrtf import HEAD_RADIUS_M, SPEED_OF_SOUND, get_hrtf

        self.layout = "binaural"
        self.order = int(order)
        self.sr = int(sr)
        self.channels = 2
        self.channel_names = ["L", "R"]
        self.channel_mask = CHANNEL_MASKS["binaural"]
        self.method = str(method).lower()
        if hrir_length is None:
            # ~5.3 ms (256 taps at 48 kHz) at any rate: shorter truncates the head-shadow filter's
            # low-frequency phase (the 90-deg ITD drops by ~55 us at 96 kHz with 256 taps)
            hrir_length = _next_pow2(int(round(256 * self.sr / 48000.0)))
        self.hrtf = get_hrtf(hrtf, self.sr, hrir_length)
        if cutoff_hz is None:
            cutoff_hz = max(1, self.order) * SPEED_OF_SOUND / (2.0 * math.pi * HEAD_RADIUS_M)
        self.cutoff_hz = float(cutoff_hz)
        if self.order <= 3:
            self.virtual_az, self.virtual_el, _ = lebedev26()
        else:
            self.virtual_az, self.virtual_el = fibonacci_sphere(max(50, 2 * n_channels(self.order)))
        h = _binaural_filters(self.hrtf, self.order, self.sr, self.method, bool(covariance),
                              round(self.cutoff_hz, 6), float(maxre_hz), int(pre_delay))
        self.filters = h                                                  # (2, N, Lf), shared, read-only
        self._Lf = h.shape[-1]
        self._N = n_channels(self.order)
        self._tail = np.zeros((2, self._Lf - 1))
        self._spec: dict[int, np.ndarray] = {}
        self.latency = int(pre_delay) + _hrir_onset(self.hrtf, self.sr)

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
    order_style 'wav'|'smpte' (7.x channel order), method 'magls'|'vls' / cutoff_hz / covariance /
    maxre_hz / hrir_length (binaural).
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
                               hrir_length=kw.get("hrir_length"),
                               method=kw.get("method", "magls"), cutoff_hz=kw.get("cutoff_hz"))
    raise ValueError(f"unknown output layout {layout!r}")
