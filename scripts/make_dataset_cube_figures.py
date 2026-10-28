# -*- coding: utf-8 -*-
"""Render hyperspectral data-cube illustrations (front scene + dense stacked
spectral band slices on the right/top faces) for PaviaU, Xuzhou and
WHU-Hi-HongHu.

Style: front face = true-color composite of the scene; a dense, near-seamless
stack of wavelength-tinted band slices fans out toward the upper-right along
the spectral axis (stack depth ~0.55x the front-face width, ~42 deg rise).

The combined 1x3 panel draws all three cubes in ONE shared-scale axes so the
true relative pixel dimensions are preserved (e.g. HongHu 475x940 renders
proportionally wider/taller than Xuzhou 260x500 and PaviaU 340x610).

Outputs PNG (300 dpi) + PDF per dataset into dataset_cube_figures/.
"""
import os
import sys

import numpy as np
import scipy.io as scio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from loadData.data_read import PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM  # noqa: E402

OUT_DIR = os.path.join(ROOT, "dataset_cube_figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_SLICES = 48        # dense, near-seamless stack like the reference figure
DEPTH_FRAC = 0.55    # total horizontal spread of the stack, relative to width
RISE_SLOPE = 0.9     # vertical rise per unit horizontal depth (~42 deg)
FILL = 0.72          # slice thickness / slice pitch (rest is white gap)
SLICE_CMAP = plt.get_cmap("jet")


def load_cube(name):
    if name == "PaviaU":
        img = scio.loadmat(os.path.join(ROOT, "dataset", "PaviaU", "PaviaU.mat"))["paviaU"]
        wl = PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM.astype(np.float64)
        display = "Pavia University"
        sensor = "ROSIS-03"
    elif name == "Xuzhou":
        img = scio.loadmat(os.path.join(ROOT, "dataset", "Xuzhou", "xuzhou.mat"))["xuzhou"]
        wl = np.linspace(415.0, 2508.0, img.shape[2])
        display = "Xuzhou"
        sensor = "Airborne VNIR+SWIR"
    else:
        img = scio.loadmat(
            os.path.join(ROOT, "dataset", "WHU-Hi", "WHU-Hi-HongHu", "WHU_Hi_HongHu.mat")
        )["WHU_Hi_HongHu"]
        wl = np.linspace(400.0, 1000.0, img.shape[2])
        display = "WHU-Hi-HongHu"
        sensor = "Headwall VNIR"
    return img.astype(np.float64), wl, display, sensor


def nearest_band(wl, target):
    return int(np.argmin(np.abs(wl - target)))


def stretch(gray, lo_pct=1.0, hi_pct=99.0, gamma=0.85):
    gray = gray.astype(np.float64)
    lo, hi = np.percentile(gray, [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        return np.zeros_like(gray)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0) ** gamma


def rgb_composite(img, wl, balance=True):
    idx_r = nearest_band(wl, 660.0)
    idx_g = nearest_band(wl, 550.0)
    idx_b = nearest_band(wl, 480.0)
    if balance:
        # Urban/balanced scenes: per-channel stretch + gray-world balance.
        rgb = np.stack(
            [stretch(img[:, :, idx_r]), stretch(img[:, :, idx_g]), stretch(img[:, :, idx_b])],
            axis=-1,
        )
        means = rgb.reshape(-1, 3).mean(axis=0)
        target = means.mean()
        gains = target / np.maximum(means, 1e-6)
        rgb = np.clip(rgb * gains[None, None, :], 0.0, 1.0)
    else:
        # Vegetation-dominated scenes: one shared lo/hi over all three bands so
        # spectral shape (and thus natural hues) is preserved; no gray-world,
        # which would wrongly neutralize the green dominance of vegetation.
        chans = np.stack(
            [img[:, :, idx_r], img[:, :, idx_g], img[:, :, idx_b]], axis=-1
        )
        lo, hi = np.percentile(chans, [1.0, 99.0])
        rgb = np.clip((chans - lo) / max(hi - lo, 1e-12), 0.0, 1.0) ** 0.9
    return rgb, (idx_r, idx_g, idx_b)


def tinted_slice(band, color):
    gray = stretch(band, 1.0, 99.0, gamma=1.0)
    rgba = np.zeros(band.shape + (4,), dtype=np.float64)
    rgba[..., :3] = gray[..., None] * np.asarray(color)[None, None, :]
    rgba[..., 3] = 1.0
    return rgba


def draw_cube(ax, img, wl, title, subtitle, balance=True,
              origin=(0.0, 0.0), autoscale=True, title_y=None, sub_y=None):
    """Draw one data cube with its front-face lower-left corner at `origin`.

    autoscale=False keeps the caller in charge of xlim/ylim (shared-scale
    combined panel); title_y/sub_y force a common title baseline across cubes.
    """
    h, w, nb = img.shape
    x0, y0 = origin
    rgb, (ir, ig, ib) = rgb_composite(img, wl, balance=balance)

    depth_x = DEPTH_FRAC * w
    gx, gy = depth_x / N_SLICES, RISE_SLOPE * depth_x / N_SLICES
    tx, ty = FILL * gx, FILL * gy

    band_idx = np.unique(
        np.round(np.linspace(0.03, 0.97, N_SLICES) * (nb - 1)).astype(int)
    )
    n = len(band_idx)

    # Dense stack fanning toward the upper-right; draw far slices first so the
    # nearer ones (and finally the front face) sit on top.
    for i in range(n - 1, -1, -1):
        frac = band_idx[i] / (nb - 1)
        color = SLICE_CMAP(0.08 + 0.84 * frac)[:3]
        rgba = tinted_slice(img[::2, ::2, band_idx[i]], color)
        z = 3.0 + (n - i) * 0.01

        # Right face: unit square -> u along thickness, v along (0, h).
        aff_right = Affine2D.from_values(
            tx, ty, 0.0, h, x0 + w + i * gx, y0 + i * gy
        ) + ax.transData
        ax.imshow(rgba, extent=(0, 1, 0, 1), origin="upper",
                  transform=aff_right, interpolation="bilinear", zorder=z)

        # Top face: unit square -> u along (w, 0), v along thickness.
        aff_top = Affine2D.from_values(
            w, 0.0, tx, ty, x0 + i * gx, y0 + h + i * gy
        ) + ax.transData
        ax.imshow(rgba, extent=(0, 1, 0, 1), origin="upper",
                  transform=aff_top, interpolation="bilinear", zorder=z)

        for corners in (
            [(x0 + w + i * gx, y0 + i * gy), (x0 + w + i * gx + tx, y0 + i * gy + ty),
             (x0 + w + i * gx + tx, y0 + i * gy + ty + h), (x0 + w + i * gx, y0 + i * gy + h)],
            [(x0 + i * gx, y0 + h + i * gy), (x0 + i * gx + w, y0 + h + i * gy),
             (x0 + i * gx + w + tx, y0 + h + i * gy + ty), (x0 + i * gx + tx, y0 + h + i * gy + ty)],
        ):
            ax.add_patch(Polygon(
                corners, closed=True, fill=False,
                edgecolor=(0.2, 0.2, 0.2, 0.35), lw=0.3, zorder=z + 0.001,
            ))

    # Front face: real scene RGB composite.
    ax.imshow(rgb, extent=(x0, x0 + w, y0, y0 + h), origin="upper",
              interpolation="bilinear", zorder=6)
    ax.add_patch(Polygon(
        [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], closed=True,
        fill=False, edgecolor=(0.1, 0.1, 0.1, 0.9), lw=1.0, zorder=7,
    ))

    # Dimension arrows: width (bottom), height (left), spectral depth (top-right).
    arrow_kw = dict(arrowstyle="<->", color="#333333", lw=0.9,
                    shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(x0 + w, y0 - 0.055 * h), xytext=(x0, y0 - 0.055 * h),
                arrowprops=arrow_kw, zorder=8)
    ax.text(x0 + w / 2, y0 - 0.075 * h, f"{w} px", ha="center", va="top",
            fontsize=8.5, color="#333333", zorder=8)
    ax.annotate("", xy=(x0 - 0.05 * w, y0 + h), xytext=(x0 - 0.05 * w, y0),
                arrowprops=arrow_kw, zorder=8)
    ax.text(x0 - 0.075 * w, y0 + h / 2, f"{h} px", ha="center", va="center",
            fontsize=8.5, color="#333333", rotation=90, zorder=8)

    corner = (x0 + w + n * gx, y0 + h + n * gy)
    ax.annotate("", xy=corner, xytext=(x0 + w, y0 + h),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=8)
    # Label sits just beyond the fan's top-right corner, extending left ABOVE
    # the fan boundary so no slice edges cross the text.
    ax.text(
        corner[0] + 0.03 * w, corner[1] + 0.015 * h,
        "Spectral dimension\n"
        f"{wl[0]:.0f}\u2013{wl[-1]:.0f} nm, {nb} bands",
        ha="right", va="bottom", fontsize=8.5, color="#333333", zorder=9,
    )

    # Title block below the cube (keeps the top free for the spectral label).
    x_center = x0 + (w + depth_x) / 2.0
    ty_ = y0 - 0.12 * h if title_y is None else title_y
    sy_ = (ty_ - 0.065 * h) if sub_y is None else sub_y
    ax.text(x_center, ty_, title, ha="center", va="top",
            fontsize=12, fontweight="semibold", color="#111111", zorder=9)
    ax.text(x_center, sy_, subtitle, ha="center", va="top",
            fontsize=9, color="#444444", zorder=9)

    ax.set_aspect("equal")
    if autoscale:
        ax.set_xlim(x0 - 0.14 * w, x0 + w + depth_x + 0.10 * w)
        ax.set_ylim(y0 - 0.26 * h, y0 + 1.38 * h)
        ax.axis("off")
    return (ir, ig, ib)


def main():
    specs = []
    for name in ("PaviaU", "Xuzhou", "HongHu"):
        img, wl, display, sensor = load_cube(name)
        h, w, nb = img.shape
        subtitle = f"{h} \u00d7 {w} \u00d7 {nb} ({sensor})"
        balance = name != "HongHu"
        specs.append((img, wl, display, subtitle, balance))

    for img, wl, display, subtitle, balance in specs:
        fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=300)
        bands_used = draw_cube(ax, img, wl, display, subtitle, balance=balance)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        stem = display.replace(" ", "_")
        fig.savefig(os.path.join(OUT_DIR, f"datacube_{stem}.png"),
                    dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(os.path.join(OUT_DIR, f"datacube_{stem}.pdf"),
                    facecolor="white", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        print(f"{display}: RGB band indices (R,G,B)={bands_used}")

    # Combined panel: single axes, shared scale -> true relative proportions.
    h_max = max(img.shape[0] for img, *_ in specs)
    gap = 0.25 * h_max

    x = 0.0
    origins = []
    extents = []
    for img, *_ in specs:
        h, w, nb = img.shape
        origins.append(x)
        extents.append((x - 0.14 * w, x + w + DEPTH_FRAC * w + 0.10 * w))
        x += w + DEPTH_FRAC * w + 0.10 * w + gap
    xlim = (extents[0][0], extents[-1][1])
    ylim = (-0.30 * h_max, 1.38 * h_max)

    units_w = xlim[1] - xlim[0]
    units_h = ylim[1] - ylim[0]
    fig_w = 15.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * units_h / units_w), dpi=300)
    for (img, wl, display, subtitle, balance), x0 in zip(specs, origins):
        draw_cube(ax, img, wl, display, subtitle, balance=balance,
                  origin=(x0, 0.0), autoscale=False,
                  title_y=-0.16 * h_max, sub_y=-0.225 * h_max)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(os.path.join(OUT_DIR, "datacube_three_datasets.png"),
                dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(os.path.join(OUT_DIR, "datacube_three_datasets.pdf"),
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Combined panel saved (shared scale, units {units_w:.0f} x {units_h:.0f}).")
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
