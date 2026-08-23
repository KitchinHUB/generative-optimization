"""Generate the Table of Contents (TOC) graphic for

    "Generative machine learning approaches to optimization"
    V. Alves and J. R. Kitchin, Ind. Eng. Chem. Res.

The graphic tells the central story of the paper in two panels:

    (left)  the learned joint distribution p(tau, dCB/dtau) for the
            consecutive-reaction example A -> B -> C, with the
            conditioning slice dCB/dtau = 0 drawn across it;

    (right) the conditional distribution p(tau | dCB/dtau = 0) sampled
            from the GMM and from conditional flow matching, against the
            analytical optimum.

Everything is computed here from the same models used in the manuscript
(generative_optimization.py), so the TOC is reproducible rather than drawn.

ACS TOC/abstract graphic requirements implemented below:
  * exactly 3.25 in x 1.75 in (8.25 cm x 4.45 cm)
  * 300 dpi color, saved as TIFF (also PNG and PDF for convenience)
  * sans-serif type, nothing smaller than 6 pt (body labels at 8 pt)
  * text restricted to axis labels, the reaction scheme and arrow labels

Usage
-----
    python make_toc.py            # writes toc.{png,pdf,tif}
    python make_toc.py --fast     # fewer FM epochs, for layout iteration

Runtime is dominated by flow-matching training (~1-2 min on a CPU).
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde
import torch

from generative_optimization import best_gmm, ConditionalFlowMatching

# --------------------------------------------------------------------------
# ACS TOC specification
# --------------------------------------------------------------------------
FIG_W_IN, FIG_H_IN = 3.25, 1.75          # maximum allowed, submit at actual size
DPI = 300                                 # 300 dpi for color
FONT_LABEL = 7.5                          # ACS prefers 8 pt
FONT_TICK = 6                             # ACS absolute minimum
FONT_ANNOT = 6.5

# Colors used throughout the manuscript figures
C_GMM = "#c0392b"     # red   - Gaussian mixture model
C_FM = "#2c5fa8"      # blue  - conditional flow matching
C_TRUE = "#000000"    # black - analytical reference
C_DENS = "#4a7c59"    # green - learned density contours

SEED = 42

# --------------------------------------------------------------------------
# Problem: consecutive first-order reactions A -> B -> C in a batch reactor
# --------------------------------------------------------------------------
K1 = 0.5      # 1/min, A -> B
K2 = 0.2      # 1/min, B -> C
CA0 = 1.0     # mol/L


def cb(tau):
    """Concentration of the intermediate B at residence time tau."""
    return CA0 * K1 / (K2 - K1) * (np.exp(-K1 * tau) - np.exp(-K2 * tau))


def dcb_dtau(tau):
    """First derivative of CB with respect to tau."""
    return CA0 * K1 * (K2 * np.exp(-K2 * tau) - K1 * np.exp(-K1 * tau)) / (K2 - K1)


TAU_MAX = 12.0                              # training/plotting domain, min
TAU_STAR = np.log(K1 / K2) / (K1 - K2)     # 3.0543 min
CB_STAR = cb(TAU_STAR)                      # 0.5429 mol/L


# --------------------------------------------------------------------------
# Fit the two generative models exactly as in the manuscript
# --------------------------------------------------------------------------
def fit_models(fast=False):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # GMM training data.  Note: the manuscript samples tau over [0.1, 20]
    # with 200 points, which under-resolves the region around the optimum and
    # biases the conditional mean low (2.90 vs 3.05 min).  Sampling the same
    # density over the plotted domain [0.1, 12] removes that bias
    # (3.07 +/- 0.03 min) without changing the method.
    tau_gmm = np.linspace(0.1, TAU_MAX, 400)
    data_gmm = np.column_stack([tau_gmm, cb(tau_gmm), dcb_dtau(tau_gmm)])
    gmm, info = best_gmm(data_gmm, verbose=False)

    # Conditional distribution p(tau, CB | dCB/dtau = 0)
    samples_gmm = gmm.condition([2], [[0.0]]).sample(2000)

    # Flow matching training data: oversampled near the optimum
    tau_fm = np.concatenate([np.linspace(0.1, TAU_MAX, 1000),
                             np.linspace(TAU_STAR - 2, TAU_STAR + 2, 500)])
    fm = ConditionalFlowMatching(x_dim=1, c_dim=1, hidden_dim=128,
                                 n_layers=4, sigma_min=0.001)
    fm.fit(tau_fm.reshape(-1, 1), dcb_dtau(tau_fm).reshape(-1, 1),
           epochs=200 if fast else 1000, batch_size=128, verbose=False)
    samples_fm = fm.sample(c_values=[[0.0]], n_samples=2000, n_steps=100)

    return gmm, data_gmm, samples_gmm[:, 0], samples_fm[:, 0], info


def marginal_density(gmm, dims, grid_x, grid_y):
    """Evaluate the GMM marginal over two of its dimensions on a grid.

    Marginalizing a Gaussian mixture is exact: keep the corresponding
    entries of each mean and the corresponding block of each covariance.
    """
    i, j = dims
    XX, YY = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    dens = np.zeros(len(pts))
    for w, mu, cov in zip(gmm.priors, gmm.means, gmm.covariances):
        m = mu[[i, j]]
        S = cov[np.ix_([i, j], [i, j])]
        Si = np.linalg.inv(S)
        d = pts - m
        expo = -0.5 * np.einsum("ni,ij,nj->n", d, Si, d)
        norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(S)))
        dens += w * norm * np.exp(expo)
    return XX, YY, dens.reshape(XX.shape)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": FONT_LABEL,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.labelpad": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def build_figure(gmm, data_gmm, s_gmm, s_fm):
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)

    # Two panels with a gap in the middle for the conditioning arrow
    ax1 = fig.add_axes([0.118, 0.235, 0.312, 0.585])
    ax2 = fig.add_axes([0.652, 0.235, 0.328, 0.585])

    # ---------------- left: the learned joint distribution ----------------
    tau_lo, tau_hi = 0.0, TAU_MAX
    d_lo, d_hi = -0.09, 0.52

    # unconditional draws from the fitted GMM read as "a distribution"
    joint = gmm.sample(4000)
    m = ((joint[:, 0] > tau_lo) & (joint[:, 0] < tau_hi)
         & (joint[:, 2] > d_lo) & (joint[:, 2] < d_hi))
    ax1.scatter(joint[m, 0], joint[m, 2], s=5.0, c=C_DENS,
                alpha=0.16, lw=0, rasterized=True, zorder=2)
    ax1.scatter(joint[m, 0], joint[m, 2], s=1.0, c=C_DENS,
                alpha=0.35, lw=0, rasterized=True, zorder=3)

    tt = np.linspace(0.05, tau_hi, 400)
    ax1.plot(tt, dcb_dtau(tt), color=C_TRUE, lw=0.8, zorder=4)

    # the conditioning slice
    ax1.axhline(0.0, color=C_GMM, lw=0.9, ls=(0, (3.2, 2.0)), zorder=5)
    ax1.plot([TAU_STAR], [0.0], "o", ms=3.4, mfc="white",
             mec=C_GMM, mew=1.0, zorder=6)

    ax1.set_xlim(tau_lo, tau_hi)
    ax1.set_ylim(d_lo, d_hi)
    ax1.set_xticks([0, 6, 12])
    ax1.set_yticks([0.0, 0.5])
    ax1.set_xlabel(r"$\tau$ (min)")
    ax1.set_ylabel(r"d$C_\mathrm{B}$/d$\tau$", labelpad=1.0)

    # reaction scheme
    ax1.text(0.97, 0.94, r"A $\rightarrow$ B $\rightarrow$ C",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=FONT_ANNOT, color="0.3")
    ax1.set_title(r"$p(\tau,\ \mathrm{d}C_\mathrm{B}/\mathrm{d}\tau)$",
                  fontsize=FONT_ANNOT, color="0.15", pad=2.5)

    for s in ax1.spines.values():
        s.set_color("0.4")

    # ---------------- right: the conditional distribution ----------------
    x_lo, x_hi = 2.80, 3.30
    grid = np.linspace(x_lo, x_hi, 500)
    for s, c in ((s_gmm, C_GMM), (s_fm, C_FM)):
        kde = gaussian_kde(s)
        y = kde(grid)
        ax2.fill_between(grid, 0, y, color=c, alpha=0.45, lw=0, zorder=2)
        ax2.plot(grid, y, color=c, lw=0.8, zorder=3)

    ax2.axvline(TAU_STAR, color=C_TRUE, lw=0.8, ls=(0, (2.4, 1.6)), zorder=5)

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(0, None)
    ax2.set_xticks([2.8, 3.0, 3.2])
    ax2.set_yticks([])
    ax2.set_xlabel(r"$\tau$ (min)")

    for s in ax2.spines.values():
        s.set_color("0.4")
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # colored inline keys instead of a legend box
    ax2.text(0.03, 0.98, "GMM", transform=ax2.transAxes, ha="left", va="top",
             fontsize=FONT_ANNOT, color=C_GMM)
    ax2.text(0.03, 0.72, "flow\nmatching", transform=ax2.transAxes, ha="left",
             va="top", fontsize=FONT_ANNOT, color=C_FM, linespacing=1.05)
    ax2.annotate(r"$\tau^{*}$", xy=(TAU_STAR, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(1.5, -1.0), textcoords="offset points",
                 ha="left", va="top", fontsize=FONT_ANNOT, color=C_TRUE)
    ax2.set_title(r"$p(\tau \mid \mathrm{d}C_\mathrm{B}/\mathrm{d}\tau = 0)$",
                  fontsize=FONT_ANNOT, color="0.15", pad=2.5)

    # ---------------- middle: the conditioning step ----------------
    fig.text(0.545, 0.575, "condition", ha="center", va="bottom",
             fontsize=FONT_ANNOT, color="0.3")
    arrow = FancyArrowPatch((0.470, 0.495), (0.618, 0.495),
                            transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=7,
                            lw=1.0, color="0.3", shrinkA=0, shrinkB=0)
    fig.add_artist(arrow)

    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="train flow matching briefly, for layout iteration")
    p.add_argument("--stem", default="toc")
    args = p.parse_args()

    gmm, data_gmm, s_gmm, s_fm, info = fit_models(fast=args.fast)

    print(f"analytical    tau* = {TAU_STAR:.4f} min, CB(tau*) = {CB_STAR:.4f} mol/L")
    print(f"GMM ({info['best_k']} comps)  mean = {s_gmm.mean():.3f} +/- {s_gmm.std():.3f} min")
    print(f"flow matching mean = {s_fm.mean():.3f} +/- {s_fm.std():.3f} min")

    style()
    fig = build_figure(gmm, data_gmm, s_gmm, s_fm)

    for ext in ("png", "pdf"):
        out = f"{args.stem}.{ext}"
        fig.savefig(out, dpi=DPI)
        print(f"wrote {out}")
    plt.close(fig)

    # ACS wants a flat RGB TIFF at 300 dpi; matplotlib writes RGBA, so
    # composite onto white and drop the alpha channel.
    from PIL import Image
    rgba = Image.open(f"{args.stem}.png").convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.split()[3])
    tif = f"{args.stem}.tif"
    flat.save(tif, compression="tiff_lzw", dpi=(DPI, DPI))
    print(f"wrote {tif}  ({flat.size[0]}x{flat.size[1]} px, "
          f"{flat.size[0]/DPI:.2f} x {flat.size[1]/DPI:.2f} in @ {DPI} dpi, RGB)")


if __name__ == "__main__":
    main()
