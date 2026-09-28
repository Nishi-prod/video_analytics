"""Static figures: synthetic-store preview and the sensitivity small multiples."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def preview(layout, tracks, per_cam, path, n_tracks=40, seed=0):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(13, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    W, H = layout.floor
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=INK2, lw=1))
    cust = tracks[tracks.role == "customer"]
    ids = rng.choice(cust.id.unique(), size=min(n_tracks, cust.id.nunique()), replace=False)
    for pid in ids:
        g = cust[cust.id == pid]
        ax.plot(g.x, g.y, color="#9a9890", lw=0.6, alpha=0.6)
    for i, z in enumerate(layout.zones):
        ax.add_patch(Polygon(z.polygon, closed=True, fc=SERIES[i % len(SERIES)], alpha=0.22, ec=SERIES[i % len(SERIES)], lw=2))
        c = z.centroid
        ax.text(c[0], c[1], z.label, ha="center", va="center", fontsize=8, color=INK)
    e = layout.entrance
    ax.plot([e.a[0], e.b[0]], [e.a[1], e.b[1]], color=INK, lw=3)
    ax.text((e.a[0] + e.b[0]) / 2, e.a[1] - 0.45, "entrance line", ha="center", fontsize=8, color=INK)
    for cam in layout.cameras:
        ax.plot(cam.pos[0], cam.pos[1], marker="s", ms=8, color=INK)
        ax.annotate("", xy=cam.look_at[:2], xytext=cam.pos[:2], arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
        ax.text(cam.pos[0], cam.pos[1] + 0.35, cam.id, ha="center", fontsize=8, color=INK2)
    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim(-3.2, H + 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Synthetic store '{layout.name}' - {len(ids)} sample shopper paths (metres)", fontsize=10, color=INK, loc="left")
    for k, cam in enumerate(layout.cameras):
        a = fig.add_subplot(gs[1, k])
        _style(a)
        ses = next(iter(per_cam[cam.id]))
        df = per_cam[cam.id][ses]
        busiest = df.groupby("frame").size().idxmax()
        fr = df[df.frame == busiest]
        a.add_patch(Rectangle((0, 0), cam.width, cam.height, fc="#f1f0ec", ec=GRID))
        for i, z in enumerate(layout.zones):
            if z.camera == cam.id:
                a.add_patch(Polygon(cam.ground_to_pixel(z.polygon), closed=True, fc=SERIES[i % len(SERIES)], alpha=0.22,
                                    ec=SERIES[i % len(SERIES)], lw=1.5))
        if layout.entrance.camera == cam.id:
            pe = cam.ground_to_pixel(np.array([e.a, e.b]))
            a.plot(pe[:, 0], pe[:, 1], color=INK, lw=2)
        for _, r in fr.iterrows():
            col = INK if r.role == "customer" else SERIES[1]
            a.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, ec=col, lw=1.2, ls="-" if r.vis > 0.5 else "--"))
        a.set_xlim(0, cam.width)
        a.set_ylim(cam.height, 0)
        a.set_aspect("equal")
        a.set_title(f"{cam.id} · busiest frame · {len(fr)} people", fontsize=8.5, color=INK, loc="left")
    fig.text(0.01, 0.005, "Camera views (640x360): black = customer, orange = staff, dashed = less than 50% visible. "
             "Shaded areas = zones watched by that camera; thick line = entrance line.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def sensitivity_chart(df, path):
    labels = [f"{p}" + (f"\n+ReID\n{int(100*r)}%" if r > 0 else "") for p, r in zip(df.preset, df.reid_recovery)]
    panels = [("IDF1", "Tracking IDF1 (higher is better)", False),
              ("footfall_total_pct_err", "Footfall total error (%)", True),
              ("visits_pct_err_mean", "Zone visit count error (%, mean |err|)", False),
              ("stop_vs_pass_F1", "Stop-vs-pass F1 (higher is better)", False),
              ("mean_dwell_abs_err_s", "Mean dwell error (s)", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4), facecolor=SURFACE)
    x = np.arange(len(df))
    for ax, (col, title, signed) in zip(axes, panels):
        _style(ax)
        v = df[col].to_numpy(float)
        ax.bar(x, v, width=0.6, color=SERIES[0], zorder=3)
        ax.axhline(0, color=INK2, lw=0.8, zorder=4)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi, vi, f"{vi:+.1f}" if signed else f"{vi:.2f}" if abs(vi) < 10 else f"{vi:.0f}",
                        ha="center", va="bottom" if vi >= 0 else "top", fontsize=7, color=INK)
    fig.suptitle("How simulated tracker errors propagate into business metrics", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
