"""Representation + dichotomy schematic for the v5 paper (Fig. 2).
Top: the configuration-invariant double-integrator task model  e_ddot = a_e + d_eff.
Bottom: the transient-vs-sustained dichotomy with opposite-sign footstep response
(CAPTURE = step toward the fall; HOLD = step against + offset-free integral).
Illustrative curves, not measured data. Vector PDF + PNG."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)

C_INT = "#D55E00"   # interaction / d_eff (matches Fig.1 layer)
C_CAP = "#0072B2"   # capture (transient)
C_HOLD = "#3B7A3B"  # hold (sustained)
GREY = "#888"; TXT = "#1a1a1a"

fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.55], hspace=0.5, wspace=0.26,
                      left=0.09, right=0.97, top=0.95, bottom=0.11)

# ---------------- top: the invariant model ----------------
axm = fig.add_subplot(gs[0, :]); axm.set_xlim(0, 10); axm.set_ylim(0, 3); axm.axis("off")


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def arw(ax, x1, y1, x2, y2, c=TXT, lw=1.6, ms=12, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=c, zorder=3,
                                 connectionstyle=f"arc3,rad={rad}"))


# interaction source -> d_eff -> double integrator -> task error
box(axm, 0.2, 1.3, 2.5, 1.5, "#FDF1E6", C_INT)
axm.text(1.45, 2.28, "External interaction", ha="center", fontsize=9.2, fontweight="bold", color=C_INT)
axm.text(1.45, 1.55, "terrain · contact ·\nexternal wrench", ha="center", fontsize=8, color="#8a4a12")

box(axm, 4.1, 1.3, 3.0, 1.5, "#FFFFFF", "#333", lw=1.8)
axm.text(5.6, 2.28, "Task double integrator", ha="center", fontsize=9.2, fontweight="bold")
axm.text(5.6, 1.72, r"$\ddot e = a_e + d_{\mathrm{eff}}$", ha="center", fontsize=13)

box(axm, 8.5, 1.3, 1.35, 1.5, "#EAF3EA", "#3B7A3B")
axm.text(9.17, 1.97, r"task error $e$", ha="center", fontsize=9, fontweight="bold", color="#2f5f2f")

arw(axm, 2.7, 1.98, 4.1, 1.98, c=C_INT)
axm.text(3.4, 2.28, r"$d_{\mathrm{eff}}$", ha="center", fontsize=11, color=C_INT, fontweight="bold")
arw(axm, 7.1, 1.98, 8.5, 1.98, c="#333")
axm.text(7.8, 2.25, r"$a_e$+$d_{\mathrm{eff}}$", ha="center", fontsize=8, color="#555")
# feedback tick
arw(axm, 9.17, 1.3, 9.17, 0.65, c=GREY, lw=1.2)
arw(axm, 9.2, 0.7, 5.55, 0.7, c=GREY, lw=1.2)
arw(axm, 5.6, 0.65, 5.6, 1.35, c=GREY, lw=1.2)
axm.text(5.6, 0.15, r"controller closes $a_e$ on $e,\ \dot e$", ha="center", fontsize=7.6, color=GREY, style="italic")
axm.text(1.45, 0.35, "one representation across\ngait phase · terrain · contact",
         ha="center", fontsize=7.8, color=C_INT, style="italic")

# ---------------- bottom-left: transient -> capture ----------------
t = np.linspace(0, 4, 400)
axt = fig.add_subplot(gs[1, 0])
# push impulse at t0; error rises then is captured back toward 0
e_tr = np.where(t > 0.6, 1.0 * (t - 0.6) * np.exp(-(t - 0.6) / 0.55), 0.0)
e_tr = e_tr / e_tr.max()
axt.plot(t, e_tr, color=C_CAP, lw=2.2, zorder=3)
axt.axhline(0, color="#bbb", lw=0.8)
# impulse arrow
axt.annotate("", xy=(0.6, 0.0), xytext=(0.6, 0.72),
             arrowprops=dict(arrowstyle="-|>", color="#B22222", lw=2))
axt.text(0.14, 1.02, "push\n(impulse)", color="#B22222", fontsize=8, ha="left", va="center")
# capture annotation near the recovery
axt.annotate("capture step\n$+\\dot e$: toward the motion", xy=(1.5, e_tr[t <= 1.5][-1]),
             xytext=(1.9, 0.72), fontsize=8, color=C_CAP, ha="left",
             arrowprops=dict(arrowstyle="-|>", color=C_CAP, lw=1.4))
axt.text(3.9, 0.05, "recovers", fontsize=7.6, color=GREY, ha="right", style="italic")
axt.set_title("Transient push  $\\rightarrow$  CAPTURE", fontsize=10, color=C_CAP, fontweight="bold")
axt.set_ylabel("lateral task error  $e$", fontsize=9)
axt.set_xlabel("time", fontsize=8.5)
axt.set_ylim(-0.15, 1.15); axt.set_xlim(0, 4)
axt.set_yticks([]); axt.set_xticks([])
for s in ("top", "right"):
    axt.spines[s].set_visible(False)

# ---------------- bottom-right: sustained -> hold ----------------
axs = fig.add_subplot(gs[1, 1])
e_run = np.where(t > 0.6, 0.62 * (t - 0.6), 0.0)          # no hold -> runaway
e_hold = np.where(t > 0.6, 0.42 * (1 - np.exp(-(t - 0.6) / 0.5)), 0.0)  # hold -> bounded offset
axs.plot(t, e_run, color=GREY, lw=1.8, ls=(0, (5, 3)), zorder=2, label="no hold")
axs.plot(t, e_hold, color=C_HOLD, lw=2.2, zorder=3, label="hold")
axs.axhline(0, color="#bbb", lw=0.8)
# sustained-force arrow (constant)
axs.annotate("", xy=(2.4, 1.02), xytext=(0.6, 1.02),
             arrowprops=dict(arrowstyle="-|>", color="#B22222", lw=2))
axs.text(1.5, 1.08, "force (sustained)", color="#B22222", fontsize=8, ha="center", va="bottom")
axs.text(2.78, 1.27, "runaway (no hold)", color=GREY, fontsize=7.8, ha="left", va="center", style="italic")
axs.annotate("hold: step against\n+ offset-free $\\int$", xy=(3.0, e_hold[t <= 3.0][-1]),
             xytext=(1.15, 0.62), fontsize=8, color=C_HOLD, ha="left",
             arrowprops=dict(arrowstyle="-|>", color=C_HOLD, lw=1.4))
axs.set_title("Sustained force  $\\rightarrow$  HOLD", fontsize=10, color=C_HOLD, fontweight="bold")
axs.set_xlabel("time", fontsize=8.5)
axs.set_ylim(-0.15, 1.35); axs.set_xlim(0, 4)
axs.set_yticks([]); axs.set_xticks([])
for s in ("top", "right"):
    axs.spines[s].set_visible(False)

fig.text(0.5, 0.015, "Opposite-sign footstep response to the same-signed error — "
         "the discriminator is persistence of $d_{\\mathrm{eff}}$, not its sign.",
         ha="center", fontsize=8, style="italic", color="#444")

fig.savefig(HERE / "figures" / "model_dichotomy.pdf", bbox_inches="tight")
fig.savefig(HERE / "figures" / "model_dichotomy.png", dpi=170, bbox_inches="tight")
print("saved figures/model_dichotomy.pdf and .png")
