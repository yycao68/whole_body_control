"""Fig. 2 for the v5 paper: the interaction-reasoning pipeline.
Interaction Observation -> Interaction State -> Interaction Reasoning -> Interaction Action,
with the two confidence decisions branching into defer / capture / hold. This is the
conceptual abstraction of which the estimator, gate, and persistence classifier are parts.
Vector PDF + PNG."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)

C_OBS = "#EAF3EA"; C_OBS_E = "#3B7A3B"
C_ST = "#E8EEF4"; C_ST_E = "#5B7A99"
C_RE = "#FDF1E6"; C_RE_E = "#D55E00"
C_BLK = "#FFFFFF"
C_DEFER = "#F4F4F4"; C_CAP = "#E7F0F7"; C_HOLD = "#EAF3EA"
TXT = "#1a1a1a"; GREY = "#888"

fig, ax = plt.subplots(figsize=(10.2, 3.7))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.5, r=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def txt(x, y, s, size=9.5, w="normal", c=TXT, style="normal", ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=w, color=c, style=style, zorder=5)


def arrow(x1, y1, x2, y2, c=TXT, lw=1.7, ms=13, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=c, zorder=4, connectionstyle=f"arc3,rad={rad}"))


# ---- stage 1: Interaction Observation ----
box(0.35, 1.55, 2.0, 1.5, C_OBS, C_OBS_E)
txt(1.4, 2.75, "Interaction", size=9.6, w="bold", c=C_OBS_E)
txt(1.4, 2.42, "Observation", size=9.6, w="bold", c=C_OBS_E)
txt(1.4, 1.98, r"$\ddot c$, $\sum_i F_i$,", size=8.6)
txt(1.4, 1.72, r"$e$, $\dot e$", size=8.6)

# ---- stage 2: Interaction State ----
box(3.05, 1.55, 2.3, 1.5, C_ST, C_ST_E)
txt(4.3, 2.75, "Interaction", size=9.6, w="bold", c=C_ST_E)
txt(4.3, 2.42, "State", size=9.6, w="bold", c=C_ST_E)
txt(4.3, 1.98, r"$\hat F_{ext}$ + persistence", size=8.2)
txt(4.3, 1.72, "present? / strength / duration", size=7.4, c="#40566b")

# ---- stage 3: Interaction Reasoning (two decisions) ----
box(5.95, 1.15, 3.0, 2.3, C_RE, C_RE_E, lw=1.8)
txt(7.45, 3.2, "Interaction Reasoning", size=9.6, w="bold", c=C_RE_E)
box(6.2, 2.4, 2.5, 0.6, C_BLK, "#555")
txt(7.45, 2.8, "D1  confidence gate", size=8.3, w="bold")
txt(7.45, 2.56, "is the interaction real?", size=7.6)
box(6.2, 1.32, 2.5, 0.62, C_BLK, "#555")
txt(7.45, 1.78, "D2  persistence", size=8.3, w="bold")
txt(7.45, 1.53, "transient or sustained?", size=7.6)
arrow(7.45, 2.25, 7.45, 2.16, c=C_RE_E)

# ---- stage 4: Interaction Action (three outcomes) ----
box(10.2, 2.62, 2.65, 0.6, C_HOLD, C_OBS_E)
txt(11.52, 3.05, "HOLD", size=9, w="bold", c=C_OBS_E)
txt(11.52, 2.8, r"step against + $\int$", size=7.4)
box(10.2, 1.74, 2.65, 0.6, C_CAP, "#0072B2")
txt(11.52, 2.10, "CAPTURE", size=9, w="bold", c="#0072B2")
txt(11.52, 1.87, "step toward the fall", size=7.4)
box(10.2, 0.86, 2.65, 0.6, C_DEFER, GREY)
txt(11.52, 1.22, "DEFER", size=9, w="bold", c="#555")
txt(11.52, 0.99, "frozen policy only", size=7.4)

txt(11.52, 3.62, "Interaction Action", size=9.6, w="bold", c=TXT)

# ---- flow arrows ----
arrow(2.35, 2.3, 3.1, 2.3, c="#555")
arrow(5.35, 2.3, 5.95, 2.3, c="#555")
# reasoning -> actions with conditions
arrow(8.95, 2.55, 10.2, 2.95, c=C_OBS_E, rad=-0.12)
txt(9.55, 3.02, "confident\n& sustained", size=6.8, c=C_OBS_E, ha="center")
arrow(8.95, 2.2, 10.2, 2.1, c="#0072B2", rad=0.0)
txt(9.58, 2.36, "confident\n& transient", size=6.8, c="#0072B2", ha="center")
arrow(8.95, 1.7, 10.2, 1.22, c=GREY, rad=0.14)
txt(9.62, 1.72, "not\nconfident", size=6.8, c=GREY, ha="center")

# ---- component mapping strip ----
ax.plot([0.15, 12.85], [0.5, 0.5], color="#ddd", lw=0.8)
txt(6.5, 0.28, "estimator  $\\rightarrow$  observation → state      |      confidence gate + persistence classifier  $\\rightarrow$  reasoning"
    "      |      command modulation  $\\rightarrow$  action",
    size=7.6, c="#555", style="italic")

fig.tight_layout(pad=0.3)
fig.savefig(HERE / "figures" / "reasoning_pipeline.pdf", bbox_inches="tight")
fig.savefig(HERE / "figures" / "reasoning_pipeline.png", dpi=170, bbox_inches="tight")
print("saved figures/reasoning_pipeline.pdf and .png")
