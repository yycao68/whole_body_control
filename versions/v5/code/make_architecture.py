"""Architecture figure for the v5 paper: the Interaction Dynamics layer within a
model-based Physical AI hierarchy, with the layer's internal estimate -> confidence-
arbitrate -> realize dataflow expanded. Vector PDF + PNG."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)

# palette (restrained, print-friendly, works light)
C_PLAN = "#E8EEF4"; C_PLAN_E = "#5B7A99"
C_LAYER = "#FDF1E6"; C_LAYER_E = "#D55E00"
C_REAL = "#EAF3EA"; C_REAL_E = "#3B7A3B"
C_BLK = "#FFFFFF"; C_HI = "#FBE3CE"
TXT = "#1a1a1a"; GREY = "#888"

fig, ax = plt.subplots(figsize=(7.4, 8))
ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.4, r=0.12, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, size=10, w="normal", c=TXT, style="normal", ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=w, color=c, style=style, zorder=5)


def arrow(x1, y1, x2, y2, c=TXT, lw=1.6, style="-|>", ms=12, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=c, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---- time-scale spine (left) ----
txt(0.8, 8.3, "long\ntime scale\n100 ms – s", size=8, c=GREY, style="italic")
txt(0.8, 6.1, "short\ntime scale\n1 – 20 ms", size=8, c=GREY, style="italic")
txt(0.8, 1.3, "actuation", size=8, c=GREY, style="italic")
#for y in (9.9, 1.55):
#    ax.plot([0.9, 10], [y, y], ls=(0, (6, 4)), color=GREY, lw=0.9, zorder=1)

# ---- Zone 1: Perception / planning ----
box(1.4, 7.7, 8.0, 1.2, C_PLAN, C_PLAN_E)
txt(5.4, 8.6, "Perception · World Model · Motion Planning", size=11, w="bold", c=C_PLAN_E)
txt(5.4, 8.1, "terrain recognition · human intent · obstacle prediction · foothold replanning",
    size=8.5, c="#40566b")

# ---- Zone 2: Interaction Dynamics Layer (the contribution) ----
box(1.4, 1.85, 8.0, 5.0, C_LAYER, C_LAYER_E, lw=2.0)
txt(5.4, 6.5, "Interaction Dynamics Layer", size=11.5, w="bold", c=C_LAYER_E)

# estimator block
box(1.7, 4.70, 3.1, 0.95, C_BLK, "#555")
txt(3.25, 5.4, "External-wrench estimator", size=9.2, w="bold")
txt(3.25, 4.9, r"$\hat F_{ext}=m\,\ddot c_{xy}-\sum_i F_{i,xy}$", size=9.5)

# confidence arbitration container (highlighted)
box(5.1, 2.1, 4.0, 4.0, C_HI, C_LAYER_E, lw=1.6)
txt(7.2, 5.85, "Interaction-confidence arbitration", size=9.4, w="bold", c=C_LAYER_E)
# level 1 gate
box(5.55, 4.8, 3.3, 0.8, C_BLK, "#555")
txt(7.2, 5.3, "1) confidence gate", size=8.8, w="bold")
txt(7.2, 5.0, r"$g:\ |\hat F_{ext}| > f_{floor}+k_\sigma\,\sigma_f$ ?", size=8.6)
# level 2 persistence
box(5.55, 3.4, 3.3, 0.8, C_BLK, "#555")
txt(7.2, 3.9, "2) persistence", size=8.8, w="bold")
txt(7.2, 3.6, r"$p:$  transient  $\leftrightarrow$  sustained", size=8.6)
# blend
box(5.55, 2.3, 3.3, 0.6, "#FFF7EF", C_LAYER_E)
txt(7.2, 2.6, r"$u=g\,[(1-p)\,u_{capture}+p\,u_{hold}]$", size=8.6, w="bold")

# state input strip
box(1.7, 2.1, 3.1, 0.7, "#F4F4F4", "#999")
txt(3.25, 2.5, "robot state:  IMU · CoM vel\n· foot wrenches", size=8.2)

# internal arrows
arrow(4.8, 5.2, 5.6, 5.2, c=C_LAYER_E)          # estimator -> gate
arrow(7.2, 4.8, 7.2, 4.15, c=C_LAYER_E)          # gate -> persistence
arrow(7.2, 3.4, 7.2, 2.85, c=C_LAYER_E)           # persistence -> blend
arrow(3.25, 2.8, 3.25, 4.75, c="#777", lw=1.3)     # state -> estimator

# ---- Zone 3: realization + robot ----
box(5.5, 0.1, 3.2, 1.0, C_REAL, C_REAL_E)
txt(7.2, 0.85, "Whole-body realization", size=10, w="bold", c=C_REAL_E)
txt(7.2, 0.48, "frozen RL locomotion policy", size=8.6, c="#2f5f2f")
box(2.0, 0.2, 2.0, 0.8, C_PLAN_E, C_PLAN_E)
txt(3.0, 0.6, "Robot", size=11, w="bold", c="white")

# ---- inter-zone dataflow (labels sit in the gap between zones) ----
arrow(6.5, 7.7, 6.5, 6.8, c=C_PLAN_E)          # planning -> layer (command)
txt(6.72, 7.2, "walk command ·\nfootstep plan", size=7.4, c=C_PLAN_E, ha="left")
# command modulation: layer bias -> policy
arrow(7.2, 2.5, 7.2, 1.05, c=C_LAYER_E)   # bias -> policy (command modulation)
txt(8.0, 1.5, "velocity-command\nbias  u", size=8, c=C_LAYER_E)
arrow(5.5, 0.6, 3.9, 0.6, c=C_REAL_E)             # policy -> robot
# feedback: robot state up the left margin into the layer
arrow(3.2, 0.9, 3.2, 2.15, c="#777", lw=1.2)
txt(4.5, 1.3, "robot state feedback \nIMU, kinematics, foot wrenches", size=7.6, c="#777")
# up-signal: interaction state + authority -> planner (far from the command arrow)
arrow(3.2, 6.8, 3.2, 7.7, c="#B22222", lw=1.6)
txt(3.5, 7.2, r"interaction state + $\dot\theta_{fall}$" + "\n$\\to$ replan / widen / slow / abort",
    size=7.3, c="#B22222", ha="left")

fig.tight_layout(pad=0.4)
fig.savefig(HERE / "figures" / "architecture.pdf", bbox_inches="tight")
fig.savefig(HERE / "figures" / "architecture.png", dpi=170, bbox_inches="tight")
print("saved figures/architecture.pdf and .png")
