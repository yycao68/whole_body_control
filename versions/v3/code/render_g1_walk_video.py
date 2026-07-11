#!/usr/bin/env python3
"""Render a Neuralink-style MP4 for the Unitree G1 10 s walking verification.

Left: MuJoCo off-screen replay of the Unitree G1.
Right: live plots of CoM tracking, CoM height, torso attitude, and contacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "g1_wbc.xml"
RESULTS = HERE / "results"

VIEW_W, VIEW_H = 880, 640
PANEL_W = 520
FPS = 30


def _font(size):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=float),
        np.eye(3).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _hide_demo_props(model):
    """Hide A-to-B demo props that are not part of the walking verification."""
    for name in ("tree_trunk", "tree_canopy", "apple"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_group[gid] = 5
            model.geom_rgba[gid, 3] = 0.0


def _hud_title(summary):
    if summary.get("push_enabled", False):
        return "Unitree G1 dual-MPC root-assisted walking with short lateral push"
    if summary.get("root_assist_enabled", False):
        return "Unitree G1 dual-MPC root-assisted 10 s walking demo"
    return "Unitree G1 10 s walking with dual interaction-MPC"


def _hud_footer(summary):
    if summary.get("root_assist_enabled", False):
        push_text = "short lateral push" if summary.get("push_enabled", False) else "no push"
        return f"body/task normalized MPC commands + root assist, alternating one-foot swing, {push_text}"
    return "S4-lite: position-actuated G1, pelvis wrench recovery + J^T/Kp task mapping"


def _pass_value(summary):
    return summary.get("passes_s4_lite", summary.get("passes_visual_demo", False))


def _target_distance(summary):
    return summary.get("expected_distance_m", summary.get("commanded_distance_m", summary.get("distance_m", 0.0)))


def _push_window(summary):
    if not summary.get("push_enabled", False):
        return None
    start = float(summary["push_start_s"])
    return start, start + float(summary["push_duration_s"])


def _draw_hud(rgb, title, metrics, footer):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    w, _ = img.size
    title_font = _font(23)
    metric_font = _font(17)
    small_font = _font(14)
    draw.rectangle([0, 0, w, 132], fill=(13, 17, 24, 188))
    draw.text((14, 10), title, font=title_font, fill=(255, 255, 255, 255))
    y = 44
    for text, color in metrics:
        draw.text((16, y), text, font=metric_font, fill=color + (255,))
        y += 25
    draw.text(
        (16, 112),
        footer,
        font=small_font,
        fill=(210, 218, 230, 255),
    )
    return np.asarray(img)


class LivePanel:
    def __init__(self, log, summary):
        self.log = log
        self.summary = summary
        dpi = 100
        self.fig, axes = plt.subplots(
            4, 1, figsize=(PANEL_W / dpi, VIEW_H / dpi), dpi=dpi, sharex=True
        )
        self.axes = axes
        self.fig.patch.set_facecolor("#0f1218")
        self.fig.subplots_adjust(left=0.18, right=0.96, top=0.93, bottom=0.09, hspace=0.33)

        t = log["t"]
        self.cursor_lines = []
        self.lines = []
        lateral_panel = bool(summary.get("push_enabled", False))
        specs = [
            ("com_x", "CoM x [m]", "#60a5fa"),
            ("com_y" if lateral_panel else "com_z", "CoM y [m]" if lateral_panel else "CoM z [m]", "#34d399"),
            ("att", "roll/pitch [rad]", "#f59e0b"),
            ("contact", "contact", "#f472b6"),
        ]
        for ax, (_, ylabel, _) in zip(axes, specs):
            ax.set_facecolor("#161a22")
            for sp in ax.spines.values():
                sp.set_color("#4b5563")
            ax.grid(alpha=0.20, color="#697386")
            ax.tick_params(colors="#cbd5e1", labelsize=8)
            ax.set_ylabel(ylabel, color="#e5e7eb", fontsize=9)
            ax.set_xlim(t[0], t[-1])
            self.cursor_lines.append(ax.axvline(t[0], color="#f9fafb", lw=1.1, alpha=0.8))

        window = _push_window(summary)
        if window is not None:
            for ax in axes:
                ax.axvspan(window[0], window[1], color="#ef4444", alpha=0.16)

        axes[0].plot(t, log["com"][:, 0], color="#60a5fa", lw=1.8, label="CoM")
        axes[0].plot(t, log["com_ref"][:, 0], "--", color="#fb923c", lw=1.4, label="ref")
        axes[0].legend(loc="upper left", fontsize=8, facecolor="#111827", labelcolor="#e5e7eb")

        if lateral_panel:
            axes[1].plot(t, log["com"][:, 1], color="#34d399", lw=1.8, label="CoM")
            axes[1].plot(t, log["com_ref"][:, 1], "--", color="#fb923c", lw=1.4, label="ref")
            axes[1].legend(loc="upper left", fontsize=8, facecolor="#111827", labelcolor="#e5e7eb")
        else:
            axes[1].plot(t, log["com"][:, 2], color="#34d399", lw=1.8)
            axes[1].axhline(0.38, color="#ef4444", ls="--", lw=1.0)

        axes[2].plot(t, log["rpy"][:, 0], color="#38bdf8", lw=1.5, label="roll")
        axes[2].plot(t, log["rpy"][:, 1], color="#f97316", lw=1.5, label="pitch")
        axes[2].legend(loc="upper left", fontsize=8, facecolor="#111827", labelcolor="#e5e7eb")

        axes[3].step(t, log["contact"][:, 0], where="post", color="#60a5fa", lw=1.4, label="left")
        axes[3].step(t, log["contact"][:, 1] + 1.2, where="post", color="#fb923c", lw=1.4, label="right")
        axes[3].set_ylim(-0.15, 2.45)
        axes[3].set_yticks([0, 1, 1.2, 2.2])
        axes[3].legend(loc="upper left", fontsize=8, facecolor="#111827", labelcolor="#e5e7eb")
        axes[3].set_xlabel("time [s]", color="#e5e7eb", fontsize=9)

        self.fig.suptitle(
            "Live verification traces",
            color="#f9fafb",
            fontsize=13,
            fontweight="bold",
        )

    def frame(self, idx):
        t = float(self.log["t"][idx])
        for line in self.cursor_lines:
            line.set_xdata([t, t])
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        return np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)[:, :, [1, 2, 3]]

    def close(self):
        plt.close(self.fig)


def load_log(prefix):
    log_path = RESULTS / f"{prefix}_log.npz"
    summary_path = RESULTS / f"{prefix}_summary.json"
    if not log_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {log_path.name} or {summary_path.name}. Run run_g1_root_assist_demo.py first."
        )
    log = dict(np.load(log_path))
    with summary_path.open() as f:
        summary = json.load(f)
    if "qpos" not in log:
        raise RuntimeError("Log does not contain qpos. Re-run run_g1_root_assist_demo.py.")
    return log, summary


def render_video(prefix="g1_walk_10s", fps=FPS, out=None):
    log, summary = load_log(prefix)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    _hide_demo_props(model)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = VIEW_W
    model.vis.global_.offheight = VIEW_H

    renderer = mujoco.Renderer(model, height=VIEW_H, width=VIEW_W)
    camera = mujoco.MjvCamera()
    camera.distance = 2.45
    camera.azimuth = 74.0
    camera.elevation = -9.0
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 0
    panel = LivePanel(log, summary)

    out = out or str(RESULTS / f"{prefix}_video.mp4")
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)

    stride = max(1, int(round((1.0 / fps) / 0.001)))
    indices = list(range(0, len(log["t"]), stride))
    if indices[-1] != len(log["t"]) - 1:
        indices.append(len(log["t"]) - 1)

    hand_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")

    for idx in indices:
        data.qpos[:] = log["qpos"][idx]
        mujoco.mj_forward(model, data)
        com = log["com"][idx]
        camera.lookat[:] = [com[0] + 0.10, 0.0, 0.78]
        renderer.update_scene(data, camera=camera, scene_option=opt)

        # Overlay only small task markers. Large force arrows can occlude the
        # robot in off-screen renders, so the force history is shown in plots.
        _add_sphere(renderer.scene, log["com"][idx], 0.025, [0.2, 0.65, 1.0, 1.0])
        _add_sphere(renderer.scene, log["com_ref"][idx], 0.018, [1.0, 0.55, 0.15, 1.0])
        _add_sphere(renderer.scene, data.site_xpos[hand_sid], 0.020, [0.95, 0.15, 0.15, 1.0])

        rgb = renderer.render()
        metrics = [
            (f"t = {log['t'][idx]:4.2f} s    pass = {_pass_value(summary)}", (255, 255, 255)),
            (f"distance = {summary['distance_m']:.3f} m / {_target_distance(summary):.3f} m", (145, 205, 255)),
        ]
        if summary.get("push_enabled", False):
            metrics.append(
                (
                    f"push y-accel = {summary['push_accel_mps2']:.2f} m/s^2   max lateral dev = {summary['max_lateral_deviation_m']:.3f} m",
                    (180, 255, 205),
                )
            )
        else:
            metrics.append(
                (
                    f"min CoM z = {summary['min_com_height_m']:.3f} m   max |roll,pitch| = {summary['max_abs_roll_pitch_rad']:.3f} rad",
                    (180, 255, 205),
                )
            )
        rgb = _draw_hud(rgb, _hud_title(summary), metrics, _hud_footer(summary))
        curves = panel.frame(idx)
        writer.append_data(np.hstack([rgb, curves]))

    writer.close()
    renderer.close()
    panel.close()
    print(f"[video] {len(indices)} frames @ {fps} fps -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="g1_walk_10s")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--mode",
        choices=("mujoco",),
        default="mujoco",
        help="render the real MuJoCo G1 viewport.",
    )
    args = parser.parse_args()
    render_video(prefix=args.prefix, fps=args.fps, out=args.out)


if __name__ == "__main__":
    main()
