#!/usr/bin/env python3
"""Illustrative walking-demo video: the root-assisted G1 walks at ~1.2 m/s for
10 s and meets three environmental events on the way --

    t = 2 s   lateral external push  (body interaction MPC recovers)
    t = 5 s   a hole in the ground    (~6 cm deep, at x = 5.4 m)
    t = 8 s   a step up               (10 cm high, at x = 9.0 m)

This is a visualization built on the kinematic root-assist scaffold of
``run_g1_root_assist_demo.py`` (root position scripted, one-foot-swing gait
commanded); it is an animation for communication, not a torque-level physics
proof.  The physics benchmark remains ``run_uneven_ground_benchmark.py``.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf"


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(_FONT_DIR / name), size)


def draw_overlay(frame: np.ndarray, t: float) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    W = im.width
    d.text((12, 8), "Interaction-Dynamics MPC  ·  1.2 m/s walk",
           font=_font(17, True), fill=(255, 255, 255))
    d.text((12, 31), f"t = {t:4.1f} s", font=_font(15), fill=(235, 235, 130))
    banner, color = None, (255, 255, 255)
    if 2.0 <= t < 3.1:
        banner, color = "EXTERNAL PUSH — interaction MPC recovers", (255, 150, 95)
    elif 4.6 <= t < 5.9:
        banner, color = "HOLE  (−6 cm)", (130, 205, 150)
    elif 7.7 <= t < 9.4:
        banner, color = "STEP UP  (+10 cm)", (155, 185, 245)
    if banner:
        f = _font(19, True)
        w = d.textlength(banner, font=f)
        d.rectangle([(W / 2 - w / 2 - 10, 440), (W / 2 + w / 2 + 10, 470)],
                    fill=(0, 0, 0))
        d.text((W / 2 - w / 2, 444), banner, font=f, fill=color)
    return np.asarray(im)

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from run_g1_root_assist_demo import (
    COMMAND_DT, DISTANCE, DURATION, MODEL_PATH, SIM_DT,
    G1CommandLayer, LocalTrajectory, apply_commanded_pose, apply_root_assist,
    body_id, pin_support_foot, site_id, support_phase, support_site_name,
    trajectory,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VIDEODIR = HERE.parent / "videos"
VIDEODIR.mkdir(exist_ok=True)

# ---- event geometry (x-locations reached at t = 5 s and 8 s at 1.2 m/s) -----
HOLE_X, HOLE_HW, HOLE_DEPTH, HOLE_EDGE = 5.4, 0.35, 0.06, 0.16
STEP_X, STEP_HEIGHT, STEP_EDGE = 9.0, 0.10, 0.12
PUSH_START, PUSH_DURATION, PUSH_ACCEL = 2.0, 0.18, 2.6
FPS = 30


def _smooth(a: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def _bump(x, c, hw, edge):
    d = abs(x - c)
    if d <= hw:
        return 1.0
    if d >= hw + edge:
        return 0.0
    return _smooth((hw + edge - d) / edge)


def _stepup(x, x0, edge):
    if x <= x0 - edge:
        return 0.0
    if x >= x0 + edge:
        return 1.0
    return _smooth((x - (x0 - edge)) / (2.0 * edge))


def terrain_height(x: float) -> float:
    """Ground height under the walking line (0 nominal, dips over the hole,
    rises onto the step)."""
    return -HOLE_DEPTH * _bump(x, HOLE_X, HOLE_HW, HOLE_EDGE) \
        + STEP_HEIGHT * _stepup(x, STEP_X, STEP_EDGE)


def _box(world, name, x0, x1, top, rgba):
    cx, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    ET.SubElement(world, "geom", {
        "name": name, "type": "box",
        "pos": f"{cx:.5f} 0 {top - 0.5:.5f}", "size": f"{hx:.5f} 1.4 0.5",
        "rgba": rgba, "friction": "0.9 0.02 0.001", "condim": "3",
        "contype": "1", "conaffinity": "1",
    })


def build_demo_terrain() -> Path:
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    comp = root.find("compiler")
    if comp is not None:
        comp.set("meshdir", str((HERE / "models" / "assets").resolve()))
    world = root.find("worldbody")
    floor = next(g for g in world.findall("geom") if g.attrib.get("name") == "floor")
    world.remove(floor)
    # strip decorative props (tree, apple hand-target) for a clean walk scene
    for g in list(world.findall("geom")):
        if g.attrib.get("name") in ("tree_trunk", "tree_canopy"):
            world.remove(g)
    for b in list(world.findall("body")):
        if b.attrib.get("name") == "apple_body":
            world.remove(b)
    _box(world, "ground_a", -2.0, HOLE_X - HOLE_HW, 0.0, "0.55 0.52 0.47 1")
    _box(world, "hole_pit", HOLE_X - HOLE_HW, HOLE_X + HOLE_HW, -HOLE_DEPTH, "0.16 0.13 0.11 1")
    _box(world, "ground_b", HOLE_X + HOLE_HW, STEP_X, 0.0, "0.55 0.52 0.47 1")
    _box(world, "step_up", STEP_X, 12.0, STEP_HEIGHT, "0.44 0.45 0.50 1")
    out = RESULTS / "g1_wbc_walk_demo_terrain.xml"
    tree.write(out, encoding="unicode")
    return out


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(build_demo_terrain()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    command = G1CommandLayer()
    torso = body_id(model, "torso_link")
    body_mpc = NormalizedMPC(dim=2, dt=SIM_DT, horizon=40, q_pos=85.0, q_vel=18.0,
                             qf_pos=120.0, qf_vel=25.0, r=0.04,
                             u_max=np.array([3.0, 1.4]))
    body_observer = RandomWalkDisturbanceObserver(dim=2, dt=SIM_DT, q_d=0.06, r_y=8e-5)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso
    cam.distance, cam.azimuth, cam.elevation = 3.9, 125.0, -12.0

    steps = int(round(DURATION / SIM_DT))
    command_period = max(1, int(round(COMMAND_DT / SIM_DT)))
    v_stride = max(1, round(1.0 / (FPS * SIM_DT)))

    root_p = data.qpos[:2].copy()
    root_v = np.zeros(2)
    d_body_hat = np.zeros(2)
    ctrl = command.step(0.0, trajectory(0.0))
    locked_support, plant_xy = None, None
    frames, frame_times = [], []

    for k in range(steps):
        t = k * SIM_DT
        ref = trajectory(t)
        u_body = body_mpc.solve(np.concatenate([root_p - ref.position,
                                                root_v - ref.velocity]), d_hat=d_body_hat)
        push_xy = np.zeros(2)
        if PUSH_START <= t < PUSH_START + PUSH_DURATION:
            push_xy[1] = PUSH_ACCEL
        root_acc = ref.acceleration + u_body + push_xy
        root_p = root_p + root_v * SIM_DT + 0.5 * root_acc * SIM_DT**2
        root_v = root_v + root_acc * SIM_DT
        d_body_hat, _ = body_observer.step(root_p - trajectory(min(t + SIM_DT, DURATION)).position,
                                           u_body)
        traj = LocalTrajectory(root_p.copy(), root_v.copy(), root_acc.copy(), heading=0.0)
        if k % command_period == 0:
            ctrl = command.step(t, traj)

        apply_root_assist(data, traj, t)
        data.qpos[2] += terrain_height(root_p[0])
        apply_commanded_pose(model, data, ctrl)
        mujoco.mj_forward(model, data)

        support = support_phase(t)
        if support != locked_support:
            locked_support = support
            plant_xy = data.site_xpos[site_id(model, support_site_name(support)), :2].copy()
        else:
            pin_support_foot(model, data, support, plant_xy)
        data.qpos[2] = 0.82 + 0.008 * max(0.0, math.sin(2 * math.pi * 2.2 * t)) \
            + terrain_height(data.qpos[0])
        mujoco.mj_forward(model, data)

        if k % v_stride == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            frame_times.append(t)
    renderer.close()

    frames = [draw_overlay(f, ft) for f, ft in zip(frames, frame_times)]
    out = VIDEODIR / "walk_hole_step_push.mp4"
    imageio.mimwrite(out, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")
    print(f"events: push @ {PUSH_START}s (x={0.6+1.2*(PUSH_START-1):.1f}m), "
          f"hole @ x={HOLE_X}m (t~5s), step @ x={STEP_X}m (t~8s)")


if __name__ == "__main__":
    main()
