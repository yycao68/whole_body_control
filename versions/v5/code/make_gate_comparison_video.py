#!/usr/bin/env python3
"""Render the confidence-gate's headline mechanism: capture-runaway vs gated.

Both panels see IDENTICAL conditions -- process noise past the deadband,
NO real external force (force_override pinned to zero) -- so any lateral
drift is purely the ID layer's own response to noise it cannot classify as
a real disturbance. This is the exact failure mode documented in
STAGE2_FINDINGS.md and Sec. "Capture-amplification mechanism" of wbc_v5.tex:

    LEFT  (gate forced open, reproducing the pre-fix ungated law): capture
          steps TOWARD unclassified drift it engages on, closing a positive-
          feedback loop -- a runaway lateral excursion (no fall, but a large,
          unbounded-looking drift).
    RIGHT (gate enabled, the shipped confidence gate): capture stays
          disengaged until the wrench estimate rises a confidence margin
          above its own noise floor, so sub-threshold noise never latches it.

The paper's own headline ablation (Sec. "Capture-amplification mechanism")
reports 1539mm (ungated) vs ~95mm (gated, the policy's own floor) at 4N
process noise, 20-seed medians of a floor-corrected paired metric. This
video reruns the SAME mechanism on ONE seed with the raw (unpaired)
lat_offset_mm for a direct, watchable illustration -- it is not a
substitute for the paper's statistics, just their visual explanation.

Run: python3 make_gate_comparison_video.py
     (writes figures/gate_comparison.mp4)
"""
from __future__ import annotations

import numpy as np
import mujoco
import imageio.v2 as imageio
import cv2

import stage2_id_on_policy as S

W, H = 480, 480
FPS = 30
SEED = 2000
NOISE = 4.0
DURATION = 8.0
ZERO = lambda t: (0.0, 0.0)

OUT = S.HERE.parent / "figures" / "gate_comparison.mp4"


def _rollout(disable_gate: bool) -> dict:
    return S.run(
        "id_mpc", push_n=1.0, push_t=3.0, push_dir=(0, 1), push_dur=3.0,
        push_phase="time", duration=DURATION, seed=SEED, process_noise=NOISE,
        id_mode="wrench", force_override=ZERO, disable_gate=disable_gate,
        video={"fps": FPS, "width": W, "height": H, "distance": 2.2,
               "azimuth": 135.0, "elevation": -12.0},
    )


def _annotate(frame, label, t, ey_mm, final=None):
    img = frame.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (35, 30, 25), -1)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (212, 210, 205), 1, cv2.LINE_AA)
    col = (90, 90, 235) if abs(ey_mm) > 150 else (170, 165, 160)
    cv2.putText(img, f"t={t:4.2f}s  instantaneous lateral error={ey_mm:+7.1f} mm",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    if final is not None:
        ok, text = final
        c = (120, 200, 110) if ok else (90, 90, 235)
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]),
                      (35, 30, 25), -1)
        cv2.putText(img, text, (6, img.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, c, 1, cv2.LINE_AA)
    return img


def _title_card(width, height, title, lines):
    top = np.array([32, 26, 20], dtype=np.float32)
    bot = np.array([48, 34, 26], dtype=np.float32)
    grad = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1, 1)
    img = (top * (1 - grad) + bot * grad)
    img = np.broadcast_to(img, (height, width, 3)).astype(np.uint8).copy()

    accent = (86, 196, 255)
    fail_col = (90, 90, 235)
    ok_col = (120, 200, 110)
    body_col = (212, 210, 205)
    dim_col = (150, 145, 140)

    y = int(height * 0.10)
    cv2.putText(img, title, (24, y), cv2.FONT_HERSHEY_DUPLEX, 0.56, accent, 1, cv2.LINE_AA)
    y += 14
    cv2.line(img, (24, y), (width - 24, y), (64, 58, 50), 1, cv2.LINE_AA)
    y += 24
    for line in lines:
        col = body_col
        if line.startswith("LEFT"):
            col = fail_col
        elif line.startswith("RIGHT"):
            col = ok_col
        cv2.putText(img, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        y += 21

    mid = width // 2
    cv2.line(img, (mid, int(height * 0.06)), (mid, int(height * 0.97)), dim_col, 1, cv2.LINE_AA)
    return img


def main() -> None:
    print("rolling out ungated (gate forced open)...")
    r_open = _rollout(disable_gate=True)
    print("rolling out gated (shipped confidence gate)...")
    r_gate = _rollout(disable_gate=False)
    drift_open = float(r_open["lat_offset_mm"])
    drift_gate = float(r_gate["lat_offset_mm"])
    print(f"ungated final lat_offset: {drift_open:+.1f} mm")
    print(f"gated   final lat_offset: {drift_gate:+.1f} mm")

    frames_o, frames_g = r_open["frames"], r_gate["frames"]
    t_o, t_g = r_open["t"], r_gate["t"]
    ey_o, ey_g = r_open["ey_mm"], r_gate["ey_mm"]
    v_stride = max(1, round(1.0 / (FPS * S.SIM_DT)))
    idx_o = list(range(0, len(t_o), v_stride))[:len(frames_o)]
    idx_g = list(range(0, len(t_g), v_stride))[:len(frames_g)]

    n = max(len(frames_o), len(frames_g))
    frames_o += [frames_o[-1]] * (n - len(frames_o))
    frames_g += [frames_g[-1]] * (n - len(frames_g))
    idx_o += [idx_o[-1]] * (n - len(idx_o))
    idx_g += [idx_g[-1]] * (n - len(idx_g))

    HOLD = int(2.0 * FPS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT), fps=FPS, codec="libx264", quality=8,
                                 macro_block_size=None)

    intro = _title_card(2 * W + 6, H,
        "The confidence gate stops noise from becoming a runaway.", [
        "Both panels see the SAME process noise and NO real external",
        "force (it is pinned to zero) -- any lateral drift here is the",
        "ID layer reacting to noise it cannot classify as a disturbance.",
        "",
        "LEFT: gate forced open (the pre-fix ungated capture law).",
        "Capture steps TOWARD drift it engages on but cannot classify,",
        "closing a positive-feedback loop -- a runaway lateral excursion.",
        "",
        "RIGHT: the shipped confidence gate. Capture stays disengaged",
        "until the wrench estimate clears a self-calibrated margin above",
        "its own noise floor, so sub-threshold noise never latches it.",
        "",
        f"This run: {drift_open:+.0f} mm (ungated) vs {drift_gate:+.0f} mm (gated).",
        "Paper's 20-seed median (floor-corrected, 4N noise): 1539 vs ~95 mm.",
    ])
    for _ in range(int(8.0 * FPS)):
        writer.append_data(intro)

    for k in range(n + HOLD):
        kk = min(k, n - 1)
        io_, ig_ = idx_o[kk], idx_g[kk]
        final_o = (False, f"RUNAWAY: {drift_open:+.0f} mm windowed drift, from noise alone") if k >= n - 1 else None
        final_g = (True, f"STABLE: {drift_gate:+.0f} mm windowed drift -- gate stayed closed") if k >= n - 1 else None
        a_o = _annotate(frames_o[kk], "gate forced OPEN (ungated capture law)",
                         t_o[io_], ey_o[io_], final_o)
        a_g = _annotate(frames_g[kk], "gate ENABLED (shipped confidence gate)",
                         t_g[ig_], ey_g[ig_], final_g)
        gap = np.full((H, 6, 3), 200, dtype=np.uint8)
        writer.append_data(np.concatenate([a_o, gap, a_g], axis=1))
    writer.close()
    print(f"wrote {OUT}  ({n + HOLD} frames @ {FPS}fps = {(n + HOLD) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
