#!/usr/bin/env python3
"""Render D1 (baseline PD) vs D7 (proposed, contact-consistent WBC+MPC+Kalman)
on Scenario A -- the fixed double-support 8N step pHRI disturbance that
produces Table III's headline steady-state numbers.

Both panels see the IDENTICAL disturbance (same F_DIST, same T_DIST, no
seed jitter -- seed=None reproduces the deterministic nominal case used in
the table). This reuses scenario_a.py's own run_controller() with its new
opt-in `video` parameter, so the rendered episode is the exact, audited D1
and D7 controllers Table III reports -- not a simplified stand-in.

Run: python3 make_scenario_a_video.py
     (writes results/scenario_a_video.mp4)
"""
from __future__ import annotations

import numpy as np
import imageio.v2 as imageio
import cv2

from scenario_a import run_controller, compute_metrics, CONTROLLERS, OUT_DIR

W, H = 480, 480
FPS = 30
OUT = OUT_DIR / "scenario_a_video.mp4"


def _annotate(frame, label, t, e_mm, final=None):
    img = frame.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (35, 30, 25), -1)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (212, 210, 205), 1, cv2.LINE_AA)
    col = (90, 90, 235) if e_mm > 12.0 else (170, 165, 160)
    cv2.putText(img, f"t={t:4.2f}s  hand tracking error={e_mm:6.2f} mm",
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
    video_cfg = {"fps": FPS, "width": W, "height": H,
                 "distance": 1.6, "azimuth": 75.0, "elevation": -8.0}
    print("simulating D1 SK05 PD...")
    t1, e1, v1 = run_controller('D1 SK05 PD', CONTROLLERS['D1 SK05 PD'], video=video_cfg)
    print("simulating D7 Proposed Full...")
    t7, e7, v7 = run_controller('D7 Proposed Full', CONTROLLERS['D7 Proposed Full'], video=video_cfg)
    rms1, ss1 = compute_metrics(t1, e1)
    rms7, ss7 = compute_metrics(t7, e7)
    print(f"D1 steady-state: {ss1:.2f} mm   D7 steady-state: {ss7:.2f} mm")

    frames1, frames7 = v1["frames"], v7["frames"]
    emag1 = np.linalg.norm(e1, axis=1) * 1000.0
    emag7 = np.linalg.norm(e7, axis=1) * 1000.0
    v_stride = max(1, round(1.0 / (FPS * 0.001)))
    idx1 = list(range(0, len(t1), v_stride))[:len(frames1)]
    idx7 = list(range(0, len(t7), v_stride))[:len(frames7)]

    n = max(len(frames1), len(frames7))
    frames1 += [frames1[-1]] * (n - len(frames1))
    frames7 += [frames7[-1]] * (n - len(frames7))
    idx1 += [idx1[-1]] * (n - len(idx1))
    idx7 += [idx7[-1]] * (n - len(idx7))

    HOLD = int(2.0 * FPS)
    writer = imageio.get_writer(str(OUT), fps=FPS, codec="libx264", quality=8,
                                 macro_block_size=None)

    intro = _title_card(2 * W + 6, H,
        "Scenario A: an 8N step force at the hand, both feet planted.", [
        "Same disturbance, same instant, both panels -- the deterministic",
        "nominal case behind Table III's steady-state numbers.",
        "",
        "LEFT: D1, a plain joint-space PD baseline (SS error = F/Kp,",
        "the theoretical 10mm droop of an 800 N/m stiffness).",
        "",
        "RIGHT: D7, the proposed contact-consistent WBC + impedance MPC",
        "+ Kalman disturbance estimator. The MPC's integral-like",
        "disturbance rejection drives the steady-state error toward zero",
        "while the contact-consistent projection keeps the correction",
        "from disturbing stance.",
        "",
        f"This run: D1 {ss1:.1f}mm vs D7 {ss7:.1f}mm steady-state error.",
    ])
    for _ in range(int(7.0 * FPS)):
        writer.append_data(intro)

    for k in range(n + HOLD):
        kk = min(k, n - 1)
        i1, i7 = idx1[kk], idx7[kk]
        final1 = (False, f"D1 steady-state: {ss1:.2f} mm (theoretical 10.0 mm droop)") if k >= n - 1 else None
        final7 = (True, f"D7 steady-state: {ss7:.2f} mm ({ss1/max(ss7,0.01):.0f}x improvement)") if k >= n - 1 else None
        a1 = _annotate(frames1[kk], "D1 SK05 PD (baseline)", t1[i1], emag1[i1], final1)
        a7 = _annotate(frames7[kk], "D7 Proposed Full (WBC+MPC+Kalman)", t7[i7], emag7[i7], final7)
        gap = np.full((H, 6, 3), 200, dtype=np.uint8)
        writer.append_data(np.concatenate([a1, gap, a7], axis=1))
    writer.close()
    print(f"wrote {OUT}  ({n + HOLD} frames @ {FPS}fps = {(n + HOLD) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
