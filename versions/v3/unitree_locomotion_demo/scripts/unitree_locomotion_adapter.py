#!/usr/bin/env python3
"""Adapter skeleton for a Unitree locomotion-base demo.

This file intentionally does not import a Unitree SDK. Different G1 software
stacks expose different interfaces. The point is to keep the v3 correction
module separate from the locomotion controller and to make the command boundary
clear.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np


@dataclass
class LocomotionState:
    t: float
    base_position: np.ndarray
    base_rpy: np.ndarray
    base_velocity: np.ndarray
    commanded_velocity: np.ndarray
    left_contact: bool
    right_contact: bool


@dataclass
class InteractionCorrection:
    body_velocity_correction: np.ndarray
    body_pose_correction: np.ndarray
    hand_reference_correction: Optional[np.ndarray] = None
    disturbance_estimate: Optional[np.ndarray] = None


class UnitreeLocomotionAdapter:
    """Thin wrapper around a validated Unitree locomotion stack.

    Expected future implementation:
    - read base/IMU/contact state from Unitree SDK or ROS bridge;
    - send high-level velocity/body/task correction commands if exposed;
    - never bypass Unitree safety or inject raw joint torques by default.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.t0 = time.time()
        self.commanded_velocity = np.zeros(3)

    def start(self) -> None:
        if self.dry_run:
            print("Unitree adapter dry-run: no robot commands will be sent.")
            return
        raise NotImplementedError("Connect this method to the chosen Unitree SDK/ROS bridge.")

    def set_velocity_command(self, vx: float, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        self.commanded_velocity = np.array([vx, vy, yaw_rate], dtype=float)
        if self.dry_run:
            print(f"dry velocity command: vx={vx:.3f}, vy={vy:.3f}, yaw={yaw_rate:.3f}")
            return
        raise NotImplementedError("Send velocity command through the Unitree locomotion stack.")

    def read_state(self) -> LocomotionState:
        t = time.time() - self.t0
        if self.dry_run:
            return LocomotionState(
                t=t,
                base_position=np.array([self.commanded_velocity[0] * t, 0.0, 0.78]),
                base_rpy=np.zeros(3),
                base_velocity=self.commanded_velocity.copy(),
                commanded_velocity=self.commanded_velocity.copy(),
                left_contact=True,
                right_contact=True,
            )
        raise NotImplementedError("Read state from Unitree SDK/ROS bridge.")

    def send_interaction_correction(self, correction: InteractionCorrection) -> None:
        if self.dry_run:
            print(
                "dry correction: "
                f"vel={correction.body_velocity_correction}, "
                f"pose={correction.body_pose_correction}"
            )
            return
        raise NotImplementedError(
            "Map correction to supported Unitree high-level body/hand interface."
        )

    def stop(self) -> None:
        self.set_velocity_command(0.0, 0.0, 0.0)


def main() -> None:
    adapter = UnitreeLocomotionAdapter(dry_run=True)
    adapter.start()
    adapter.set_velocity_command(0.4, 0.0, 0.0)
    state = adapter.read_state()
    correction = InteractionCorrection(
        body_velocity_correction=np.array([0.0, -0.05, 0.0]),
        body_pose_correction=np.array([0.0, 0.0, 0.0]),
        disturbance_estimate=np.array([0.0, 15.0, 0.0]),
    )
    adapter.send_interaction_correction(correction)
    adapter.stop()
    print(state)


if __name__ == "__main__":
    main()
