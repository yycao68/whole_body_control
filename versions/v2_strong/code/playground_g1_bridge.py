"""
Bridge between MuJoCo Playground G1 locomotion and the arm interaction MPC.

MuJoCo Playground supplies the G1 joystick locomotion environment.  Its policy
action is a 29-vector of normalized joint-position offsets:

    motor_targets = default_pose + action * action_scale

This bridge keeps a locomotion policy in charge of the legs/waist/left arm and
overwrites only the right-arm entries with targets produced by our interaction
layer.  The wheel currently ships environments but no pretrained G1 checkpoints,
so this file is a runnable integration smoke test plus the adapter that a trained
policy/checkpoint can call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import mjx_env, registry


HERE = Path(__file__).resolve().parent
EXTERNAL_DEPS = HERE / "external_deps"
MENAGERIE = EXTERNAL_DEPS / "mujoco_menagerie"

ENV_NAME = "G1JoystickFlatTerrain"
RIGHT_ARM_INDICES = np.arange(22, 29)
LEFT_ARM_INDICES = np.arange(15, 22)
LEG_WAIST_INDICES = np.arange(0, 15)


def configure_playground_paths() -> None:
    """Redirect Playground external assets to the project-local Menagerie clone."""
    mjx_env.EXTERNAL_DEPS_PATH = EXTERNAL_DEPS
    mjx_env.MENAGERIE_PATH = MENAGERIE


def load_g1_env():
    """Load Playground's G1 joystick environment with deterministic settings."""
    configure_playground_paths()
    return registry.load(
        ENV_NAME,
        config_overrides={
            "impl": "jax",              # The installed warp backend is not usable here.
            "noise_config.level": 0.0,
            "push_config.enable": False,
        },
    )


class PlaygroundG1ArmBridge:
    """Action-space adapter for overriding G1 right-arm targets."""

    def __init__(self, env):
        self.env = env
        self.default_pose = np.asarray(env._default_pose)
        self.action_scale = float(env._config.action_scale)
        self.right_arm_indices = RIGHT_ARM_INDICES.copy()
        self.left_arm_indices = LEFT_ARM_INDICES.copy()
        self.leg_waist_indices = LEG_WAIST_INDICES.copy()

    def target_to_action(self, joint_target: np.ndarray) -> np.ndarray:
        return (np.asarray(joint_target) - self.default_pose) / self.action_scale

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        return self.default_pose + np.asarray(action) * self.action_scale

    def override_right_arm_target(
        self,
        locomotion_action: np.ndarray,
        right_arm_target: np.ndarray,
    ) -> np.ndarray:
        """Return action with right-arm entries replaced by desired q targets."""
        action = np.asarray(locomotion_action, dtype=float).copy()
        action[self.right_arm_indices] = (
            np.asarray(right_arm_target, dtype=float)
            - self.default_pose[self.right_arm_indices]
        ) / self.action_scale
        return np.clip(action, -1.0, 1.0)

    def hold_right_arm_action(self, locomotion_action: np.ndarray) -> np.ndarray:
        """Smoke-test override: hold the right arm at Playground's default pose."""
        return self.override_right_arm_target(
            locomotion_action,
            self.default_pose[self.right_arm_indices],
        )

    def right_arm_qpos(self, state) -> np.ndarray:
        return np.asarray(state.data.qpos[7:])[self.right_arm_indices]

    def contacts(self, state) -> np.ndarray:
        return np.asarray(state.info["last_contact"], dtype=bool)


PolicyFn = Callable[[dict], np.ndarray]


def zero_policy(obs: dict) -> np.ndarray:
    del obs
    return np.zeros(29)


def rollout_with_arm_override(
    policy: PolicyFn = zero_policy,
    steps: int = 50,
    seed: int = 0,
):
    """Run a short Playground rollout with the right arm reserved for our layer."""
    env = load_g1_env()
    bridge = PlaygroundG1ArmBridge(env)
    state = env.reset(jax.random.PRNGKey(seed))
    logs = []
    for k in range(steps):
        base_action = np.asarray(policy(state.obs), dtype=float)
        if base_action.shape != (env.action_size,):
            raise ValueError(f"Policy action must have shape {(env.action_size,)}, got {base_action.shape}")
        action = bridge.hold_right_arm_action(base_action)
        state = env.step(state, jp.asarray(action))
        logs.append({
            "step": k,
            "reward": float(state.reward),
            "done": float(state.done),
            "command": np.asarray(state.info["command"]),
            "contacts": bridge.contacts(state),
            "right_arm_q": bridge.right_arm_qpos(state),
        })
        if float(state.done):
            break
    return env, bridge, logs


if __name__ == "__main__":
    env, bridge, logs = rollout_with_arm_override(steps=20)
    print(f"Loaded {ENV_NAME}: nq={env.mj_model.nq}, nv={env.mj_model.nv}, nu={env.mj_model.nu}")
    print(f"Right-arm action indices: {bridge.right_arm_indices.tolist()}")
    print("No pretrained locomotion checkpoint is bundled with the installed wheel.")
    print("Smoke rollout with zero locomotion policy + right-arm reservation:")
    print(f"  steps={len(logs)} final_done={logs[-1]['done']:.1f} final_reward={logs[-1]['reward']:.4f}")
    print(f"  final_contacts={logs[-1]['contacts'].tolist()}")
    print(f"  final_right_arm_q={np.round(logs[-1]['right_arm_q'], 3).tolist()}")
