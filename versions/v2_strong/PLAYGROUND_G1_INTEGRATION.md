# MuJoCo Playground G1 Integration Status

This folder now contains the first integration hook between the paper's
floating-base arm interaction MPC and MuJoCo Playground's Unitree G1 locomotion
environment.

## What Works

- `playground` package is installed and imports as `mujoco_playground`.
- MuJoCo Playground's `G1JoystickFlatTerrain` environment loads locally.
- The project-local MuJoCo Menagerie clone is pinned to the commit expected by
  the installed Playground package:

  ```text
  1b86ece576591213e2b666ebf59508454200ca97
  ```

- The bridge script runs a smoke rollout:

  ```bash
  python3 whole_body_control/versions/v2_strong/code/playground_g1_bridge.py
  ```

- The action interface has been verified:

  ```text
  action_size = 29
  motor_targets = default_pose + action * action_scale
  right-arm action indices = [22, 23, 24, 25, 26, 27, 28]
  ```

This means a locomotion policy can control the legs/waist/left arm, while the
paper's interaction MPC overwrites only the right-arm joint targets.

## Current Limitation

The installed Playground wheel does **not** bundle pretrained G1 locomotion
checkpoints:

```text
*.ckpt       0
*.onnx       0
*.pkl        0
*.npz        0
*.msgpack    0
*.safetensors 0
```

Therefore, we can use Playground as the validated locomotion environment/API,
but we still need either:

- a trained G1 joystick policy checkpoint, or
- a training run using Playground's PPO scripts.

Until one of those exists, the paper should not claim validation on a
pretrained dynamic walking policy.

## Integration Path

1. Train or obtain a `G1JoystickFlatTerrain` policy.
2. Use the policy to generate a 29-D locomotion action.
3. Pass the action through `PlaygroundG1ArmBridge`.
4. Replace right-arm action entries with the interaction-MPC target.
5. Step the Playground G1 environment.
6. Log:
   - walking command tracking,
   - foot contacts,
   - arm tracking RMS/peak error,
   - pHRI disturbance response,
   - contact-mode transitions.

## Important Environment Note

Installing Playground upgraded `mujoco` to `3.10.0`, which conflicts with
`myosuite`'s requirement of `mujoco<3.7`. The WBC MuJoCo scripts should still be
checked after this dependency change.
