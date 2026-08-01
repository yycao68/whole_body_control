# Frozen Walking Platform — Unitree G1 Pretrained Policy

**Status: FROZEN (Stage 1 complete).** Per the change-of-direction plan
(`../../Interaction_Dynamics_Change_Direction_Plan.md`), this is the permanent
validation platform. No further gait optimization is allowed. Interaction
Dynamics must never modify this platform; it is identical for every controller.

## Why we migrated

The in-house DCM/capture-step reference failed the Stage-1 acceptance bar: under
the plan's Step-1 config (nominal MPC = ID OFF, stabilizer ON, phase-sync OFF),
flat-ground walking fell at **3.7–7.7 s (0/5 at the 20 s bar)** — matching the
previously-characterized single-support realizability dead end. Per the plan's
Decision Point, we migrated to the official Unitree MuJoCo locomotion baseline.

## What the platform is

- **Model:** `g1_description/scene.xml` → `g1_12dof.xml` (official Unitree
  12-DoF legs-only G1, from `unitree_rl_gym`), 27 referenced STL meshes.
- **Policy:** `motion.pt` — pretrained TorchScript walking policy from
  `unitree_rl_gym/deploy/pre_train/g1/`.
- **Interface (from `configs/g1.yaml`, mirrored exactly in `run_policy_walk.py`):**
  47-dim observation, 12 leg actions, `action*0.25 + default_angles` → PD
  (kp=[100,100,100,150,40,40]×2, kd=[2,2,2,4,2,2]×2) → torque, 50 Hz control on a
  500 Hz (2 ms) sim. Command `[vx, vy, wz] = [0.5, 0, 0]`.

## Acceptance results (`run_policy_walk.py --duration 20 --seeds 0..9`)

| criterion | bar | result |
|---|---|---|
| flat-ground walking | 20 s | 20.00/20 s |
| successful trials | ≥9/10 | **10/10** |
| swing clearance | sufficient | 4.9–8.3 cm foot lift |
| stepping | no freeze/drag | 28 steps, 104 contact switches |
| single-support ratio | realistic | 91% single / 9% double / 0% flight |
| repeatable | yes | deterministic across seeds |

Mean forward speed 0.477 m/s (cmd 0.5). CoM height 0.682 m (±3.6 mm).

## Frozen nominal reference

`reference/frozen_walk_seed0.npz` — the canonical 20 s walk at 500 Hz. Fields:
`t, base_pos, base_quat, base_linvel, base_angvel, com, qj, dqj, lfoot, rfoot,
contact, action`. Stage 2 (ID) tracks THIS recorded trajectory via the existing
ID-MPC + WBC stack; the platform itself is not re-run per controller.

## Reproduce

```bash
cd whole_body_control/versions/v3/code/unitree_baseline
python3 run_policy_walk.py --duration 20 --seeds 0 1 2 3 4 5 6 7 8 9      # acceptance
python3 run_policy_walk.py --duration 20 --seeds 0 --save reference/frozen_walk_seed0.npz
```

Provenance of downloaded artifacts: `github.com/unitreerobotics/unitree_rl_gym`
(`deploy/pre_train/g1/motion.pt`, `deploy/deploy_mujoco/{deploy_mujoco.py,configs/g1.yaml}`,
`resources/robots/g1_description/{scene.xml,g1_12dof.xml,meshes/*}`).
