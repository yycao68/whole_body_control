# Large assets: what is tracked, and what you must supply

The repository's `.gitignore` excludes `*.STL` and `*.npz`, but **v5's G1 meshes
are an explicit exception and ARE committed** (see the `!versions/v5/...` rule at
the end of the root `.gitignore`). They are the only part of the platform that
cannot be reconstructed from any public source, so they are carried in-tree.

After cloning, the one remaining step is regenerating the frozen nominal
reference (§2). Everything else is present.

External review (2026-08-30) could not run the publication path at all, because
at that time neither the meshes nor the reference were available. Run the
preflight first — it reports anything missing at once with the exact fix:

```bash
cd code
python3 check_platform.py
```

## Summary

| Asset | Tracked in git? | Size | How to obtain |
| --- | --- | ---: | --- |
| `motion.pt` (frozen RL policy) | **yes** | 145,745 B | already present |
| `g1_description/meshes/*.STL` (27 files) | **yes** (gitignore exception) | 25.2 MB | already present |
| `reference/frozen_walk_seed0.npz` | no (`*.npz`) | 3,552,081 B | regenerate (command below) |

## 1. Python packages

There is no lockfile in this repository; `requirements.txt` (added alongside
this file) lists the minimum set. Versions used for the committed results are
recorded there as comments.

```bash
pip install -r requirements.txt
```

## 2. Frozen nominal reference — `reference/frozen_walk_seed0.npz`

Regenerable, so it is deliberately not committed. It needs the meshes and
`torch` to exist first.

```bash
python3 run_policy_walk.py --duration 20 --seeds 0 --save reference/frozen_walk_seed0.npz
```

- size 3,552,081 bytes
- `sha256 ff5187be469ef53d499af984bc250272d2fb7bc14c35eb06d30cbc8db2ef1e33`
- canonical 20 s walk at 500 Hz → 10,000 samples

Fields (all `float64`, first axis 10000):

| field | shape | field | shape |
| --- | --- | --- | --- |
| `t` | (10000,) | `qj` | (10000, 12) |
| `base_pos` | (10000, 3) | `dqj` | (10000, 12) |
| `base_quat` | (10000, 4) | `action` | (10000, 12) |
| `base_linvel` | (10000, 3) | `lfoot` | (10000, 3) |
| `base_angvel` | (10000, 3) | `rfoot` | (10000, 3) |
| `com` | (10000, 3) | `contact` | (10000, 2) |

`stage2_id_on_policy.load_reference()` reads only `t` and `com`.

## 3. G1 meshes — `g1_description/meshes/` (27 files, 25.2 MB)

`g1_description/g1_12dof.xml` declares `meshdir="meshes/"` and references the 27
files below. This is the **12-DoF legs-only G1 from `unitree_rl_gym`**, not the
MuJoCo Menagerie model.

**`robot_descriptions` / mujoco_menagerie is NOT a sufficient source.** 24 of the
27 filenames also appear in `mujoco_menagerie/unitree_g1/assets`, but three do
not, so the v3/v4 auto-fetch fallback cannot be reused here:

- `torso_link_23dof_rev_1_0.STL` — no Menagerie equivalent
- `left_wrist_roll_rubber_hand.STL` — Menagerie names it `left_rubber_hand.STL`
- `right_wrist_roll_rubber_hand.STL` — Menagerie names it `right_rubber_hand.STL`

**These 27 files are committed to this repository** precisely because of that
gap: with three of them unavailable from Menagerie -- and those three being the
largest, 14.8 MB of the 25.2 MB total -- no auto-fetch fallback can rebuild the
set, so carrying them in-tree is the only way v5 reproduces from a clean clone.
(v3/v4 meshes are *not* committed; `robot_descriptions` supplies those.)

If you ever need to replace them, copy
`resources/robots/g1_description/meshes/` from a `unitree_rl_gym` checkout into
`code/g1_description/meshes/`, or point `$G1_MESH_DIR` at a directory
containing all 27. The manifest below lets you verify any such copy.

Verify with:

```bash
python3 check_platform.py
```

### Manifest

`sha256` is truncated to 16 hex chars; sizes in bytes.

| file | size | sha256 (16) | also in menagerie? |
| --- | ---: | --- | --- |
| `head_link.STL` | 932,784 | `005fb67fbd3eff94` | menagerie |
| `left_ankle_pitch_link.STL` | 71,184 | `d49e3abc6f5b12e5` | menagerie |
| `left_ankle_roll_link.STL` | 653,384 | `c4092af943141d4d` | menagerie |
| `left_elbow_link.STL` | 88,784 | `fa752198accd104d` | menagerie |
| `left_hip_pitch_link.STL` | 181,684 | `4725168105ee768e` | menagerie |
| `left_hip_roll_link.STL` | 192,184 | `91f25922ee8a7c31` | menagerie |
| `left_hip_yaw_link.STL` | 296,284 | `a16d88aa6ddac808` | menagerie |
| `left_knee_link.STL` | 854,884 | `8d92b9e3d3a63676` | menagerie |
| `left_shoulder_pitch_link.STL` | 176,784 | `f0d1cfd02fcf0d42` | menagerie |
| `left_shoulder_roll_link.STL` | 400,284 | `fb9df21687773522` | menagerie |
| `left_shoulder_yaw_link.STL` | 249,184 | `1aa97e9748e92433` | menagerie |
| `left_wrist_roll_rubber_hand.STL` | 3,484,884 | `e81030abd023bd9e` | unitree_rl_gym ONLY |
| `logo_link.STL` | 243,384 | `8571a0a19bc4916f` | menagerie |
| `pelvis.STL` | 1,060,884 | `5ba6bbc888e63055` | menagerie |
| `pelvis_contour_link.STL` | 1,805,184 | `5cc5c2c7a312329e` | menagerie |
| `right_ankle_pitch_link.STL` | 71,184 | `15be426539ec1be7` | menagerie |
| `right_ankle_roll_link.STL` | 653,784 | `4b66222ea56653e6` | menagerie |
| `right_elbow_link.STL` | 88,784 | `1be925d7aa268bb8` | menagerie |
| `right_hip_pitch_link.STL` | 181,284 | `e4f3c99d7f4a7d34` | menagerie |
| `right_hip_roll_link.STL` | 192,684 | `4c254ef66a356f49` | menagerie |
| `right_hip_yaw_link.STL` | 296,284 | `e479c2936ca2057e` | menagerie |
| `right_knee_link.STL` | 852,284 | `63c4008449c9bbe7` | menagerie |
| `right_shoulder_pitch_link.STL` | 176,784 | `24cdb387e0128dfe` | menagerie |
| `right_shoulder_roll_link.STL` | 401,884 | `962b97c48f9ce9e8` | menagerie |
| `right_shoulder_yaw_link.STL` | 249,984 | `a0b76489271da0c7` | menagerie |
| `right_wrist_roll_rubber_hand.STL` | 3,481,584 | `0729aff1ac4356f9` | unitree_rl_gym ONLY |
| `torso_link_23dof_rev_1_0.STL` | 7,825,434 | `3cd0d56fde14b73c` | unitree_rl_gym ONLY |
