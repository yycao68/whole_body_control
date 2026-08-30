"""Stage 2 (pivoted) — Interaction Dynamics as a residual ON TOP of the policy.

The Stage-2 gate showed a task-space controller that REPLACES the realizer
destabilizes the pretrained policy gait (its balance is a joint-space closed-loop
property). This module keeps the policy in the loop carrying balance, and adds
the Interaction-Dynamics correction as a POLICY-NATIVE velocity-command bias
(NOT a torque injection — that fought the policy):

    cmd = [0.5 + dvx, dvy, 0]          # nominal walk command + ID bias
    tau = policy(obs(cmd))             # policy realizes the bias by foot placement

The bias is produced by IDResidual. `kind` selects policy / impedance / id_mpc;
`mode` selects the interaction law (transient capture, sustained hold, the CoM-only
`unified` negative baseline, the wrench-gated `wrench` blend, or `oracle`). A
deadband + rate limit keep the layer out of the nominal loop (bias ~ 0 when
undisturbed). External-wrench estimate F_ext = m*c_ddot_xy - GRF_xy gates the
capture-vs-hold blend. See STAGE2_FINDINGS.md for the full result chain and the
V1-V6 validation campaign.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))              # NormalizedMPC, estimator live alongside this file
from run_policy_walk import (               # reuse the exact policy interface
    SCENE, POLICY, SIM_DT, CONTROL_DECIMATION, KPS, KDS, DEFAULT_ANGLES,
    ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE, CMD_SCALE,
    NUM_ACTIONS, NUM_OBS, GAIT_PERIOD, gravity_orientation, pd_control,
)
from normalized_mpc import NormalizedMPC
from interaction_estimator import FilteredAccelerationResidualEstimator

REF = HERE / "reference" / "frozen_walk_seed0.npz"
CONTROLLERS = ("policy", "impedance", "id_mpc")
GROUND = HERE / "g1_description"


def terrain_scene(step_type, h, x_step=1.1):
    """Write a scene with a step the robot walks over (sustained terrain
    interaction). step_up: floor at 0, a raised slab (top +h) for x>=x_step.
    step_down: the robot starts on a raised slab (top +h) for x<=x_step, then
    drops to the floor at 0. Returns (scene_path, init_base_z)."""
    if step_type == "up":
        slab = f'<geom name="step" type="box" pos="{x_step + 2.0} 0 {h/2}" size="2.0 1.0 {h/2}" material="groundplane"/>'
        init_z = 0.793
    else:  # down: slab behind x_step, robot starts on it
        slab = f'<geom name="step" type="box" pos="{x_step - 2.0} 0 {h/2}" size="2.0 1.0 {h/2}" material="groundplane"/>'
        init_z = 0.793 + h
    xml = f'''<mujoco model="g1 terrain">
  <include file="g1_12dof.xml"/>
  <asset>
    <texture type="2d" name="gp" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="gp" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>
    {slab}
  </worldbody>
</mujoco>'''
    p = GROUND / "_terrain_scene.xml"
    p.write_text(xml)
    return p, init_z


def load_reference():
    # The frozen reference is a *regenerable* artifact, and the repository
    # gitignores *.npz, so a fresh clone does not have it. Fail with the exact
    # regeneration command rather than a bare FileNotFoundError -- external
    # review found the run path unreproducible partly for this reason.
    if not REF.exists():
        raise FileNotFoundError(
            f"missing frozen nominal reference: {REF}\n"
            "It is gitignored (*.npz) and must be regenerated once with the "
            "command recorded in FROZEN_PLATFORM.md:\n"
            "    python3 run_policy_walk.py --duration 20 --seeds 0 "
            "--save reference/frozen_walk_seed0.npz\n"
            "That step itself needs the G1 meshes (see check_platform.py)."
        )
    d = np.load(REF)
    return d["t"], d["com"]


def grf_world(model, data):
    """Total ground-reaction force ON the robot in world frame (foot F/T on hw).

    mj_contactForce returns the contact force on geom2 (in the contact frame).
    A contact's geom order is by geom id, so the robot can be geom1 OR geom2
    depending on the environment geom's id (floor id 0 -> robot is geom2 on flat
    ground; a box slab added later -> robot can be geom1). We therefore orient
    each force to be the one acting ON the robot: +fw when the robot is geom2,
    -fw when it is geom1. (The earlier `+fw` for all contacts was correct on flat
    ground only, and gave a sign-flipped GRF on box terrain.)
    """
    F = np.zeros(3)
    for i in range(data.ncon):
        c = data.contact[i]
        w = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, w)
        fw = c.frame.reshape(3, 3).T @ w[:3]
        env1 = model.geom_bodyid[c.geom1] == 0   # world/terrain is body 0
        env2 = model.geom_bodyid[c.geom2] == 0
        if env1 and not env2:
            F += fw                              # robot is geom2: fw acts on robot
        elif env2 and not env1:
            F -= fw                              # robot is geom1: negate
    return F


class IDResidual:
    """Interaction-dynamics correction as a POLICY-NATIVE velocity command bias.

    Instead of injecting CoM torque (which fights the policy), the ID layer
    outputs a planar velocity-command bias [dvx, dvy] added to the policy's
    walk command, so the policy recovers by placing its feet — the exact
    mechanism it is strong at. Output units are m/s.
    """

    def __init__(self, kind, control_dt, mode="transient"):
        self.kind = kind
        self.dt = control_dt
        # 'transient': capture-assist (step toward the fall) for pushes.
        # 'sustained': step AGAINST the disturbance; id_mpc adds an integral term
        #  for offset-free rejection of a constant force. The two disturbance
        #  classes need OPPOSITE command signs, so the mode selects which.
        self.mode = mode
        self.kp_s = 3.0                      # sustained proportional (against)
        self.ki_s = 8.0                      # sustained integral (offset-free)
        self.e_int = np.zeros(2)
        self.est = FilteredAccelerationResidualEstimator(2, control_dt, bandwidth_hz=5.0)
        # Horizon predictor (Eq. 11) on the CoM-error double integrator; used by
        # the id_mpc controller. Offset-free with the estimated residual d_eff.
        # Velocity-dominant (arrest the push, don't chase absolute position, which
        # over-drives a walking robot). NOTE: the offset-free residual integration
        # is a persistent correction — an asset for SUSTAINED disturbances (its
        # design purpose) but a liability for transient forward pushes; see
        # STAGE2_FINDINGS.md. Tuned for the lateral single-support case.
        self.mpc = NormalizedMPC(dim=2, dt=control_dt, horizon=20,
                                 q_pos=15.0, q_vel=100.0, r=0.05,
                                 u_max=np.array([6.0, 6.0]))
        # Capture-assist: command the policy IN the direction of the CoM error-
        # velocity so it steps toward the fall (gets a foot under the CoM), then
        # a gentle position term returns it to nominal once momentum is arrested.
        self.kv = 1.2                        # capture gain (error-velocity -> cmd)
        self.kret = 0.8                      # gentle return-to-nominal gain
        self.k_map = 0.12                    # maps MPC accel -> capture velocity cmd
        self.vmax = 0.6                      # [m/s] command-bias clamp
        # Lateral-focus: the policy is already sagittally robust, so a forward
        # capture command over-drives its fast forward walk and adds falls. Keep
        # the correction on the lateral (y) channel; attenuate forward (x).
        self.axis_scale = np.array([0.2, 1.0])
        self.v = np.zeros(2)
        self.d_int = np.zeros(2)             # leaky-integrated interaction residual
        # Persistence discriminator: capture-first, escalate to hold only if the
        # deviation OUTLASTS a transient recovery. A transient push is arrested by
        # capture and returns under the deadband within ~1.5 s; a sustained force
        # keeps |e| above the deadband indefinitely. (Residual-magnitude fails
        # here: a transient push's recovery keeps the residual nonzero, so it
        # would masquerade as sustained.)
        self.t_dev = 0.0                     # continuous time |e| has exceeded deadband
        self.t0 = 1.0                        # pure capture below this
        self.t_ramp = 1.0                    # ramp capture->hold over this
        # external-wrench discriminator thresholds [N]: |F_ext|<f_lo -> capture,
        # >f_hi -> hold. Sustained forces settle near their magnitude; a transient
        # push's estimate decays to ~0 within ~0.6 s.
        self.f_lo = 3.0
        self.f_hi = 6.0
        # wrench-persistence: how long |F_ext| has stayed above f_thresh. A
        # transient push clears it within ~0.4 s (force gone); a sustained force
        # keeps it growing. Cleaner than the deviation timer because F_ext falls
        # to ~0 the moment the force ends, even mid-recovery.
        self.t_force = 0.0
        self.f_thresh = 3.0
        self.tf0 = 0.4                       # pure capture below this
        self.tf_ramp = 0.4
        # Capture GATE (fixes capture-runaway): an INTERACTION-CONFIDENCE criterion.
        # Capture (interaction-oriented behaviour) is enabled only when the wrench
        # estimate rises a confidence margin above its OWN noise floor — i.e. when
        # there is statistical evidence of a real external interaction; otherwise
        # the layer defers to the policy's internal stabilisation. The threshold
        # is NOT a tuned constant: the noise floor sigma_f is estimated online from
        # |F_ext| during quiescent walking (|e|<deadband), and f_cap = f_floor +
        # k_conf * sigma_f (k_conf sigmas above noise). So it self-calibrates to
        # the sensing conditions. g_cap latches on evidence and is held open while
        # the robot is still recovering, so a transient push keeps full capture
        # through its recovery while sub-threshold noise never engages it.
        self.g_cap = 0.0
        self.f_floor = 1.0                   # [N] minimum meaningful force (estimator resolution)
        self.k_conf = 3.0                    # confidence margin (sigmas above the wrench noise floor)
        self.sig2_f = self.f_floor ** 2      # online EWMA of |F_ext|^2 during quiescence
        self.a_sig = 1.0 - np.exp(-2 * np.pi * 0.3 * control_dt)   # ~0.3 Hz noise tracker
        self.last_f_cap = self.f_floor
        self.cap_hold = 1.6                  # [s] capture-enable decay window
        # Stay OUT of the nominal loop until a real disturbance moves the CoM
        # error past a deadband (else it closes a loop on the walking sway).
        self.deadband = 0.03                 # [m]
        self.slew = 6.0                      # [m/s per s] command-bias rate limit

    def _apply(self, target):
        step = self.slew * self.dt
        self.v = self.v + np.clip(target - self.v, -step, step)
        self.v = np.clip(self.v, -self.vmax, self.vmax)
        return self.v

    def update(self, com_xy, vcom_xy, com_ref_xy, vcom_ref_xy, f_ext=None, p_oracle=None):
        e = com_xy - com_ref_xy
        edot = vcom_xy - vcom_ref_xy
        emag = np.linalg.norm(e)
        e_eff = e * max(0.0, emag - self.deadband) / (emag + 1e-9)
        d_eff = self.est.step(edot, np.zeros(2), np.zeros(2)).effective
        # capture-vs-hold weight p in [0,1]. Prefer the external-wrench estimate
        # (persistent only while a real force acts); fall back to the deviation
        # timer if no wrench estimate is supplied.
        if self.mode == "oracle" and p_oracle is not None:
            p = float(p_oracle)                  # upper bound: true class, no delay
        elif self.mode == "wrench" and f_ext is not None:
            # persistence timer on the external-wrench estimate; DECAYS (not hard
            # reset) below threshold so process-noise dips don't drop a latched
            # sustained hold, while a transient force still clears it in ~0.3 s.
            if float(np.linalg.norm(f_ext)) > self.f_thresh:
                self.t_force += self.dt
            else:
                self.t_force = max(0.0, self.t_force - 2.0 * self.dt)
            p = float(np.clip((self.t_force - self.tf0) / self.tf_ramp, 0.0, 1.0))
        else:
            if emag >= self.deadband:
                self.t_dev += self.dt
            else:
                self.t_dev = 0.0
            p = float(np.clip((self.t_dev - self.t0) / self.t_ramp, 0.0, 1.0))
        self.last_p = p
        # capture-enable gate on interaction confidence: f_cap = f_floor + k_conf*
        # sigma_f self-calibrates to the wrench noise floor (estimated online in
        # quiescence). Latch on confident evidence; hold open while still recovering
        # (deviation large) so a long capture recovery keeps full capture; decay
        # once recovered. Sub-threshold noise never latches it -> no runaway.
        if f_ext is not None:
            fmag = float(np.linalg.norm(f_ext))
            if emag < self.deadband:             # quiescent -> track the noise floor
                self.sig2_f = (1 - self.a_sig) * self.sig2_f + self.a_sig * fmag * fmag
            f_cap = self.f_floor + self.k_conf * np.sqrt(self.sig2_f)
            self.last_f_cap = f_cap
            if fmag > f_cap:
                self.g_cap = 1.0
            elif emag < 1.5 * self.deadband:     # recovered -> release the gate
                self.g_cap = max(0.0, self.g_cap - self.dt / self.cap_hold)
            # else: recovery from a detected disturbance in progress -> hold g_cap
        else:
            self.g_cap = 1.0                    # no wrench available: unguarded
        if self.kind == "policy" or emag < self.deadband:
            self.e_int *= 0.9
            return self._apply(np.zeros(2))     # nominal: decay bias to zero

        if self.mode == "sustained":
            # HOLD specialist: step AGAINST the force. impedance = proportional
            # (droops); id_mpc adds the offset-free integral.
            self.e_int = 0.98 * self.e_int + e_eff * self.dt
            if self.kind == "impedance":
                return self._apply((-self.kp_s * e_eff) * self.axis_scale)
            return self._apply((-self.kp_s * e_eff - self.ki_s * self.e_int) * self.axis_scale)

        if self.kind == "impedance":
            # CAPTURE specialist (non-adaptive), for transient pushes; gated so it
            # cannot amplify sub-threshold (noise) drift into a runaway.
            return self._apply((self.g_cap * (self.kv * edot - self.kret * e_eff)) * self.axis_scale)

        # id_mpc: horizon-MPC accel mapped to the capture command, capture-gated
        a_e = self.mpc.solve(np.concatenate([e_eff, edot]), d_hat=d_eff)
        u_capture = -self.k_map * a_e * self.g_cap
        if self.mode in ("unified", "wrench", "oracle"):
            # ADDITIVE: capture is ALWAYS on (handles transient momentum whether or
            # not a force is currently sensed); the offset-free HOLD is ADDED and
            # gated by p (only a sustained force). When hold arrests a sustained
            # drift, edot -> 0 so the capture term self-vanishes; on a transient
            # the wrench p decays and only capture remains. mode='wrench' gates on
            # the external-wrench estimate; mode='unified' on the deviation timer
            # (NEGATIVE baseline — see STAGE2_FINDINGS.md).
            self.e_int = 0.98 * self.e_int + p * e_eff * self.dt
            u_hold = -self.kp_s * e_eff - self.ki_s * self.e_int
            return self._apply(((1.0 - p) * u_capture + p * u_hold) * self.axis_scale)
        # default "transient": capture specialist (the push-recovery result)
        return self._apply(u_capture * self.axis_scale)


def run(controller, push_n=0.0, push_t=2.5, push_dir=(0, 1), push_dur=0.15,
        duration=8.0, seed=0, push_phase="time", process_noise=0.0,
        scene_path=None, init_base_z=0.793, id_mode="transient", force_override=None,
        grf_bias=(0.0, 0.0), grf_noise=0.0):
    """push_phase: 'time' fires at push_t; 'DS'/'SS' gate the push on the first
    measured double-/single-support after push_t (paper's phase-locked protocol).
    process_noise: std [N] of a seeded lateral force on the torso each control
    step, so seeds diverge and per-cell statistics are meaningful.
    scene_path: terrain scene (a step to walk over) instead of flat SCENE."""
    model = mujoco.MjModel.from_xml_path(str(scene_path or SCENE)); model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    ref_t, ref_com = load_reference()
    ref_vcom = np.gradient(ref_com, ref_t[1] - ref_t[0], axis=0)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    lfoot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rfoot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    mass = float(model.body_mass[1:].sum())

    def foot_support():
        c = [False, False]
        for ci in range(data.ncon):
            for gi in (data.contact[ci].geom1, data.contact[ci].geom2):
                b = model.geom_bodyid[gi]
                if b == lfoot or model.body_parentid[b] == lfoot: c[0] = True
                if b == rfoot or model.body_parentid[b] == rfoot: c[1] = True
        return c

    data.qpos[:3] = [0, 0, init_base_z]; data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = DEFAULT_ANGLES
    rng = np.random.default_rng(seed)
    data.qvel[6:] += rng.normal(0, 2e-4, size=model.nv - 6)
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(str(POLICY))
    idr = IDResidual(controller, CONTROL_DECIMATION * SIM_DT, mode=id_mode)

    def com_state():
        com = (model.body_mass[1:, None] * data.xipos[1:]).sum(0) / mass
        Jc = np.zeros((3, model.nv)); mujoco.mj_jacSubtreeCom(model, data, Jc, pelvis)
        return com, Jc, Jc @ data.qvel

    n = int(round(duration / SIM_DT))
    action = np.zeros(NUM_ACTIONS, np.float32)
    target = DEFAULT_ANGLES.copy(); obs = np.zeros(NUM_OBS, np.float32)
    settle = 0.5
    counter = 0
    v_bias = np.zeros(2)                       # ID velocity-command bias [dvx, dvy]
    dev = []; roll = []; pitch = []; latv = []; ey = []; comy = []; fell = None
    push_vec = np.array([push_dir[0], push_dir[1], 0.0], float)
    push_vec = push_vec / (np.linalg.norm(push_vec) + 1e-9)
    push_start = None                          # actual push onset time (gated)
    pnoise = np.zeros(3)
    # External-wrench estimator (CoM linear-momentum residual). Full 3D balance
    # is  m*c_ddot = m*g + sum_i F_contact_i + F_ext, so
    #     F_ext = m*(c_ddot - g) - sum_i F_contact_i.
    # We use ONLY the HORIZONTAL (xy) components, where g has no component, so the
    # gravity term drops out exactly:  F_ext_xy = m*c_ddot_xy - GRF_xy  (no -m*g
    # needed here; the full 3D force WOULD need it). ~0 during a transient
    # recovery, persistent under a sustained force — the unified discriminator.
    # Inputs here are simulation-clean (kinematic CoM velocity from full state;
    # MuJoCo-exact contact forces); on hardware this is the momentum residual from
    # IMU + estimated CoM velocity + foot six-axis wrenches (not yet hw-validated).
    vcom_prev = None; f_ext = np.zeros(2)
    a_ext = 1.0 - np.exp(-2 * np.pi * 3.0 * SIM_DT)

    for k in range(n):
        t = k * SIM_DT
        # policy carries the gait; NO torque injection in command-modulation mode
        tau = pd_control(target if t >= settle else DEFAULT_ANGLES,
                         data.qpos[7:], KPS, data.qvel[6:], KDS)
        com, Jc, vcom = com_state()
        cddot = np.zeros(3) if vcom_prev is None else (vcom - vcom_prev) / SIM_DT
        vcom_prev = vcom.copy()
        # measured GRF = true + horizontal sensor bias + noise (V5 robustness)
        grf_meas = grf_world(model, data)[:2] + np.asarray(grf_bias, float)
        if grf_noise > 0:
            grf_meas = grf_meas + rng.normal(0, grf_noise, size=2)
        f_ext = (1 - a_ext) * f_ext + a_ext * (mass * cddot[:2] - grf_meas)
        data.xfrc_applied[:] = 0.0
        # seeded lateral process noise (refresh at control rate) so seeds diverge
        if process_noise > 0 and t >= settle:
            if counter % CONTROL_DECIMATION == 0:
                pnoise = np.array([0.0, rng.normal(0, process_noise), 0.0])
            data.xfrc_applied[pelvis, :3] += pnoise
        # arbitrary force profile (ramped/intermittent, for held-out V3 tests)
        if force_override is not None:
            if t >= push_t and push_start is None:
                push_start = push_t
            if t >= push_t:
                data.xfrc_applied[pelvis, :2] += np.asarray(force_override(t - push_t), float)
        else:
            # phase-locked external torso push, gated on measured support (paper protocol)
            if push_n > 0 and push_start is None and t >= push_t:
                sup = foot_support()
                n_c = int(sup[0]) + int(sup[1])
                if (push_phase == "time"
                        or (push_phase == "DS" and n_c == 2)
                        or (push_phase == "SS" and n_c == 1)):
                    push_start = t
            if push_start is not None and push_start <= t < push_start + push_dur:
                data.xfrc_applied[pelvis, :3] += push_n * push_vec
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        if t >= settle and counter % CONTROL_DECIMATION == 0:
            qj = (data.qpos[7:] - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj = data.qvel[6:] * DOF_VEL_SCALE
            grav = gravity_orientation(data.qpos[3:7]); omega = data.qvel[3:6] * ANG_VEL_SCALE
            phase = (counter * SIM_DT) % GAIT_PERIOD / GAIT_PERIOD
            # ID modulates the policy's walk command (the policy recovers by
            # foot placement); nominal cmd is [0.5, 0, 0].
            cmd = np.array([0.5 + v_bias[0], v_bias[1], 0.0])
            obs[:3] = omega; obs[3:6] = grav; obs[6:9] = cmd * CMD_SCALE
            obs[9:21] = qj; obs[21:33] = dqj; obs[33:45] = action
            obs[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * ACTION_SCALE + DEFAULT_ANGLES
            # ID layer update at the control rate -> velocity-command bias
            com, Jc, vcom = com_state()
            ridx = min(k, len(ref_t) - 1)
            # oracle: knows the true disturbance class from the (known) force
            # duration, with zero detection delay — the upper bound for V2.
            p_oracle = 1.0 if (push_dur >= 1.0 and push_start is not None
                               and push_start <= t < push_start + push_dur) else 0.0
            v_bias = idr.update(com[:2], vcom[:2], ref_com[ridx, :2], ref_vcom[ridx, :2],
                                f_ext=f_ext, p_oracle=p_oracle)

        ridx = min(k, len(ref_t) - 1)
        qw, qx, qy, qz = data.qpos[3:7]
        roll_a = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
        pitch_a = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        dev.append(np.hypot(*(com[:2] - ref_com[ridx, :2])))
        roll.append(abs(float(roll_a)))
        pitch.append(abs(float(pitch_a)))
        latv.append(abs(float(vcom[1])))
        ey.append(float(com[1] - ref_com[ridx, 1]))
        comy.append(float(com[1]))
        up = gravity_orientation(data.qpos[3:7])[2]
        if fell is None and t >= settle and (data.qpos[2] < 0.45 or up > -0.5):
            fell = t
    dev = np.array(dev); roll = np.array(roll); pitch = np.array(pitch)
    latv = np.array(latv); ey = np.array(ey); comy = np.array(comy)
    # measure from the ACTUAL (gated) push onset
    pi = int((push_start if push_start is not None else push_t) / SIM_DT)
    pi = min(pi, len(roll) - 1)
    # Sustained-force metrics over the last 1 s of the force window (steady state):
    #  lat_off  = mean(CoM_y − recorded-ref_y)   — has a ~30 mm phase-drift floor
    #             (robot desyncs from the fixed recorded reference under noise).
    #  lat_drift = mean(CoM_y | force) − mean(CoM_y | 1 s before force onset)
    #             — SELF-REFERENCED, floor-free (no recorded reference), isolates
    #             the force-induced lateral shift within one run. Use this one.
    lat_off = 0.0; lat_drift = 0.0
    if push_start is not None and push_dur >= 1.0:
        a = int((push_start + push_dur - 1.0) / SIM_DT); b = int((push_start + push_dur) / SIM_DT)
        b = min(b, len(ey)); a = max(0, min(a, b - 1))
        lat_off = float(np.mean(ey[a:b])) * 1000 if b > a else 0.0
        ba = int((push_start - 1.0) / SIM_DT); bb = int(push_start / SIM_DT)
        ba = max(0, ba); bb = max(ba + 1, min(bb, len(comy)))
        if b > a and bb > ba:
            lat_drift = (float(np.mean(comy[a:b])) - float(np.mean(comy[ba:bb]))) * 1000
    lo = pi if push_n > 0 else int(1.0 / SIM_DT)   # terrain: skip the initial settle
    peak_roll = float(roll[lo:].max()) * 180 / np.pi
    peak_pitch = float(pitch[lo:].max()) * 180 / np.pi
    nom_latv = float(np.percentile(latv[:pi], 95)) if pi > 0 else 0.2
    band = max(1.5 * nom_latv, 0.15)
    rec = None
    if push_n > 0 and fell is None:
        w = int(0.3 / SIM_DT)
        for j in range(pi, len(latv) - w):
            if np.all(latv[j:j + w] < band):
                rec = (j - pi) * SIM_DT; break
    return {"fell": fell is not None, "peak_roll_deg": peak_roll,
            "peak_pitch_deg": peak_pitch, "peak_tilt_deg": max(peak_roll, peak_pitch),
            "recovery_s": rec, "survived": duration if fell is None else fell,
            "lat_offset_mm": lat_off, "lat_drift_mm": lat_drift, "push_onset": push_start}


def stats_cell(controller, push_n, push_dir, push_phase, seeds, process_noise, id_mode="transient"):
    """Run one (condition, controller) cell over seeds; return per-cell stats."""
    rolls, recs, falls = [], [], 0
    for s in seeds:
        r = run(controller, push_n=push_n, push_dir=push_dir, push_phase=push_phase,
                seed=s, process_noise=process_noise, id_mode=id_mode)
        rolls.append(r["peak_roll_deg"])
        falls += int(r["fell"])
        if r["recovery_s"] is not None:
            recs.append(r["recovery_s"])
    rolls = np.array(rolls)
    upright = rolls[rolls < 90]                # medians over non-fallen seeds
    return {"peak_roll_med": float(np.median(upright)) if len(upright) else float("nan"),
            "rec_med": float(np.median(recs)) if recs else None,
            "recovered": len(recs), "falls": falls, "n": len(seeds)}


def run_stats(push_n, n_seeds, process_noise, id_mode="transient"):
    seeds = list(range(1000, 1000 + n_seeds))
    conditions = [("lateral", (0, 1), "DS"), ("lateral", (0, 1), "SS"),
                  ("forward", (1, 0), "DS"), ("forward", (1, 0), "SS")]
    print(f"Phase-locked push study: {push_n:.0f} N, {n_seeds} seeds/cell, "
          f"process_noise={process_noise:.0f} N, id_mode={id_mode}")
    print(f"{'condition':16s}{'controller':11s}{'peak_roll(med)':>15s}"
          f"{'recovery(med)':>15s}{'recovered':>11s}{'falls':>7s}")
    for dname, pvec, phase in conditions:
        for c in CONTROLLERS:
            r = stats_cell(c, push_n, pvec, phase, seeds, process_noise, id_mode)
            rec = f"{r['rec_med']:.2f}s" if r["rec_med"] is not None else "—"
            print(f"{dname+','+phase:16s}{c:11s}{r['peak_roll_med']:12.1f}deg"
                  f"{rec:>15s}{r['recovered']:>7d}/{r['n']:<3d}{r['falls']:>4d}/{r['n']:<2d}",
                  flush=True)


def run_terrain(heights, n_seeds, process_noise):
    seeds = list(range(1000, 1000 + n_seeds))
    print(f"Step terrain study: {n_seeds} seeds/cell, process_noise={process_noise:.0f} N")
    print(f"{'terrain':14s}{'controller':11s}{'peak_pitch(med)':>16s}"
          f"{'peak_roll(med)':>15s}{'falls':>8s}")
    for stype in ("up", "down"):
        for h in heights:
            scene, iz = terrain_scene(stype, h)
            for c in CONTROLLERS:
                tilts, rolls, falls = [], [], 0
                for s in seeds:
                    r = run(c, push_n=0.0, duration=8.0, seed=s, process_noise=process_noise,
                            scene_path=scene, init_base_z=iz)
                    falls += int(r["fell"])
                    if not r["fell"]:
                        tilts.append(r["peak_pitch_deg"]); rolls.append(r["peak_roll_deg"])
                mp = np.median(tilts) if tilts else float("nan")
                mr = np.median(rolls) if rolls else float("nan")
                print(f"{'step-'+stype+f' {int(h*1000)}mm':14s}{c:11s}{mp:13.1f}deg"
                      f"{mr:12.1f}deg{falls:>5d}/{n_seeds:<2d}", flush=True)


def run_sustained(forces, n_seeds, process_noise, hold=3.0, id_mode="sustained"):
    """Constant lateral force held for `hold` s during walking — the offset-free
    test. Reports the steady-state lateral offset (last 1 s of the force) and
    falls; the offset-free ID-MPC should hold near the deadband while policy
    and impedance droop in the force direction."""
    seeds = list(range(1000, 1000 + n_seeds))
    print(f"Sustained lateral-force study: {hold:.0f} s hold, {n_seeds} seeds/cell, "
          f"process_noise={process_noise:.0f} N")
    print(f"{'force':9s}{'controller':11s}{'lat_offset(med)':>16s}{'falls':>8s}")
    for F in forces:
        for c in CONTROLLERS:
            offs, falls = [], 0
            for s in seeds:
                r = run(c, push_n=F, push_t=3.0, push_dir=(0, 1), push_dur=hold,
                        push_phase="time", duration=3.0 + hold + 2.0, seed=s,
                        process_noise=process_noise, id_mode=id_mode)
                falls += int(r["fell"])
                if not r["fell"]:
                    offs.append(abs(r["lat_offset_mm"]))
            med = np.median(offs) if offs else float("nan")
            print(f"{int(F):>4d} N   {c:11s}{med:12.0f} mm{falls:>5d}/{n_seeds:<2d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--push-dir", default="lateral", choices=["lateral", "forward"])
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    ap.add_argument("--stats", action="store_true", help="phase-locked push seed study")
    ap.add_argument("--terrain", action="store_true", help="step-up/step-down seed study")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.03, 0.05])
    ap.add_argument("--sustained", action="store_true", help="sustained lateral-force study")
    ap.add_argument("--forces", type=float, nargs="+", default=[20, 40, 60])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--process-noise", type=float, default=15.0)
    ap.add_argument("--unified", action="store_true",
                    help="timer-discriminated unified id_mpc (documented to underperform)")
    ap.add_argument("--wrench", action="store_true",
                    help="external-wrench-discriminated unified id_mpc")
    args = ap.parse_args()
    id_mode = "unified" if args.unified else ("wrench" if args.wrench else None)
    if args.stats:
        run_stats(args.push, args.seeds, args.process_noise,
                  id_mode=id_mode or "transient")
        return
    if args.terrain:
        run_terrain(args.heights, args.seeds, args.process_noise)
        return
    if args.sustained:
        run_sustained(args.forces, args.seeds, args.process_noise,
                      id_mode=id_mode or "sustained")
        return
    pd = (0, 1) if args.push_dir == "lateral" else (1, 0)
    print(f"push={args.push}N dir={args.push_dir}")
    for c in args.controllers:
        r = run(c, push_n=args.push, push_dir=pd, duration=args.duration)
        rec = f"{r['recovery_s']:.2f}s" if r["recovery_s"] is not None else "—"
        print(f"  {c:10s}: fell={r['fell']} survived={r['survived']:.2f}s "
              f"peak_roll={r['peak_roll_deg']:5.1f}deg recovery={rec}", flush=True)


if __name__ == "__main__":
    main()
