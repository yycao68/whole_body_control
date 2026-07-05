"""
Sinusoidal CPG Gait Generator.

Produces desired leg joint angles for a 1 Hz walking-in-place gait
that creates smooth contact-mode transitions for demonstrating the
covariance-inflation protocol (Section VI of paper).
"""

import numpy as np


# Leg joint names in the order expected by WBCController.compute()
LEG_JOINT_NAMES = [
    'left_hip_x',  'left_hip_y',  'left_knee_y',  'left_ankle_y',
    'right_hip_x', 'right_hip_y', 'right_knee_y', 'right_ankle_y',
]

# Arm joint names
ARM_JOINT_NAMES = [
    'right_shoulder_x', 'right_shoulder_y', 'right_elbow_y',
]


class CPGGait:
    """
    Central Pattern Generator for bipedal walking.

    Phase convention:
      left leg  phase = ω*t
      right leg phase = ω*t + π   (anti-phase)

    Contact mode:
      A foot is in 'stance' (contact) when its vertical clearance is small.
      We model this via the hip/knee reference: when a leg is in swing its
      knee lifts, which raises the foot off the ground.
    """

    def __init__(self,
                 frequency=0.8,       # gait frequency [Hz]
                 hip_amp=0.30,        # hip_y oscillation amplitude [rad]
                 knee_lift=0.45,      # max knee bend during swing [rad]
                 ankle_scale=0.25,    # ankle compensation ratio
                 lateral_amp=0.08,    # hip_x lateral sway amplitude [rad]
                 lean_fwd=0.04):      # static forward lean [rad]
        self.f       = frequency
        self.omega   = 2 * np.pi * frequency
        self.hip_amp = hip_amp
        self.knee_lift = knee_lift
        self.ankle_scale = ankle_scale
        self.lateral_amp = lateral_amp
        self.lean_fwd = lean_fwd

    def get_refs(self, t, com_y=0.0):
        """
        Compute desired leg joint angles at time t.

        Parameters
        ----------
        t     : simulation time [s]
        com_y : lateral CoM position for balance feedback [m]

        Returns
        -------
        q_ref  : (8,) desired joint angles [rad]
        dq_ref : (8,) desired joint velocities [rad/s]
        """
        phase_l = self.omega * t
        phase_r = phase_l + np.pi

        # Hip pitch (forward swing)
        hip_y_l  = self.hip_amp * np.sin(phase_l) + self.lean_fwd
        hip_y_r  = self.hip_amp * np.sin(phase_r) + self.lean_fwd
        dhip_y_l = self.hip_amp * self.omega * np.cos(phase_l)
        dhip_y_r = self.hip_amp * self.omega * np.cos(phase_r)

        # Knee lift (only during swing = positive phase)
        sw_l     = np.clip(np.sin(phase_l), 0, 1)
        sw_r     = np.clip(np.sin(phase_r), 0, 1)
        knee_l   = self.knee_lift * sw_l
        knee_r   = self.knee_lift * sw_r
        dknee_l  = self.knee_lift * self.omega * np.cos(phase_l) * (np.sin(phase_l) > 0)
        dknee_r  = self.knee_lift * self.omega * np.cos(phase_r) * (np.sin(phase_r) > 0)

        # Ankle (partial compensation)
        ankle_l  = -self.ankle_scale * hip_y_l
        ankle_r  = -self.ankle_scale * hip_y_r
        dankle_l = -self.ankle_scale * dhip_y_l
        dankle_r = -self.ankle_scale * dhip_y_r

        # Lateral hip (balance + sway)
        hip_x_l  =  self.lateral_amp * np.cos(phase_l) - 2.0 * com_y
        hip_x_r  = -self.lateral_amp * np.cos(phase_l) - 2.0 * com_y
        dhip_x_l = -self.lateral_amp * self.omega * np.sin(phase_l)
        dhip_x_r =  self.lateral_amp * self.omega * np.sin(phase_l)

        q_ref = np.array([
            hip_x_l, hip_y_l, knee_l, ankle_l,
            hip_x_r, hip_y_r, knee_r, ankle_r,
        ])
        dq_ref = np.array([
            dhip_x_l, dhip_y_l, dknee_l, dankle_l,
            dhip_x_r, dhip_y_r, dknee_r, dankle_r,
        ])
        return q_ref, dq_ref

    def get_contact_mode(self, t):
        """
        Returns (left_contact, right_contact) boolean tuple.
        A foot is in contact when it is in stance phase.
        Stance = positive half of cos (knee not lifted).
        """
        phase_l = self.omega * t
        phase_r = phase_l + np.pi
        left_contact  = np.cos(phase_l) > -0.2    # ~60% of cycle in stance
        right_contact = np.cos(phase_r) > -0.2
        return bool(left_contact), bool(right_contact)

    def mode_key(self, t):
        lc, rc = self.get_contact_mode(t)
        return frozenset(
            (['left'] if lc else []) + (['right'] if rc else [])
        )


class StanceBalance:
    """
    Static PD balance for fixed double-support stance (Scenario A).
    Returns leg joint refs that maintain upright posture.
    """
    def __init__(self):
        self.q_stand = np.array([
            0.0, -0.05, 0.10, -0.05,   # left: hip_x, hip_y, knee, ankle
            0.0, -0.05, 0.10, -0.05,   # right
        ])

    def get_refs(self, t, com_y=0.0):
        q_ref  = self.q_stand.copy()
        # Lateral balance
        q_ref[0] = -1.5 * com_y   # left hip_x
        q_ref[4] = -1.5 * com_y   # right hip_x
        return q_ref, np.zeros(8)

    def get_contact_mode(self, t=0.0):
        return True, True

    def mode_key(self, t=0.0):
        return frozenset(['left', 'right'])
