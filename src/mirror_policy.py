# -----------------------------------------------------------------------------
# mirror_policy.py
#
# Self‑contained OOP implementation of an online Mirror‑Descent scheduler
# that chooses the (channel, encoding‑profile) action at each frame,
# using full‑information gradient feedback.
#
# The solver can be plugged straight into the earlier `StreamingEnv`
# via the profile_policy and a channel‑choice hook.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class Action:
    """One decision option: (channel c, profile r)."""
    channel: int   # 0 = local processing
    profile: str   # e.g. "low", "medium", "high"


class OnlineMirrorDescent:
    """
    Full‑information Mirror Descent over the probability simplex.

    Attributes
    ----------
    actions : List[Action]
        Catalogue of all possible (channel, profile) pairs.
    wL, wE, wQ : float
        Objective weights for latency, energy, and PSNR.
    eta0 : float
        Initial learning‑rate numerator (η_t = η0 / sqrt(t)).
    """

    def __init__(self,
                 actions: List[Action],
                 wL: float,
                 wE: float,
                 wQ: float,
                 eta0: float = 0.3,
                 rng: np.random.Generator | None = None):
        self.actions = actions
        self.A = len(actions)
        self.wL, self.wE, self.wQ = wL, wE, wQ
        self.eta0 = eta0
        self.rng = np.random.default_rng() if rng is None else rng
        # Initial uniform policy
        self.pi: np.ndarray = np.full(self.A, 1 / self.A)
        self.timestep = 0

    # ------------------------------------------------------------------
    # Helper: Euclidean projection onto simplex
    # ------------------------------------------------------------------
    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(len(u)) + 1
        rho = ind[u - cssv / ind > 0][-1]
        theta = cssv[rho - 1] / rho
        return np.maximum(v - theta, 0)

    # ------------------------------------------------------------------
    # API called by the simulator
    # ------------------------------------------------------------------
    def choose_action(self) -> Tuple[int, Action]:
        """Sample an action index according to current π."""
        idx = self.rng.choice(self.A, p=self.pi)
        return idx, self.actions[idx]

    def update(self,
               cost_vector: np.ndarray):
        """
        Perform one mirror‑descent step.

        Parameters
        ----------
        cost_vector : np.ndarray
            Array of shape (A,) with scalar cost for *each*
            action (computed from current frame's measurements).
        """
        self.timestep += 1
        eta = self.eta0 / np.sqrt(self.timestep)
        self.pi = self._project_simplex(self.pi - eta * cost_vector)

    # ------------------------------------------------------------------
    # Convenience: current policy as (C+1)×R heat‑map matrix
    # ------------------------------------------------------------------
    def policy_matrix(self,
                      channels: List[int],
                      profiles: List[str]) -> np.ndarray:
        mat = np.zeros((len(channels), len(profiles)))
        idx_map = {(a.channel, a.profile): i
                   for i, a in enumerate(self.actions)}
        for ci, c in enumerate(channels):
            for ri, r in enumerate(profiles):
                mat[ci, ri] = self.pi[idx_map[(c, r)]]
        return mat


# -----------------------------------------------------------------------------
# Example latency/energy/quality function placeholders
# (replace with your real model; make sure they're vectorisable)
# -----------------------------------------------------------------------------

def latency(channel: int,
            profile_comp: float,
            size_bits: int,
            uplink_rate: float | None,
            downlink_rate: float | None,
            rho: int = 200,
            f_vr: float = 3e9,
            f_edge: float = 15e9) -> float:
    """Return *expected* latency in seconds."""
    S = profile_comp * size_bits
    if channel == 0:  # local
        return rho * S / f_vr
    return (S / uplink_rate) + (rho * S / f_edge) + (S / downlink_rate)


def energy(channel: int,
           profile_comp: float,
           size_bits: int,
           uplink_rate: float | None,
           downlink_rate: float | None,
           kappa: float = 5e-12,
           eps_tx: float = 1e-9,
           eps_rx: float = 1e-9,
           rho: int = 200,
           f_vr: float = 3e9) -> float:
    """Return *expected* energy in Joules."""
    S = profile_comp * size_bits
    if channel == 0:
        return kappa * f_vr**2 * rho * S
    return eps_tx * S / uplink_rate + eps_rx * S / downlink_rate


# -----------------------------------------------------------------------------
# Minimal usage demonstration with synthetic numbers
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define action set: 0=local, 1 & 2 = wireless
    profiles = {"low": 0.3, "med": 0.5, "high": 0.7}
    actions = [Action(c, p) for c in (0, 1, 2) for p in profiles]

    solver = OnlineMirrorDescent(actions, wL=1.0, wE=1e9, wQ=0.6)

    # mock throughput traces
    upl = {1: 40e6, 2: 20e6}
    dwn = {1: 50e6, 2: 25e6}
    size_bits = 2_000_000

    for step in range(500):
        # Build full cost vector under current conditions
        costs = np.zeros(len(actions))
        for i, act in enumerate(actions):
            comp = profiles[act.profile]
            if act.channel == 0:
                L = latency(0, comp, size_bits, None, None)
                E = energy(0, comp, size_bits, None, None)
            else:
                L = latency(act.channel, comp, size_bits,
                            upl[act.channel], dwn[act.channel])
                E = energy(act.channel, comp, size_bits,
                           upl[act.channel], dwn[act.channel])
            Q = 30 + 10 * comp             # toy PSNR
            costs[i] = solver.wL * L + solver.wE * E - solver.wQ * Q

        solver.update(costs)

    # Inspect learned policy
    pm = solver.policy_matrix([0, 1, 2], list(profiles))
    print("Final π:")
    print(np.round(pm, 3))
