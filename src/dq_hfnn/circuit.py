"""Differentiable two-qubit membership circuits implemented with PyTorch."""

import math

import torch
import torch.nn as nn


class DualQubitMembershipCircuit(nn.Module):
    """Author-style asymmetric two-parameter dual-qubit circuit."""

    VALID_VARIANTS = {"no_ent", "weak_ent", "strong_ent"}
    VALID_MEASUREMENT_BASES = {"x", "z"}

    def __init__(self, variant="strong_ent", measurement_basis="z"):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unsupported circuit variant: {variant}")
        if measurement_basis not in self.VALID_MEASUREMENT_BASES:
            raise ValueError(f"Unsupported measurement basis: {measurement_basis}")
        self.variant = variant
        self.measurement_basis = measurement_basis
        self.theta = nn.Parameter(torch.empty(2))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)

    @staticmethod
    def _ry_q0(state, angle):
        c, s = torch.cos(angle / 2), torch.sin(angle / 2)
        out = state.clone()
        out[:, 0], out[:, 2] = c * state[:, 0] - s * state[:, 2], s * state[:, 0] + c * state[:, 2]
        out[:, 1], out[:, 3] = c * state[:, 1] - s * state[:, 3], s * state[:, 1] + c * state[:, 3]
        return out

    @staticmethod
    def _ry_q1(state, angle):
        c, s = torch.cos(angle / 2), torch.sin(angle / 2)
        out = state.clone()
        out[:, 0], out[:, 1] = c * state[:, 0] - s * state[:, 1], s * state[:, 0] + c * state[:, 1]
        out[:, 2], out[:, 3] = c * state[:, 2] - s * state[:, 3], s * state[:, 2] + c * state[:, 3]
        return out

    @staticmethod
    def _rz_q0(state, angle):
        zero = torch.zeros_like(angle)
        low = torch.exp(torch.complex(zero, -angle / 2))
        high = torch.exp(torch.complex(zero, angle / 2))
        return state * torch.stack((low, low, high, high), dim=1)

    @staticmethod
    def _rz_q1(state, angle):
        zero = torch.zeros_like(angle)
        low = torch.exp(torch.complex(zero, -angle / 2))
        high = torch.exp(torch.complex(zero, angle / 2))
        return state * torch.stack((low, high, low, high), dim=1)

    @staticmethod
    def _cnot(state):
        return state[:, (0, 1, 3, 2)]

    @staticmethod
    def _cnot_reverse(state):
        """CNOT with q1 as control and q0 as target."""
        return state[:, (0, 3, 2, 1)]

    @staticmethod
    def _measure_memberships(state, basis):
        if basis == "z":
            probabilities = state.real.square() + state.imag.square()
            expectation0 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
            expectation1 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
        elif basis == "x":
            expectation0 = 2.0 * (
                (state[:, 0].conj() * state[:, 2]).real
                + (state[:, 1].conj() * state[:, 3]).real
            )
            expectation1 = 2.0 * (
                (state[:, 0].conj() * state[:, 1]).real
                + (state[:, 2].conj() * state[:, 3]).real
            )
        else:
            raise ValueError(f"Unsupported measurement basis: {basis}")
        return torch.stack(((expectation0 + 1) / 2, (expectation1 + 1) / 2), dim=1)

    def forward(self, pairs):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"Expected [batch, 2], got {tuple(pairs.shape)}")
        batch = pairs.shape[0]
        state = torch.zeros(batch, 4, device=pairs.device, dtype=torch.complex64)
        state[:, 0] = 1
        state = self._ry_q0(state, pairs[:, 0])
        state = self._ry_q1(state, pairs[:, 1])
        state = self._rz_q0(state, self.theta[0].expand(batch))
        if self.variant != "no_ent":
            state = self._cnot(state)
        state = self._ry_q0(state, self.theta[1].expand(batch))
        if self.variant == "strong_ent":
            state = self._cnot(state)
        return self._measure_memberships(state, self.measurement_basis)


class ReuploadDualQubitMembershipCircuit(DualQubitMembershipCircuit):
    """V215-style signed/energy re-uploading circuit with two CNOT blocks."""

    def __init__(self):
        nn.Module.__init__(self)
        self.theta = nn.Parameter(torch.empty(2, 4))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)

    def _trainable_block(self, state, parameters):
        state = self._rz_q0(state, parameters[0].expand(state.shape[0]))
        state = self._ry_q0(state, parameters[1].expand(state.shape[0]))
        state = self._rz_q1(state, parameters[2].expand(state.shape[0]))
        state = self._ry_q1(state, parameters[3].expand(state.shape[0]))
        return self._cnot(state)

    def forward(self, angles):
        if angles.ndim != 2 or angles.shape[1] != 4:
            raise ValueError(f"Expected [batch, 4], got {tuple(angles.shape)}")
        batch = angles.shape[0]
        state = torch.zeros(batch, 4, device=angles.device, dtype=torch.complex64)
        state[:, 0] = 1
        state = self._ry_q0(state, angles[:, 0])
        state = self._ry_q1(state, angles[:, 1])
        state = self._trainable_block(state, self.theta[0])
        state = self._rz_q0(state, angles[:, 2])
        state = self._rz_q1(state, angles[:, 3])
        state = self._trainable_block(state, self.theta[1])
        probabilities = state.real.square() + state.imag.square()
        z0 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
        z1 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
        return torch.stack(((z0 + 1) / 2, (z1 + 1) / 2), dim=1)
