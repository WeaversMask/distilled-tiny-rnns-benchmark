"""Shared teacher and independent, vectorized student GRUs.

The student dimension is a batch of parameter sets, not shared network weights.
The gates follow torch.nn.GRU (including reset *after* the hidden affine map).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TeacherGRU(nn.Module):
    """One shared GRU; inputs [people, trials, features], logits [people, trials]."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.readout(hidden).squeeze(-1)


class BatchedStudentGRU(nn.Module):
    """A separate GRU and readout for every person, evaluated together."""

    def __init__(self, n_subjects: int, input_size: int, hidden_size: int):
        super().__init__()
        if min(n_subjects, input_size, hidden_size) < 1:
            raise ValueError("model dimensions must be positive")
        self.n_subjects = n_subjects
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(n_subjects, 3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(n_subjects, 3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(n_subjects, 3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(n_subjects, 3 * hidden_size))
        self.readout_weight = nn.Parameter(torch.empty(n_subjects, hidden_size))
        self.readout_bias = nn.Parameter(torch.empty(n_subjects))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != self.n_subjects or x.shape[2] != self.input_size:
            raise ValueError("x must have shape [n_subjects, trials, input_size]")
        if x.shape[1] < 1:
            raise ValueError("at least one trial is required")
        # One large input multiplication avoids one small launch per time step.
        input_gates = torch.bmm(x, self.weight_ih.transpose(1, 2)) + self.bias_ih[:, None, :]
        hidden = x.new_zeros((self.n_subjects, self.hidden_size))
        outputs = []
        for trial in range(x.shape[1]):
            hidden_gates = torch.bmm(self.weight_hh, hidden.unsqueeze(-1)).squeeze(-1) + self.bias_hh
            i_reset, i_update, i_new = input_gates[:, trial].chunk(3, dim=-1)
            h_reset, h_update, h_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(i_reset + h_reset)
            update = torch.sigmoid(i_update + h_update)
            new = torch.tanh(i_new + reset * h_new)
            hidden = (1.0 - update) * new + update * hidden
            outputs.append((hidden * self.readout_weight).sum(-1) + self.readout_bias)
        return torch.stack(outputs, dim=1)


def copy_student_to_reference(model: BatchedStudentGRU, subject: int) -> TeacherGRU:
    """Create an ordinary torch.nn.GRU with exactly one student's parameters."""
    if not 0 <= subject < model.n_subjects:
        raise IndexError(subject)
    reference = TeacherGRU(model.input_size, model.hidden_size).to(
        device=model.weight_ih.device, dtype=model.weight_ih.dtype
    )
    with torch.no_grad():
        reference.gru.weight_ih_l0.copy_(model.weight_ih[subject])
        reference.gru.weight_hh_l0.copy_(model.weight_hh[subject])
        reference.gru.bias_ih_l0.copy_(model.bias_ih[subject])
        reference.gru.bias_hh_l0.copy_(model.bias_hh[subject])
        reference.readout.weight.copy_(model.readout_weight[subject].unsqueeze(0))
        reference.readout.bias.copy_(model.readout_bias[subject].reshape(1))
    return reference
