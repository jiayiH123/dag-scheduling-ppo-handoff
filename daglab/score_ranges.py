"""Explicit smooth coefficient bounds; parameter-compatible with StateActor."""
import torch
from .state_score_torch import StateActor

RANGES = {
    'control': (8., 8., 2., 2.),
    'aw_narrow': (4., 4., 2., 2.),
    'aw_wide': (16., 16., 2., 2.),
    'gap_wide': (8., 8., 4., 2.),
    'wait_wide': (8., 8., 2., 4.),
    'extras_wide': (8., 8., 4., 4.),
    'extras_narrow': (8., 8., 1., 1.),
}


def bounded(raw, bounds):
    r = raw.new_tensor(bounds)
    if len(bounds) != 4 or not bool(torch.isfinite(r).all()) or not bool((r > 0).all()):
        raise ValueError('four finite positive coefficient bounds required')
    return r * torch.tanh(raw / r)


class RangeActor(StateActor):
    """Same weights and one-forward contract; retains raw heads for diagnostics."""
    def __init__(self, bounds=RANGES['control']):
        super().__init__('both')
        self.bounds = tuple(bounds)
        self.base.head.register_forward_hook(self._base_raw)
        self.extra[-1].register_forward_hook(self._extra_raw)

    def _base_raw(self, module, args, output):
        self.raw_aw = output

    def _extra_raw(self, module, args, output):
        self.raw_extra = output

    def forward(self, graph):
        legacy = super().forward(graph)
        self.raw = torch.cat([self.raw_aw, self.raw_extra], -1)
        x = bounded(self.raw, self.bounds)
        if self.bounds == RANGES['control']:
            assert torch.equal(x, legacy), 'default bounds changed existing coefficients'
        return x
