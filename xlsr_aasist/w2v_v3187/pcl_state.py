"""Persisted, Train-only fixed scales for classification-space phone matching."""
import math
import torch
from torch import nn


class PCLScale(nn.Module):
    """Buffer-only source-visit balanced moments; frozen before joint PCL.

    Each observe call owns equal mass, regardless of its phone count. Features
    are detached and no reference forward or learned projection is involved.
    Counts denote visits, not a claim of that many independent recordings.
    """
    def __init__(self, dim=128, floor_fraction=.1, min_floor=1e-4):
        super().__init__()
        if dim < 1 or not 0 < floor_fraction <= 1 or not math.isfinite(min_floor) or min_floor <= 0:
            raise ValueError('Invalid fixed PCL scale settings')
        self.dim, self.floor_fraction, self.min_floor = dim, floor_fraction, min_floor
        self._frozen_cache = False
        self.register_buffer('mean_sum', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('second_sum', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('source_visits', torch.zeros((), dtype=torch.long))
        self.register_buffer('observed_phones', torch.zeros((), dtype=torch.long))
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('scale', torch.ones(dim, dtype=torch.float32))
        self.register_buffer('frozen', torch.zeros((), dtype=torch.bool))

    @property
    def is_frozen(self):
        return self._frozen_cache

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.sync_state()

    def sync_state(self):
        """Call after direct buffer-copy checkpoint loaders as well as load_state_dict."""
        # One synchronization on checkpoint restore, never one per source.
        self._frozen_cache = bool(self.frozen.item())
        if self._frozen_cache and not bool(torch.isfinite(self.scale).all() & (self.scale > 0).all()):
            raise ValueError('Checkpoint contains invalid frozen PCL scales')

    @torch.no_grad()
    def observe(self, phone_features, source_id, *, split='train'):
        self.observe_batch([(phone_features, source_id, split)])

    @torch.no_grad()
    def observe_batch(self, records):
        """One bounded moment update for a physical batch of source visits."""
        if self.is_frozen:
            raise RuntimeError('Frozen PCL scales cannot observe new features')
        values = []
        for features, source_id, split in records:
            if split != 'train' or source_id is None or not str(source_id):
                raise ValueError('PCL scale calibration requires identified Train sources')
            if features.ndim != 2 or features.shape[1] != self.dim:
                raise ValueError('Invalid PCL calibration feature geometry')
            if len(features):
                values.append(features.detach().to(device=self.mean_sum.device, dtype=torch.float64))
        if not values:
            return
        means = torch.stack([value.mean(0) for value in values]).sum(0)
        seconds = torch.stack([value.square().mean(0) for value in values]).sum(0)
        # CUDA moments are validated when frozen, avoiding a CPU wait for
        # every small source in each training update.
        if means.device.type == 'cpu' and not bool(torch.isfinite(means).all() & torch.isfinite(seconds).all()):
            raise ValueError('Nonfinite PCL scale calibration features')
        self.mean_sum.add_(means)
        self.second_sum.add_(seconds)
        self.source_visits.add_(len(values))
        self.observed_phones.add_(sum(len(value) for value in values))

    @torch.no_grad()
    def freeze(self):
        if self.is_frozen:
            return self.describe()
        if int(self.source_visits) < 2 or int(self.observed_phones) < 2:
            raise RuntimeError('PCL scale calibration needs at least two valid Train source visits')
        if not bool(torch.isfinite(self.mean_sum).all() & torch.isfinite(self.second_sum).all()):
            raise RuntimeError('Nonfinite PCL scale calibration moments')
        mean = self.mean_sum / self.source_visits
        deviation = (self.second_sum / self.source_visits - mean.square()).clamp_min(0).sqrt()
        positive = deviation[deviation > self.min_floor]
        if not len(positive):
            raise RuntimeError('PCL calibration features have no usable variance')
        floor = max(self.min_floor, self.floor_fraction * float(positive.median()))
        self.mean.copy_(mean.float())
        self.scale.copy_(deviation.clamp_min(floor).float())
        self.frozen.fill_(True)
        self._frozen_cache = True
        return self.describe()

    def describe(self):
        return dict(policy='train_phone_fixed_channel_std_v1', frozen=self.is_frozen,
                    source_visits=int(self.source_visits), observed_phones=int(self.observed_phones),
                    floor_fraction=self.floor_fraction, minimum_floor=self.min_floor,
                    scale_min=float(self.scale.min()), scale_max=float(self.scale.max()))
