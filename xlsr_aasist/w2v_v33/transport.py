"""Bounded CPU tensor transport; acoustic values and physical batches stay exact."""
import os
import torch
from w2v_v32.batching import PreparedBatch
from w2v_aasist.data import worker_init


def paired_worker_init(index):
    worker_init(index)
    # Set this inside each spawned producer: parent globals are not inherited.
    # Linux file_descriptor transport uses SCM_RIGHTS/ancdata per storage.
    if os.name == 'posix':
        if 'file_system' not in torch.multiprocessing.get_all_sharing_strategies():
            raise RuntimeError('V3.3 Linux loader requires file_system tensor sharing')
        torch.multiprocessing.set_sharing_strategy('file_system')


class PackedPreparedBatch(PreparedBatch):
    """Only three storages cross IPC, independent of the number of paired views.

    The old batch retained both per-view features and their padded copies. All
    aliases now refer to the same packed physical buffers, before and after pin.
    No lossy compression, re-extraction, truncation or regrouping is performed.
    """
    def __init__(self, prepared):
        self.size, self.frame_budget = prepared.size, prepared.frame_budget
        self.examples = [{k:v for k,v in ex.items()
                          if k not in ('features','mask','audibility_mask')} for ex in prepared]
        self.lengths = [ex['features'].shape[1] for ex in prepared]
        self.layout = [(list(ids), tuple(f.shape), tuple(m.shape)) for ids,f,m in prepared.batches]
        if not self.layout:
            raise ValueError('Cannot transport an empty source batch')
        self.feature_storage = torch.cat([f.reshape(-1) for _,f,_ in prepared.batches])
        self.mask_storage = torch.cat([m.reshape(-1) for _,_,m in prepared.batches])
        self.audibility_storage = torch.cat([ex['audibility_mask'].reshape(-1) for ex in prepared])
        if (self.feature_storage.device.type != 'cpu' or self.mask_storage.device.type != 'cpu'
                or self.audibility_storage.device.type != 'cpu'):
            raise ValueError('Workers must only return CPU tensors')
        self._bind_views()

    def _bind_views(self):
        self.batches = []
        feature_offset = mask_offset = 0
        for ids, fshape, mshape in self.layout:
            fn = fshape[0]*fshape[1]*fshape[2]; mn = mshape[0]*mshape[1]
            features = self.feature_storage[feature_offset:feature_offset+fn].view(fshape)
            mask = self.mask_storage[mask_offset:mask_offset+mn].view(mshape)
            self.batches.append((ids, features, mask))
            for row, index in enumerate(ids):
                n = self.lengths[index]
                self.examples[index]['features'] = features[row:row+1,:n]
                self.examples[index]['mask'] = mask[row:row+1,:n]
            feature_offset += fn; mask_offset += mn
        offset = 0
        for ex, n in zip(self.examples, self.lengths):
            ex['audibility_mask'] = self.audibility_storage[offset:offset+n]
            offset += n

    def pin_memory(self):
        self.feature_storage = self.feature_storage.pin_memory()
        self.mask_storage = self.mask_storage.pin_memory()
        # Tiny auxiliary masks stay on CPU; the pair objective consumes them.
        self._bind_views()
        return self

    def __getstate__(self):
        # Rebuild views in the consumer instead of invoking a reducer for every
        # tensor alias; pickle sends exactly these three backing storages.
        state = {k:v for k,v in self.__dict__.items() if k != 'batches'}
        state['examples'] = [{k:v for k,v in ex.items()
                              if k not in ('features','mask','audibility_mask')} for ex in self.examples]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._bind_views()
