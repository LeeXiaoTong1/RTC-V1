"""Full-source coverage with four equally budgeted language/class groups."""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from w2v_aasist.data import read_protocol, read_wave
from w2v_aasist.runtime import sha256
from w2v_v3.data import stratified_order, loader as validation_loader
from w2v_v32.batching import PreparedCollator, PreparedBatch
from w2v_v32.model import microbatches
from w2v_v33.data import canonical_sources, resolve_pair_manifest
from w2v_v33.transport import PackedPreparedBatch, paired_worker_init
from .cache import prepare_base, prepare_dev, prepare_epoch, EpochCache, _root


class SourcePlan:
    def __init__(self, sources, source_batch=16, seed=1234):
        if type(source_batch) is not int or source_batch < 1: raise ValueError('Invalid source batch')
        self.records = sources
        self.source_count, self.source_batch, self.seed = len(sources), source_batch, seed
        self.steps = math.ceil(self.source_count/source_batch)
        self.group_counts = dict(Counter(f'{s["language"]}/{s["label"]}' for s in sources))
        if set(self.group_counts) != {'en/0', 'en/1', 'zh/0', 'zh/1'}:
            raise ValueError('V3.5 requires official Train sources in all four language/class groups')
        self.weights = {group:self.source_count/(4*count) for group,count in self.group_counts.items()}
        self.full = True

    def batches(self, epoch):
        order = stratified_order(self.records, range(self.source_count), self.seed+1009*epoch)
        return [order[i:i+self.source_batch] for i in range(0, len(order), self.source_batch)]

    def tickets(self, epoch, start=0, stop=None):
        yield from self.batches(epoch)[start:stop]

    def coverage(self):
        missing = sum(s['missing_online'] for s in self.records)
        return dict(sources_per_epoch=self.source_count, source_batch=self.source_batch, steps=self.steps,
            source_groups=self.group_counts, group_weights=self.weights, missing_online=missing,
            condition_rows_per_epoch={'offline':self.source_count,'online':self.source_count-missing,
                                      'noisy_a':self.source_count,'noisy_b':self.source_count},
            note='Each canonical source once; four groups have equal aggregate classification coefficient budgets; full wave only.')


def _full_example(row, wave, source_id, condition):
    if not len(wave) or not np.isfinite(wave).all(): raise ValueError('Invalid full waveform')
    # The feature extractor needs at least two output tokens. A truly short file
    # is repeated only to the previous system's 0.4s minimum, never cropped.
    original_samples = len(wave)
    if len(wave) < 6400: wave = np.tile(wave, math.ceil(6400/len(wave)))[:6400]
    return dict(row, source_id=source_id, condition=condition, wave=np.ascontiguousarray(wave, dtype=np.float32),
        source_group=source_id, view='full', view_weight=1., full_length=True,
        source_samples=original_samples, source_audio_seconds=original_samples/16000,
        audio_seconds=len(wave)/16000, crop_start_sample=0, crop_samples=original_samples,
        augmentation='none', noisy=condition.startswith('noisy_'), composition='v35_full_'+condition)


class FullSourceDataset(Dataset):
    def __init__(self, records, cfg, epoch):
        self.records, self.cfg, self.epoch = records, cfg, epoch
        self.cache = None

    def __len__(self): return len(self.records)

    def __getitem__(self, ticket):
        source = self.records[ticket]
        if self.cache is None: self.cache = EpochCache(self.cfg, self.epoch)
        offline = source['conditions']['offline']
        wave = read_wave(offline['audio'])
        output = [_full_example(offline, wave, source['source_id'], 'offline')]
        if 'online' in source['conditions']:
            online = source['conditions']['online']
            identity = self.cache.inventory[online['id']]
            stat = Path(online['audio']).stat()
            if stat.st_size != identity['size'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise ValueError('Official Online source changed during V3.5 run')
            output.append(_full_example(online, read_wave(online['audio']), source['source_id'], 'online'))
        for noisy, metadata, path in self.cache.get(offline, wave=wave):
            a = metadata['assignment']
            row = dict(offline, audio=path, band=a['band'], processing_family=a['family'],
                       processing_mode=a['mode'], snr_db=metadata['snr_db'])
            output.append(_full_example(row, noisy, source['source_id'], a['condition']))
        return output

    def close(self):
        if self.cache is not None: self.cache.close()

    def __del__(self): self.close()


class FullCollator(PreparedCollator):
    def __init__(self, *args, source_chunk=4, **kwargs):
        super().__init__(*args, **kwargs)
        if type(source_chunk) is not int or source_chunk < 1: raise ValueError('Invalid source chunk')
        self.source_chunk = source_chunk

    def __call__(self, rows):
        if self.extractor is None:
            from transformers import AutoFeatureExtractor
            self.extractor=AutoFeatureExtractor.from_pretrained(self.ssl_path,local_files_only=True)
            if self.extractor.sampling_rate!=16000: raise ValueError('Expected official 16kHz extractor')
        expanded=[view for source in rows for view in source]
        result=[None]*len(expanded)
        ordered=sorted(range(len(expanded)),key=lambda i:len(expanded[i]['wave']))
        for start in range(0,len(ordered),self.feature_batch):
            indices=ordered[start:start+self.feature_batch]
            features=self.extractor([expanded[i]['wave'] for i in indices],sampling_rate=16000,
                padding=True,pad_to_multiple_of=2,return_attention_mask=True,return_tensors='pt')
            for j,index in enumerate(indices):
                f=features['input_features'][j:j+1]
                m=features['attention_mask'][j:j+1].long();length=int(m.sum())
                if length<2 or f.shape[-1]!=160 or not bool(m[:,:length].bool().all()):
                    raise ValueError('Invalid official feature extractor output')
                f,m=f[:,:length].contiguous().float(),m[:,:length].contiguous()
                if not bool(torch.isfinite(f).all()):raise FloatingPointError('Nonfinite extracted features')
                result[index]={**{k:v for k,v in expanded[index].items() if k!='wave'},'features':f,'mask':m}
        # Physical batches stay inside the graph's source chunk. Workers pad
        # these buffers once; the training process can consume the same pinned
        # views without rebuilding pageable CPU padding for every backward.
        groups = {}
        for index, example in enumerate(result): groups.setdefault(example['source_id'], []).append(index)
        sources = list(groups)
        batches = []
        for start in range(0, len(sources), self.source_chunk):
            indices = [index for source in sources[start:start+self.source_chunk] for index in groups[source]]
            for local, features, mask in microbatches([result[i] for i in indices], self.size, self.frame_budget):
                batches.append(([indices[i] for i in local], features, mask))
        # Transport requires a third packed boolean storage. All tokens are
        # active here; no audibility or consistency objective consumes it.
        for example in result:
            example['audibility_mask'] = torch.ones(example['features'].shape[1], dtype=torch.bool)
        return PackedPreparedBatch(PreparedBatch(result,batches,self.size,self.frame_budget))


def loader(records, cfg, *, training=False, epoch=0, batches=None):
    if not training: return validation_loader(records, cfg, training=False, epoch=epoch, batches=batches)
    prepare_epoch(cfg, epoch)
    records = records.records if isinstance(records, SourcePlan) else records
    kwargs = dict(dataset=FullSourceDataset(records,cfg,epoch),
        collate_fn=FullCollator(cfg['ssl_path'],cfg.get('microbatch',4),cfg.get('frame_budget',2400),
                               source_chunk=cfg.get('source_chunk',4)),
        num_workers=cfg.get('workers',0),pin_memory=str(cfg.get('device','cpu')).startswith('cuda'),
        generator=torch.Generator().manual_seed(cfg['seed']+epoch))
    if kwargs['num_workers']:
        kwargs.update(multiprocessing_context='spawn',worker_init_fn=paired_worker_init,
                      prefetch_factor=1,persistent_workers=False)
    if batches is None: kwargs.update(batch_size=cfg.get('source_batch',16),shuffle=False)
    else: kwargs['batch_sampler'] = batches
    print(f'V35_LOADER full_views_only=True lazy_epoch_cache=True workers={kwargs["num_workers"]} prefetch=1',flush=True)
    return DataLoader(**kwargs)


def build_data(cfg):
    if Path(cfg['train_data_path']).resolve() == Path(cfg['dev_data_path']).resolve():
        raise ValueError('Train and Dev roots must differ')
    train = read_protocol(cfg['train_protocol'],cfg['train_data_path'])
    pairs = resolve_pair_manifest(cfg)
    sources = canonical_sources(train,pairs)
    plan = SourcePlan(sources,cfg.get('source_batch',16),cfg['seed'])
    dev = read_protocol(cfg['dev_protocol'],cfg['dev_data_path'])
    offline_dev = [r for r in dev if r['domain']=='offline']
    online_dev = [r for r in dev if r['domain']=='online']
    if not offline_dev or not online_dev: raise ValueError('Fixed Dev needs Offline sources and official Online')
    prepare_base(cfg,train,dev)
    validation = {'clean':online_dev,**prepare_dev(cfg,offline_dev)}
    files = [cfg['train_protocol'],cfg['dev_protocol'],pairs,
             cfg['train_noise_manifest'],cfg['dev_noise_manifest'],Path(cfg['ssl_path'])/'preprocessor_config.json']
    for role in ('train','dev'):
        files.extend(_root(cfg,role)/name for name in ('owner.json','recipe.json','sources.json'))
    files.extend(_root(cfg,'dev')/'epoch_000'/name for name in ('manifest.json','complete.json'))
    fingerprints = {str(Path(path).resolve()):sha256(path) for path in files}
    # Do not add mutable generation manifests: epoch/seed/recipe in last.pt
    # reconstruct a partial lazy cache deterministically without changing data.
    counts = torch.bincount(torch.tensor([s['label'] for s in sources]),minlength=2)
    return plan,validation,plan.weights,counts,fingerprints


prepare_data = build_data
