"""Read-only Train streams using the production datasets, samplers and budgets."""
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import RandomSampler

from audit_w2v_dev import ReadOnlyCollator
from audit_w2v_structure import index_bank
from audit_w2v_train import Issues, protocol_rows
from .core import class_weights, sha256
from .data import DataBundle, SeededDataset, SourceTaggedDataset, combine_batches
from .language import language_id
from .sampling import BalancedOrdinarySampler


def recorded_args(config, device, microbatch=None):
    from .train import parser
    values = {a.dest: a.default for a in parser()._actions if a.dest != 'help'}
    values.update(config)
    values.update(device=device, microbatch=microbatch or config.get('microbatch', 4))
    if values['stage'] != 3:
        raise ValueError('Pair-gradient audit requires a recorded Stage3 run')
    if values['local_structure_weight'] or values['consistency_weight']:
        raise ValueError('Use the earlier run with CE + RTC + noisy InfoNCE; extra objectives are not silently omitted')
    if values['coverage_training'] and (not values['language_weighting'] or values['ordinary_sampling'] != 'legacy'):
        raise ValueError('Invalid recorded coverage recipe')
    if values['language_weighting'] and values['ordinary_sampling'] != 'legacy':
        raise ValueError('Language budgets require ordinary legacy traversal')
    if values['amp'] not in ('bf16', 'none') or values['microbatch'] < 1:
        raise ValueError('Invalid recorded precision or microbatch')
    if values.get('no_feature_cache'):
        values['feature_cache'] = None
    return argparse.Namespace(**values)


def metadata_fingerprints(config, source):
    paths = [source/'stage3'/'config.json']
    paths += [Path(config[key]) for key in ('train_protocol', 'rtc_pairs', 'train_noise_manifest')]
    paths += [Path(config['ssl_path'])/key for key in ('config.json', 'preprocessor_config.json')]
    banks = [config['train_noisy_cache']] + config.get('extra_train_noisy_cache', [])
    paths += [Path(bank)/key for bank in banks for key in ('config.json', 'manifest.jsonl')]
    recorded = config.get('data_fingerprints', {})
    result = {}
    for path in paths:
        key = str(path.expanduser().resolve())
        print('Checking Train metadata: '+key, flush=True)
        actual = sha256(path)
        if path != paths[0] and recorded.get(key) != actual:
            raise ValueError('Missing/changed recorded Train fingerprint: '+key)
        result[key] = actual
    pair_meta = Path(config['rtc_pairs']+'.meta.json')
    if pair_meta.is_file():
        result[str(pair_meta.resolve())] = sha256(pair_meta)
    return result


class AuditTrainData:
    """Plan epoch 1 exactly, then inspect sampled steps without committing it.

    No Dev loader, cache publication, worker process, sampler state save or
    augmentation cache generation. Ordinary augmentation is production code
    evaluated in RAM only for batches chosen for the gradient measurement.
    """
    def __init__(self, args):
        from utils.data_utils import build_dataset_from_protocol
        from utils.rtc_data import BalancedPairBatchSampler, RTCPairDataset
        from utils.rtc_pairs import load_pairs
        from rtc_noisy_v2.cache import RotatingNoisyDataset
        from rtc_noisy_v2.sampling import RotatingViewBatchSampler
        self.args = args
        self.language_weighting, self.coverage_training = args.language_weighting, args.coverage_training
        self.language_budgets, self.audio_hashes = {}, {}
        issues = Issues()
        rows, _ = protocol_rows(args.train_protocol, args.train_data_path, issues)
        if issues.counts:
            raise ValueError('Ambiguous Train protocol: '+str(dict(issues.counts)))
        self.protocol = {r['source_id']: r for r in rows}
        banks = [index_bank(p, 'train', args.train_protocol, args.train_data_path, self.protocol)
                 for p in [args.train_noisy_cache] + args.extra_train_noisy_cache]
        if any(c['noise']['manifest_sha256'] != sha256(args.train_noise_manifest) for _, c in banks):
            raise ValueError('Cache noise manifest differs from ordinary Train noise')
        mapping = lambda bank: {(r['source'], r['band']): (r['label'], r['source_sha256']) for r in bank}
        if any(mapping(rows) != mapping(banks[0][0]) for rows, _ in banks[1:]):
            raise ValueError('Train cache banks differ in source/label/hash')
        self.pairs = load_pairs(args.rtc_pairs, args.train_protocol, args.train_data_path)
        if any(language_id(p['offline']) != language_id(p['online']) for p in self.pairs):
            raise ValueError('RTC pair language mismatch')
        ordinary, self.ids, labels = build_dataset_from_protocol(args.train_protocol, args.train_data_path,
                                                                 mode='train', args=args, algo=args.algo)
        if ordinary.env_noise is None:
            raise ValueError('Recorded ordinary environmental noise is not enabled')
        self.weights, self.counts = class_weights(self.ids, labels)
        self.steps = math.ceil(len(ordinary)/24)
        seeded = SeededDataset(ordinary, args.seed+50)
        seeded.epoch[0] = 1
        self.ordinary = self.tag(seeded, self.ids, [labels[s] for s in self.ids], 'ordinary')
        if args.ordinary_sampling == 'balanced':
            sampler = BalancedOrdinarySampler([labels[i] for i in self.ids], 24, self.steps, args.seed+7)
            sampler.set_epoch(1)
            self.ordinary_plan = list(sampler)
            self.weights = torch.ones(2)
        else:
            generator = torch.Generator().manual_seed(args.seed+100)
            order = list(RandomSampler(seeded, generator=generator))
            self.ordinary_plan = [order[i:i+24] for i in range(0, len(order), 24)]
        real_sampler = BalancedPairBatchSampler(self.pairs, 4, self.steps, args.seed+1)
        real_sampler.set_epoch(1)
        self.rtc_plan = list(real_sampler)
        self.rtc = self.tag(RTCPairDataset(self.pairs, args.train_data_path),
                            [p['offline'] for p in self.pairs], [p['label'] for p in self.pairs], 'rtc_pair')
        self.noisy_inner = RotatingNoisyDataset([r for r, _ in banks], args.train_data_path)
        sources = self.noisy_inner.sources
        source_sampler = BalancedPairBatchSampler(sources, 4, self.steps, args.seed+2)
        if args.coverage_training:
            from .coverage import CoverageViewBatchSampler
            rotation = CoverageViewBatchSampler(source_sampler, self.noisy_inner, args.seed+3,
                                                 args.en_real_budget, args.en_fake_budget)
        else:
            rotation = RotatingViewBatchSampler(source_sampler, sources, args.seed+3, len(banks), args.noisy_bank_policy)
        if args.noisy_extra_fraction:
            rotation.configure_mixture(args.noisy_extra_fraction, round(self.steps*args.noisy_mix_warmup_epochs))
        rotation.set_epoch(1)
        self.noisy_plan, self.rotation_summary = list(rotation), rotation.plan_summary()
        self.noisy = self.tag(self.noisy_inner, [p['offline'] for p in sources], [p['label'] for p in sources], 'noisy_pair')
        self.collators = {k: ReadOnlyCollator(args.ssl_path, k, None if k == 'ordinary' else args.feature_cache,
                                             source_metadata=True) for k in ('ordinary', 'pair', 'noisy_pair')}

    def tag(self, dataset, ids, labels, branch):
        if self.language_weighting:
            return DataBundle.tag(self, dataset, ids, labels, branch)
        return SourceTaggedDataset(dataset, ids, labels)

    def verify_audio(self, path, expected=None):
        path = str(Path(path).resolve())
        if path not in self.audio_hashes:
            self.audio_hashes[path] = sha256(path)
        if expected is not None and self.audio_hashes[path] != expected:
            raise ValueError('Selected audio differs from cached source hash: '+path)

    def metadata(self, step):
        result = []
        for index in self.rtc_plan[step]:
            p = self.pairs[index]
            for side in ('offline', 'online'):
                self.verify_audio(Path(self.args.train_data_path)/p[side])
            result.append(dict(branch='rtc', source_id=p['offline'], processed_id=p['online'], label=p['label'],
                               language=['en', 'zh'][language_id(p['offline'])], bank=-1, family='real_rtc', band=-1))
        for index, bank, band in self.noisy_plan[step]:
            sid = self.noisy_inner.sources[index]['offline']
            row = self.noisy_inner.banks[bank][sid][band]
            self.verify_audio(Path(self.args.train_data_path)/sid, row['source_sha256'])
            self.verify_audio(row['audio'])
            result.append(dict(branch='noisy', source_id=sid, processed_id=row['audio'], label=row['label'],
                               language=['en', 'zh'][language_id(sid)], bank=bank, band=band,
                               family=row.get('processing', {}).get('family', 'ffmpeg'), snr_db=row['snr_db']))
        return result

    def pair_batches(self, step):
        self.metadata(step)
        return [self.collators['pair']([self.rtc[t] for t in self.rtc_plan[step]]),
                self.collators['noisy_pair']([self.noisy[t] for t in self.noisy_plan[step]])]

    def logical_batch(self, step):
        for ticket in self.ordinary_plan[step]:
            index = ticket[0] if isinstance(ticket, tuple) else ticket
            self.verify_audio(self.protocol[self.ids[index]]['audio_path'])
        ordinary = self.collators['ordinary']([self.ordinary[t] for t in self.ordinary_plan[step]])
        return combine_batches([ordinary, *self.pair_batches(step)])
