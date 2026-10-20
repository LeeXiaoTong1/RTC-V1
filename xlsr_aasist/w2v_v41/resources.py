"""Header-only bounds for full w2v-BERT recordings and original-pair PCL."""
from pathlib import Path
from .features import frame_count


def _samples(row):
    value = row.get('output_samples')
    if value is not None:
        if type(value) is not int or value < 1:
            raise ValueError('Invalid output_samples for '+str(row.get('id')))
        return value, 'saved_16khz_samples'
    import soundfile as sf
    info = sf.info(Path(row['audio']).expanduser())
    if info.frames < 1 or info.samplerate < 1 or info.channels < 1:
        raise ValueError('Invalid audio header: '+str(row.get('id')))
    return (int(info.frames)*16000+int(info.samplerate)-1)//int(info.samplerate), 'audio_header'


def _check_units(units, cfg, scope):
    limits = dict(whole_max_frames=int(cfg.get('whole_max_frames', 3000)), frame_budget=int(cfg.get('frame_budget', 9000)), microbatch=int(cfg.get('microbatch', 12)))
    if min(limits.values()) < 1:
        raise ValueError('Resource limits must be positive')
    required = dict.fromkeys(limits, 0)
    failures = []
    for unit in units:
        longest = max(r['frames'] for r in unit['rows'])
        need = dict(whole_max_frames=longest, frame_budget=longest*len(unit['rows']), microbatch=len(unit['rows']))
        for key, value in need.items():
            required[key] = max(required[key], value)
        failed = {key: value for key, value in need.items() if value > limits[key]}
        if failed:
            failures.append(dict(source_id=unit['source_id'], rows=unit['rows'], required=failed))
    if failures:
        flags = ' '.join('--'+key.replace('_', '-')+' '+str(value) for key, value in required.items() if value > limits[key])
        raise ValueError(f'{scope} full-utterance resource preflight failed for {len(failures)} units; '
            f'example={failures[0]}; minimum guard settings: {flags}. Check GPU memory; no crop or silent skip is used.')
    return dict(scope=scope, units=len(units), valid=True, limits=limits, minimum_required=required,
        feature_geometry=dict(sample_rate=16000, support_samples=560, stride_samples=320, short_reflection_samples=3920),
        waveform_policy='Full recordings; only inputs shorter than 3920 samples are reflected; no overlapping feature windows.',
        estimate='Header bounds are not measured GPU memory; attention, backward, optimizer and PCL activations need server acceptance.')


def audit_training(train, dev, cfg):
    metadata, methods = {}, {}
    for row in train:
        if row.get('split') != 'train' or row.get('condition') not in ('online', 'offline'):
            raise ValueError('Training resource inventory requires original Train Online/Offline rows')
        samples, method = _samples(row)
        metadata[row['id']] = samples
        methods[method] = methods.get(method, 0)+1
    by_id = {r['id']: r for r in train}
    units, paired = [], 0
    pcl_enabled = cfg.get('consistency') not in ('none', None) or bool(cfg.get('pcl_weight', cfg.get('pcl_platform_weight', 0.)))
    for row in train:
        samples = metadata[row['id']]
        upper = (samples*1000+969)//970
        members = [dict(id=row['id'], role='online_augmented' if row['condition'] == 'online' else 'noisy', frames=frame_count(upper), classification=True)]
        offline = by_id.get(row.get('source_id'))
        if pcl_enabled and row['condition'] == 'online' and offline is not None and offline.get('condition') == 'offline':
            # Worst case: the augmented Online CE waveform cannot be reused
            # as its original Online PCL reference. The three views are atomic.
            members.extend((dict(id=offline['id'], role='pcl_original_offline', frames=frame_count(metadata[offline['id']]), classification=False),
                dict(id=row['id'], role='pcl_original_online', frames=frame_count(samples), classification=False)))
            paired += 1
        if cfg.get('consistency')=='dual' and row['condition']=='offline':
            members.append(dict(id=row['id'],role='pcl_original_offline',frames=frame_count(samples),classification=False))
        units.append(dict(source_id=row.get('canonical_source', row['id']), rows=members))
    result = _check_units(units, cfg, 'Train')
    result.update(source_metadata=methods, augmentation_length_bound='ceil(original_samples * 1000 / 970)',
        auxiliary_policy='Four official pairs plus four source/Noisy pairs; up to twelve reference views; references never classified.',
        maximum_pcl_pairs_per_task=(8 if cfg.get('consistency')=='dual' else 4) if pcl_enabled else 0, maximum_extra_pcl_views_per_task=(12 if cfg.get('consistency')=='dual' else 8) if pcl_enabled else 0,
        possible_paired_online_rows=paired, encoder_adaptation=dict(lora_layers=24, checkpoint_layers=cfg.get('checkpoint_layers', 24)))
    result['dev'] = audit_inference(dev, cfg)
    return result


def audit_inference(rows, cfg):
    units, methods = [], {}
    for row in rows:
        samples, method = _samples(row)
        methods[method] = methods.get(method, 0)+1
        units.append(dict(source_id=row['id'], rows=[dict(id=row['id'], role='inference', frames=frame_count(samples), classification=True)]))
    result = _check_units(units, cfg, 'Inference')
    result['source_metadata'] = methods
    return result
