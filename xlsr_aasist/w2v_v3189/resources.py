"""Header-only full-waveform encoder / bounded AASIST window resource preflight."""
from pathlib import Path

SR=16000
MIN_SAMPLES=3920


def _samples(row):
    value=row.get('output_samples')
    if value is not None:
        if type(value) is not int or value<1:
            raise ValueError('Invalid output_samples for '+str(row.get('id','unknown')))
        return value,'saved_16khz_samples'
    import soundfile as sf
    path=Path(row['audio']).expanduser()
    info=sf.info(path)
    if info.frames<1 or info.samplerate<1 or info.channels<1:
        raise ValueError('Invalid audio header for '+str(row.get('id',path)))
    # scipy.signal.resample_poly (the actual loader) returns ceil(n*up/down).
    return (int(info.frames)*SR+int(info.samplerate)-1)//int(info.samplerate),'audio_header'


def _frames(samples):
    return (max(MIN_SAMPLES,samples)-400)//320+1


def _limits(cfg):
    return dict(whole_max_frames=int(cfg.get('whole_max_frames',3000)),
        frame_budget=int(cfg.get('frame_budget',9000)),
        microbatch=int(cfg.get('microbatch',12)))


def _check_units(units,cfg,scope):
    limits=_limits(cfg);violations=[];requirements=dict(whole_max_frames=0,frame_budget=0,microbatch=0)
    window=int(cfg.get('window',128));hop=int(cfg.get('hop',64));window_batch=int(cfg.get('window_batch',8))
    if window<12 or not 1<=hop<=window or window_batch<1:raise ValueError('Invalid AASIST window geometry')
    maxima=dict(classification_frames=0,auxiliary_frames=0,atomic_views=0,atomic_padded_frames=0)
    for unit in units:
        rows=unit['rows'];longest=max(r['frames'] for r in rows);view_count=len(rows)
        needs=dict(whole_max_frames=longest,
            frame_budget=longest*view_count,microbatch=view_count)
        for name,value in needs.items():requirements[name]=max(requirements[name],value)
        for row in rows:
            name='classification_frames' if row['classification'] else 'auxiliary_frames'
            maxima[name]=max(maxima[name],row['frames'])
        maxima['atomic_views']=max(maxima['atomic_views'],view_count)
        maxima['atomic_padded_frames']=max(maxima['atomic_padded_frames'],longest*view_count)
        failed={name:value for name,value in needs.items() if value>limits[name]}
        if failed:
            violations.append(dict(source_id=unit['source_id'],rows=[{k:r[k] for k in ('id','role','frames')} for r in rows],required=failed))
    report=dict(scope=scope,units=len(units),limits=limits,minimum_required= requirements,maxima=maxima,
        valid=not violations,over_limit_units=len(violations),violations=violations[:20],
        waveform_policy='Encode full waveforms once; reflect only inputs shorter than 3920 samples after RTC; cover all fused feature frames with overlapping AASIST windows; no audio truncation.',
        aasist_windows=dict(window=window,hop=hop,window_batch=window_batch,
            max_temporal_graph_cells_per_window=(window//3)**2),
        encoder_adaptation=dict(lora_layers=cfg.get('lora_layers',60),
            checkpoint_layers=cfg.get('checkpoint_layers',60),
            note='All-depth adapter training retains a backward path through the whole encoder; checkpointing trades additional computation for activation memory.'),
        estimate='Resource guards are bounds, not measured GPU-memory guarantees; they do not predict backward activation or optimizer memory.')
    if violations:
        flags=' '.join('--'+key.replace('_','-')+' '+str(value) for key,value in requirements.items() if value>limits[key])
        examples='; '.join(str(v['source_id'])+' ['+', '.join(r['role']+':'+str(r['frames'])+' frames' for r in v['rows'])+']' for v in violations[:5])
        raise ValueError(f'{scope} full-utterance resource preflight failed for {len(violations)} units: {examples}. '
            f'Minimum guard settings needed: {flags}. Check GPU memory before choosing larger values; '
            'audio was not decoded, cropped or silently skipped.')
    return report


def audit_training(train,dev,cfg):
    """Include every possible scheduled view, before spending an epoch on it.

    V3.18.9 families include 0.970x speed, so conservatively bound every base
    view by ceil(n*1000/970). All other families preserve the complete length.
    Official auxiliary Online is the unaugmented recorded receiver waveform.
    """
    enabled=cfg.get('consistency','dual')!='none'
    noise=enabled and cfg.get('pcl_noise_weight',.15)>0
    platform=enabled and cfg.get('pcl_platform_weight',.045)>0 and cfg.get('platform_sources_per_group',3)>0
    online={};counts={};metadata={};units=[]
    for row in train:
        if row.get('condition') not in ('online','offline'):
            raise ValueError('Training resource inventory must contain original Online/Offline only')
        if row.get('split')!='train':raise ValueError('Training resource inventory contains a non-Train row')
        samples,method=_samples(row);metadata[row['id']]=(samples,method)
        counts[method]=counts.get(method,0)+1
        if row['condition']=='online':
            source=row['source_id']
            if source in online:raise ValueError('Duplicate official Online source in resource inventory')
            online[source]=row
    for row in train:
        samples,_=metadata[row['id']]
        upper=(samples*1000+969)//970
        frames=_frames(upper)
        if row['condition']=='online':
            members=[dict(id=row['id'],role='online_augmented',frames=frames,classification=True)]
        else:
            members=[dict(id=row['id'],role='noisy',frames=frames,classification=True)]
            official=online.get(row['source_id']) if platform else None
            if noise or official is not None:
                members.append(dict(id=row['id'],role='clean_reference',frames=frames,classification=False))
            if official is not None:
                members.append(dict(id=official['id'],role='official_reference',frames=_frames(metadata[official['id']][0]),classification=False))
        units.append(dict(source_id=row['id'],rows=members))
    result=_check_units(units,cfg,'Train')
    result.update(source_metadata=counts,augmentation_length_bound='ceil(original_samples * 1000 / 970)',
        auxiliary_policy='Worst-case valid PCL; invalid alignment may reduce actual encoded views.')
    result['dev']=audit_inference(dev,cfg)
    return result


def audit_inference(rows,cfg):
    """Validate all fixed Dev/submission rows using saved sizes or file headers."""
    units=[];counts={}
    for row in rows:
        samples,method=_samples(row);counts[method]=counts.get(method,0)+1
        units.append(dict(source_id=row['id'],rows=[dict(id=row['id'],role='inference',
            frames=_frames(samples),classification=True)]))
    result=_check_units(units,cfg,'Inference')
    result['source_metadata']=counts
    return result
