"""V5 views plus a reusable official reference; no added phoneme dependency."""
from collections import defaultdict, deque
from w2v_v5.data import Waves as ParentWaves, prepared, _identity, TicketSampler, loader, close
from w2v_v5.processing import prepare_wave


class Waves(ParentWaves):
    def __getitem__(self, ticket):
        units = super().__getitem__(ticket)
        if not isinstance(ticket, dict) or 'contrastive_online_index' not in ticket:
            if isinstance(ticket, dict) and ticket.get('contrastive_status'):
                units[0]['contrastive_skip'] = ticket['contrastive_status']
            return units
        if self.split != 'train' or ticket.get('meta_side') != 'A' or len(units) != 1:
            raise ValueError('Contrastive references are only scheduled for A-side Train sources')
        row = units[0]
        official = self.rows[ticket['contrastive_online_index']]
        for key in ('source_id', 'canonical_source', 'language', 'label', 'group_id'):
            if not row.get(key) and key in ('source_id', 'canonical_source'):
                raise ValueError('Missing canonical contrastive identity')
            if official.get(key) != row.get(key):
                raise ValueError('Official contrastive reference disagrees on ' + key)
        if official.get('split') != 'train' or official.get('condition') != 'online':
            raise ValueError('Contrastive reference must be official Train Online')
        reason = None
        if not row.get('augmentation', {}).get('pair_eligible', False):
            reason = 'invalid_receiver_pair'
        elif row.get('original_samples', 0) < 400:
            reason = 'no_original_frames'
        if reason:
            row['contrastive_skip'] = reason
            return units
        refs = row['pcl_refs']
        if 'official' not in refs:
            wave = self.original(official)
            ref = prepared(official, wave, occurrence=ticket['occurrence'], role='official_ref',
                           base_augmentation={'time_map': _identity(len(wave))})
            wave, meta = prepare_wave(ref['wave'][:ref['original_samples']], self.cfg['v5_input'])
            refs['official'] = dict(ref, wave=wave, input_processing=meta,
                                    pcl_valid_frames=meta['pcl_valid_frames'])
        if refs['official'].get('original_samples', 0) < 400:
            row['contrastive_skip'] = 'no_official_frames'
            return units
        row.update(contrastive_enabled=True, contrastive_group_budget=ticket['contrastive_group_budget'])
        return units


def microbatches(units,size,frame_budget,*,weights=None):
    """Best-fit decreasing packing of complete source/PCL units.

    Units are never split or dropped. With active PCL, pack platform pairs,
    then other pairs, before filling spare capacity with classification-only
    units. Otherwise Online singletons can leave fewer than four paired
    sources in every physical batch and silently disable variance learning.
    Within each phase, nearby lengths and independent class/language groups
    retain the existing decreasing-length round-robin order. CE-only packing
    is unchanged. GN/LN do not require mixed Online/Noisy normalization batches.
    Callers that need original output order must use row_index, not iteration.
    """
    if type(size)is not int or size<1 or type(frame_budget)is not int or frame_budget<1:
        raise ValueError('Positive microbatch and frame budgets required')
    records=[]
    for position,unit in enumerate(units):
        if not unit:raise ValueError('Empty source unit')
        refs=unit[0].get('pcl_refs',{});pairs=unit[0].get('pcl_pairs',{})
        needed=[]
        noise=bool(pairs.get('noise')) and (weights is None or bool(weights.get('noise')))
        platform=bool(pairs.get('platform')) and (weights is None or bool(weights.get('platform')))
        # Prepared training candidates always carry a clean CE reference;
        # ordinary Dev/submission rows are single-view inference inputs.
        if noise or platform or (unit[0].get('role')=='noisy' and 'clean' in refs):needed.append('clean')
        contrast=bool(unit[0].get('contrastive_enabled'))
        if platform or contrast:needed.append('official')
        extra=[refs[key] for key in needed]
        rows=list(unit)+extra
        n=max(max(1,(len(r['wave'])-400)//320+1) for r in rows)
        if len(rows)>size or len(rows)*n>frame_budget:
            raise ValueError('A complete source/PCL unit exceeds microbatch/frame budget; '
                'increase --microbatch/--frame-budget rather than truncating audio')
        row=unit[0];group=(row.get('language',''),row.get('label',-1),row.get('role',''))
        priority=0 if (platform or contrast) else (1 if noise else 2)
        records.append((n,len(rows),group,position,unit,priority))
    paired=any(record[5]<2 for record in records)
    # Nearby lengths (<=31 frames apart) can interleave independent groups
    # without requiring an entire shortest/longest batch to share padding.
    buckets=defaultdict(lambda:defaultdict(deque))
    for record in sorted(records,key=lambda item:(item[5],-item[0],item[3])):
        buckets[(record[5],-(record[0]//32))][record[2]].append(record)
    records=[]
    for bucket in sorted(buckets):
        groups=buckets[bucket]
        while any(groups.values()):
            for group in sorted(groups):
                if groups[group]:records.append(groups[group].popleft())
    bins=[]
    for length,count,group,position,unit,priority in records:
        source=unit[0].get('source_id',unit[0].get('id',position))
        choices=[]
        for index,entry in enumerate(bins):
            if paired and source in entry['sources']:
                # An Online view and an Offline-derived unit of the same source
                # never count as two independent sources for physical PCL.
                continue
            total=entry['count']+count;longest=max(entry['longest'],length)
            if total<=size and total*longest<=frame_budget:
                choices.append((source in entry['sources'],group in entry['groups'],size-total,index))
        if choices:
            index=min(choices)[-1];entry=bins[index]
        else:
            entry=dict(units=[],count=0,longest=0,sources=set(),groups=set());bins.append(entry)
        entry['units'].append(unit);entry['count']+=count;entry['longest']=max(entry['longest'],length)
        entry['sources'].add(source);entry['groups'].add(group)
    for entry in bins:yield entry['units']
