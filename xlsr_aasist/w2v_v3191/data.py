"""Matched receiver candidates with full content and conservative PCL masking."""
from collections import OrderedDict, defaultdict, deque
import math
import numpy as np
from w2v_v3184.data import Waves as BaseWaves,close,TicketSampler
from .workers import loader
from .views import recipe as view_recipe, augment
from .augment import NoiseBank,Engines,recipe,generate,generate_matched,estimate_delay,reliable_spans
from .activity import mask_pairs
from w2v_v31810.data import make_recipe
from w2v_v3184.common import seed_for
from .augment import apply_pair, quality_check, mechanism_catalog, IMPLEMENTATIONS

MIN_SAMPLES=3920


def ready(wave):
    """Append a time reflection only to satisfy the 12-frame backend minimum.

    Endpoint samples are not doubled at a join. The entire original prefix is
    unchanged. A one-sample recording has no time direction and uses constant
    padding; it has zero valid PCL frames, as do all inputs shorter than 400.
    """
    x=np.asarray(wave,np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty/nonfinite/non-mono input cannot be padded')
    if len(x)>=MIN_SAMPLES:return x
    return np.pad(x,(0,MIN_SAMPLES-len(x)),mode='reflect' if len(x)>1 else 'edge')


def prepared(row,wave,**metadata):
    original=len(wave);output=ready(wave)
    return dict(row,**metadata,wave=output,original_samples=original,
        pcl_valid_frames=max(0,(original-400)//320+1),
        short_padding=dict(policy='time_reflection_v1',original_samples=original,
            padded_samples=len(output),added_samples=len(output)-original,
            degenerate_single_sample=original==1,
            pcl_policy='Original prefix only; exclude CNN receptive fields crossing artificial samples'))


def _valid_pairs(pairs,left,right,min_frames):
    """Avoid encoding references whose phone intervals contain no real frames."""
    result=[]
    for match in pairs:
        valid=True
        for row,start,end in ((left,match[0],match[1]),(right,match[2],match[3])):
            frames=row['pcl_valid_frames']
            a=max(0,min(frames,math.ceil((float(start)-200.)/320.)))
            b=max(0,min(frames,math.ceil((float(end)-200.)/320.)))
            valid=valid and b-a>=min_frames and 0<=start<end<=row['original_samples']
        if valid:result.append(match)
    return result


def _identity(length):
    return dict(scale=1.,offset=0.,input_samples=length,output_samples=length)


class Waves(BaseWaves):
    def __init__(self,rows,cfg,split='train'):
        super().__init__(rows,cfg,split)
        self.phonemes=None;self.phone_matches=OrderedDict()

    def _phones(self,row):
        if self.phonemes is None:
            from .phonemes import PhonemeCache
            manifest=self.cfg.get('phoneme_manifest')
            if not manifest:raise ValueError('PCL requires the prepared phoneme manifest')
            self.phonemes=PhonemeCache(manifest,self.rows,binding=self.cfg.get('phoneme_provenance'))
        return self.phonemes.get(row)['segments']

    def _matches(self,off,on,off_segments,on_segments):
        from w2v_v3186.phonemes import match_segments
        key=(off['id'],on['id'])
        if key not in self.phone_matches:
            matches=match_segments(off_segments,on_segments)
            self.phone_matches[key]=[(off_segments[a]['instance'],on_segments[b]['instance']) for a,b in matches]
            if len(self.phone_matches)>1024:self.phone_matches.popitem(last=False)
        self.phone_matches.move_to_end(key)
        return self.phone_matches[key]

    def __getitem__(self,t):
        # All paths use the same post-augmentation reflection policy, including
        # Dev, submission, probes, CE-only runs and head warmup.
        if isinstance(t,int):
            row=self.rows[t]
            return [prepared(row,self.original(row),row_index=t)]
        if 'base_variant' not in t:return self._fixed_ticket(t)
        if self.split!='train' or t.get('probe') or t.get('scene') or t.get('family'):
            raise ValueError('PCL is only allowed for scheduled Train tickets')
        row=self.rows[t['index']]
        rec=view_recipe(t.get('recipe_seed',self.cfg['seed']),row['id'],t['kind'],t['base_variant'],
            family=t.get('base_family'),generation=t.get('base_generation',0),dynamic=t.get('base_dynamic',False))
        source=self.original(row);base_wave,base_meta=augment(source,rec)
        base=dict(row,occurrence=t['occurrence'],base_augmentation=base_meta,
            view_id=t['base_view_id'])
        for key in ('batch_group_quotas','pcl_platform_budget','source_cycle','canonical_source','meta_side','task_kind','task_direction','condition_block','domain_override','heldout_condition','nuisance_seed','calibration'):
            if key in t:base[key]=t[key]
        if t['kind']=='online':
            result=prepared(base,base_wave,role='online')
            if self.cfg.get('consistency')=='dual' and t['epoch']<self.cfg.get('warm_epochs',1):
                # Calibrate true phone features on already-encoded Train Online
                # only. No clean reference or extra encoder pass in warmup.
                from w2v_v3186.phonemes import map_segments
                phones=map_segments(self._phones(row),base_meta['time_map'],len(base_wave))
                result['pcl_calibration_spans']=[[p['start_sample'],p['end_sample']] for p in phones]
                result['split']='train'
            return [result]
        if self.bank is None:
            self.bank=NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2,
                self.cfg['scene_noise_records'][self.split])
            if getattr(self.bank,'scenes',None) is not None:
                self.bank.scenes.cap=self.cfg.get('scene_cache_mib',32)*1024**2
            self.engines=Engines(self.cfg['ffmpeg'])
        active=bool(t.get('pcl_enabled'));chosen=0
        main=[];clean=None;selected_meta=None
        # Exactly ONE synthesis per source visit. Alternate recipe lanes across
        # source cycles, without creating and subsequently discarding view B.
        for view in (int(t.get('noise_recipe_view',0)),):
            noise_recipe=make_recipe(t,row,self.cfg,self.split)
            try:
                # Matched Clean-RTC is classified in all arms, including when
                # PCL is disabled. Never use raw Offline as this CE reference.
                clean,noisy,meta=generate_matched(base_wave,noise_recipe,self.bank,self.engines)
                selected_meta=meta
            except Exception as exc:raise RuntimeError('V3.19.1 Noisy/reference failed: '+base['view_id']) from exc
            main.append(prepared(base,noisy,role='noisy',condition='noisy',augmentation=meta,
                                 noise_recipe_view=view))
        assert clean is not None
        # CTC timestamps belong to the real source waveform before ready()'s
        # exceptional short-clip padding. Never apply PCL to repeated content.
        refs={'clean':prepared(base,clean,role='clean_ref',condition='clean_ref',augmentation=selected_meta)}
        pairs={};coverage={};timing={}
        if not active:
            main[0].update(pcl_refs=refs,pcl_pairs=pairs,pcl_noisy_index=0,
                pcl_coverage=coverage,pcl_timing=timing)
            return self._receiver_candidates(main,t)
        from w2v_v3186.phonemes import map_segments
        original_phones=self._phones(row)
        base_phones=map_segments(original_phones,base_meta['time_map'],len(base_wave))
        clean_delay=estimate_delay(base_wave,clean)
        timing['clean']=clean_delay
        if clean_delay['valid']:
            clean_map=dict(base_meta['time_map'],offset=base_meta['time_map']['offset']+clean_delay['offset'],output_samples=len(clean))
            clean_phones=map_segments(original_phones,clean_map,len(clean))
            kept=reliable_spans(base_wave,clean,
                [(p['start_sample'],p['end_sample']) for p in base_phones],0.,clean_delay['offset'])
            allowed={p['instance'] for p,valid in zip(base_phones,kept) if valid}
            clean_phones=[p for p in clean_phones if p['instance'] in allowed]
        else:clean_phones=[]
        clean_lookup={p['instance']:p for p in clean_phones}
        if t.get('pcl_noise'):
            noisy=main[chosen]['wave'][:len(base_wave)]
            delay=estimate_delay(clean,noisy);timing['noise']=delay
            coverage['noise_candidates']=len(base_phones)
            noise_pairs=[]
            if delay['valid'] and clean_phones and selected_meta.get('pair_eligible',False):
                spans=[(p['start_sample'],p['end_sample']) for p in clean_phones]
                kept=reliable_spans(clean,noisy,spans,0.,delay['offset'])
                for phone,valid in zip(clean_phones,kept):
                    if valid:
                        a,b=phone['start_sample'],phone['end_sample']
                        noise_pairs.append([a,b,a+delay['offset'],b+delay['offset']])
            safe,affected=mask_pairs(noise_pairs,selected_meta.get('clean_activity',{}).get('suppressed_spans',[]),
                selected_meta.get('rtc_activity',{}).get('suppressed_spans',[]))
            pairs['noise']=_valid_pairs(safe,refs['clean'],main[chosen],self.cfg.get('pcl_min_frames',1))
            coverage['noise_activity_affected_phones']=affected
            coverage['noise_activity_safe_segments']=len(safe)
            coverage['noise_original_frame_rejected']=len(safe)-len(pairs['noise'])
            coverage['noise_valid']=len(pairs['noise'])
        if 'pcl_online_index' in t:
            official=self.rows[t['pcl_online_index']]
            if any(official.get(k)!=row.get(k) for k in ('source_id','group_id','language','label')):
                raise ValueError('Official auxiliary source identity disagrees')
            ow=self.original(official)
            refs['official']=prepared(official,ow,occurrence=t['occurrence'],role='official_ref',
                base_augmentation={'time_map':_identity(len(ow))})
            on_phones=self._phones(official)
            matches=self._matches(row,official,original_phones,on_phones)
            on_lookup={p['instance']:p for p in on_phones}
            coverage['platform_candidates']=len(matches)
            platform=[]
            for off_instance,on_instance in matches:
                if off_instance not in clean_lookup or on_instance not in on_lookup:continue
                op=on_lookup[on_instance];cp=clean_lookup[off_instance]
                a,b=int(op['start_sample']),int(op['end_sample'])
                if a<0 or b>len(ow) or b-a<80 or np.mean(ow[a:b].astype(np.float64)**2)<1e-10:continue
                platform.append([a,b,cp['start_sample'],cp['end_sample']])
            safe,affected=mask_pairs(platform,[],selected_meta.get('clean_activity',{}).get('suppressed_spans',[]))
            pairs['platform']=_valid_pairs(safe,refs['official'],refs['clean'],self.cfg.get('pcl_min_frames',1))
            coverage['platform_activity_affected_phones']=affected
            coverage['platform_activity_safe_segments']=len(safe)
            coverage['platform_original_frame_rejected']=len(safe)-len(pairs['platform'])
            coverage['platform_valid']=len(pairs['platform'])
        main[0].update(pcl_refs=refs,pcl_pairs=pairs,pcl_noisy_index=chosen,
            pcl_coverage=coverage,pcl_timing=timing)
        return self._receiver_candidates(main,t)

    def _receiver_candidates(self, main, ticket):
        """Build every candidate in every arm, without consulting model/label.

        Candidate 0 is the exact inherited waveform. New candidates add a
        postprocessor to both receiver views. Only the chosen full unit is
        encoded for training; candidate scoring is handled by the trainer.
        """
        row=main[0]
        if row.get('role')!='noisy':
            return main
        mechanisms=mechanism_catalog(self.cfg)['train']
        count=int(self.cfg.get('robust_candidate_count',1+len(mechanisms)))
        if count<2 or count>1+len(mechanisms):
            raise ValueError('Candidate count must include baseline plus 1..all training mechanisms')
        base_seed=seed_for(ticket.get('recipe_seed',self.cfg['seed']),
            ticket['occurrence'],ticket.get('epoch',0),
            ticket.get('canonical_source',row.get('source_id',row['id'])),'v3191-receiver')
        start=base_seed%len(mechanisms)
        selected=[mechanisms[(start+index)%len(mechanisms)] for index in range(count-1)]
        original=row['wave'][:row['original_samples']]
        clean_row=row['pcl_refs']['clean']
        clean=clean_row['wave'][:clean_row['original_samples']]
        baseline_quality=quality_check(clean,clean,original,self.cfg)
        # Keep the old control distribution unchanged. Signal quality checks
        # gate NEW choices, while the inherited control remains the fallback.
        control_quality=dict(baseline_quality,passed=True,eligible=True,
            reason='inherited_control',policy='Baseline is not filtered by the new candidate quality gate',
            diagnostic_passed=baseline_quality['passed'])
        control=dict(row)
        control['robust_mechanism']='baseline'
        control['robust_quality']=control_quality
        control['robust_candidate']=dict(id='baseline',mechanism='baseline',
            implementation=IMPLEMENTATIONS['baseline'],seed=base_seed,quality=control_quality)
        control['robust_candidate_id']='baseline'
        candidates=[[control]+list(main[1:])]
        for index,mechanism in enumerate(selected,1):
            candidate_seed=seed_for(base_seed,mechanism,'receiver-parameters')
            clean_new,noisy_new,meta=apply_pair(clean,original,mechanism,candidate_seed,self.cfg)
            candidate=prepared(row,noisy_new,
                augmentation=dict(row.get('augmentation',{}),receiver_postprocess=meta))
            new_refs=dict(row['pcl_refs'])
            new_refs['clean']=prepared(clean_row,clean_new,
                augmentation=dict(clean_row.get('augmentation',{}),receiver_postprocess=meta))
            new_pairs={};coverage=dict(row.get('pcl_coverage',{}))
            clean_spans=meta['clean']['affected_spans']
            noisy_spans=meta['noisy']['affected_spans']
            for kind,pairs in row.get('pcl_pairs',{}).items():
                if kind=='noise':
                    left,right=new_refs['clean'],candidate
                    left_spans,right_spans=clean_spans,noisy_spans
                elif kind=='platform':
                    left,right=new_refs['official'],new_refs['clean']
                    left_spans,right_spans=[],clean_spans
                else:
                    raise ValueError('Unknown inherited PCL branch: '+str(kind))
                safe,affected=mask_pairs(pairs,left_spans,right_spans)
                new_pairs[kind]=_valid_pairs(safe,left,right,self.cfg.get('pcl_min_frames',1))
                coverage[kind+'_receiver_affected_phones']=affected
                coverage[kind+'_receiver_valid']=len(new_pairs[kind])
                coverage[kind+'_valid']=len(new_pairs[kind])
            identifier=f'{mechanism}:{candidate_seed}'
            candidate.update(pcl_refs=new_refs,pcl_pairs=new_pairs,pcl_coverage=coverage,
                pcl_timing=dict(row.get('pcl_timing',{}),receiver_postprocess={
                    'clean':meta['clean']['alignment'],'noisy':meta['noisy']['alignment']}),
                robust_mechanism=mechanism,robust_quality=meta['quality'],
                robust_candidate_id=identifier,
                robust_candidate=dict(id=identifier,mechanism=mechanism,index=index,
                    implementation=meta['implementation'],seed=candidate_seed,
                    quality=meta['quality'],processing=meta))
            candidates.append([candidate]+list(main[1:]))
        # Never put a list inside its own first candidate: workers pickle units.
        wrapper=dict(control,robust_candidates=candidates)
        return [wrapper]+list(main[1:])

    def _fixed_ticket(self,t):
        row=self.rows[t['index']];source=self.original(row)
        base=dict(row,occurrence=t['occurrence'])
        if t['kind']=='online':return [prepared(base,source,role='online')]
        if self.bank is None:
            self.bank=NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2,
                self.cfg.get('scene_noise_records',{}).get(self.split))
            if getattr(self.bank,'scenes',None) is not None:
                self.bank.scenes.cap=self.cfg.get('scene_cache_mib',32)*1024**2
            self.engines=Engines(self.cfg['ffmpeg'])
        result=[];views=(0,)
        for view in views:
            rec=recipe(self.cfg['seed'],t['occurrence'],row['group_id'],view,t['epoch'],self.cfg['warm_epochs'],
                       self.split,t.get('family'),t.get('probe',False),
                       self.cfg['rtc_stress_probability'],self.cfg['rtc_stress_ramp_epochs'],
                       self.cfg.get('scene_probability',.8),self.cfg.get('rtc_policy','legacy'),t.get('scene'),
                       activity_policy=self.cfg.get('rtc_activity_policy','mixed'))
            try:wave,meta=generate(source,rec,self.bank,self.engines)
            except Exception as exc:raise RuntimeError('V3.19.1 fixed-view generation failed: '+str(row['id'])) from exc
            result.append(prepared(base,wave,role='noisy',condition=t.get('scene') or t.get('family') or 'noisy',augmentation=meta))
        return result


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
        if platform:needed.append('official')
        extra=[refs[key] for key in needed]
        rows=list(unit)+extra
        n=max(max(1,(len(r['wave'])-400)//320+1) for r in rows)
        if len(rows)>size or len(rows)*n>frame_budget:
            raise ValueError('A complete source/PCL unit exceeds microbatch/frame budget; '
                'increase --microbatch/--frame-budget rather than truncating audio')
        row=unit[0];group=(row.get('language',''),row.get('label',-1),row.get('role',''))
        priority=0 if platform else (1 if noise else 2)
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
