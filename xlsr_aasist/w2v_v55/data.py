"""V5 contextual receiver pairs, shared input policy, and original-prefix PCL."""
from collections import OrderedDict, defaultdict, deque
import math
import numpy as np
from w2v_v3184.data import Waves as BaseWaves,close,TicketSampler
from w2v_v3189.workers import loader
from w2v_v3189.views import recipe as view_recipe, augment
from w2v_v3189.augment import NoiseBank,Engines,recipe,generate,estimate_delay,reliable_spans
from w2v_v5.processing import generate_matched_context, prepare_wave
from w2v_v3189.activity import mask_pairs
from w2v_v31810.data import make_recipe
from w2v_v3184.common import seed_for



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

    def __getitem__(self, ticket):
        units = self._unprocessed(ticket)
        def processed(row):
            x, meta = prepare_wave(row['wave'][:row['original_samples']], self.cfg['v5_input'])
            return dict(row, wave=x, input_processing=meta,
                        pcl_valid_frames=meta['pcl_valid_frames'])
        output = []
        for source in units:
            row = processed(source)
            if 'hp_positive2' in source:
                row['hp_positive2'] = processed(source['hp_positive2'])
            refs = {key: processed(value) for key, value in source.get('pcl_refs', {}).items()}
            pairs = {}
            for kind, matches in source.get('pcl_pairs', {}).items():
                if kind == 'noise':
                    left, right = refs['clean'], row
                elif kind == 'platform' and 'official' in refs:
                    left, right = refs['official'], refs['clean']
                else:
                    pairs[kind] = []
                    continue
                safe, _ = mask_pairs(matches, left['input_processing']['pcl_excluded_spans'],
                                     right['input_processing']['pcl_excluded_spans'])
                pairs[kind] = _valid_pairs(safe, left, right, self.cfg.get('pcl_min_frames', 1))
            if 'pcl_refs' in source:
                row.update(pcl_refs=refs, pcl_pairs=pairs)
            output.append(row)
        return output

    def _phones(self,row):
        if self.phonemes is None:
            from w2v_v3189.phonemes import PhonemeCache
            manifest=self.cfg.get('phoneme_manifest')
            if not manifest:raise ValueError('PCL requires the prepared phoneme manifest')
            parent_rows=[r for r in self.rows if r.get("v5_origin_split", "train")=="train"]
            self.phonemes=PhonemeCache(manifest,parent_rows,binding=self.cfg.get('phoneme_provenance'))
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

    def _unprocessed(self,t):
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
        for key in ('hp_group', 'hp_member_role', 'hp_skip'):
            if key in t: base[key] = t[key]
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
                clean,noisy,meta=generate_matched_context(base_wave,noise_recipe,self.bank,self.engines, context_seconds=self.cfg["v5_context_seconds"])
                selected_meta=meta
            except Exception as exc:raise RuntimeError('V5 Noisy/reference failed: '+base['view_id']) from exc
            main.append(prepared(base,noisy,role='noisy',condition='noisy',augmentation=meta,
                                 noise_recipe_view=view))
        assert clean is not None
        if t.get('hp_member_role') == 'anchor':
            if row['label'] != 1:
                raise ValueError('Only Real is a hard-positive anchor')
            import copy
            from w2v_v5.processing import preceding_context, _target_metadata
            # Keep the existing first view byte-for-byte. The other recipe lane
            # samples a second receiver condition from V5.1's unchanged law.
            alternative = dict(t, noise_recipe_view=1-int(t.get('noise_recipe_view', 0)))
            alternative['occurrence'] = str(t['occurrence']) + '/v55-positive2'
            rec2 = make_recipe(alternative, row, self.cfg, self.split)
            context = preceding_context(base_wave, self.cfg['v5_context_seconds'])
            stream = np.concatenate((context, base_wave)) if len(context) else base_wave
            noisy2, meta2 = generate(stream, rec2, self.bank, self.engines)
            if len(noisy2) != len(stream):
                raise ValueError('Second positive changed source duration')
            meta2 = (_target_metadata(meta2, len(context), len(base_wave))
                     if len(context) else copy.deepcopy(meta2))
            meta2['recipe'] = rec2
            meta2['receiver_context'] = dict(context_samples=len(context),
                target_samples=len(base_wave), fresh_state_per_view=True,
                same_source_context=True, processing_scope='one_extended_stream_then_crop')
            main[0]['hp_positive2'] = prepared(base, noisy2[len(context):].copy(),
                role='noisy', condition='noisy', augmentation=meta2, noise_recipe_view=alternative['noise_recipe_view'])
        # CTC timestamps belong to the real source waveform before ready()'s
        # exceptional short-clip padding. Never apply PCL to repeated content.
        refs={'clean':prepared(base,clean,role='clean_ref',condition='clean_ref',augmentation=selected_meta)}
        pairs={"noise": [], "platform": []};coverage={};timing={}
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
        return main

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
            except Exception as exc:raise RuntimeError('V5 fixed-view generation failed: '+str(row['id'])) from exc
            result.append(prepared(base,wave,role='noisy',condition=t.get('scene') or t.get('family') or 'noisy',augmentation=meta))
        return result
