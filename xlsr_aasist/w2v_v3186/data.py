"""Unchanged classification views; shared-base, timing-aware PCL references."""
from collections import OrderedDict
import numpy as np
from w2v_v3185.data import Waves as BaseWaves
from w2v_v3184.data import ready,loader,close,TicketSampler
from .views import recipe as view_recipe, augment
from .augment import NoiseBank,Engines,recipe,generate,generate_matched,estimate_delay,reliable_spans


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
        from .phonemes import match_segments
        key=(off['id'],on['id'])
        if key not in self.phone_matches:
            matches=match_segments(off_segments,on_segments)
            self.phone_matches[key]=[(off_segments[a]['instance'],on_segments[b]['instance']) for a,b in matches]
            if len(self.phone_matches)>1024:self.phone_matches.popitem(last=False)
        self.phone_matches.move_to_end(key)
        return self.phone_matches[key]

    def __getitem__(self,t):
        # Fixed probes and all validation continue to use the original
        # V3.18.5 distribution. A CE-only run creates no auxiliary references.
        if (isinstance(t,int) or 'base_variant' not in t or
            t.get('kind')=='online' or not t.get('pcl_enabled')):
            return super().__getitem__(t)
        if self.split!='train' or t.get('probe') or t.get('scene') or t.get('family'):
            raise ValueError('PCL is only allowed for scheduled Train tickets')
        row=self.rows[t['index']]
        rec=view_recipe(self.cfg['seed'],row['id'],t['kind'],t['base_variant'],
            family=t.get('base_family'),generation=t.get('base_generation',0),dynamic=t.get('base_dynamic',False))
        source=self.original(row);base_wave,base_meta=augment(source,rec)
        base=dict(row,occurrence=t['occurrence'],base_augmentation=base_meta,
            view_id=t['base_view_id'])
        if self.bank is None:
            self.bank=NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2,
                self.cfg['scene_noise_records'][self.split])
            self.engines=Engines(self.cfg['ffmpeg'])
        chosen=int(t['pcl_noisy_index']);main=[];clean=None;selected_meta=None
        for view in (0,1):
            noise_recipe=recipe(self.cfg['seed'],t['occurrence'],row['group_id'],view,t['epoch'],self.cfg['warm_epochs'],
                self.split,stress_probability=self.cfg['rtc_stress_probability'],
                stress_ramp_epochs=self.cfg['rtc_stress_ramp_epochs'],
                scene_probability=self.cfg['scene_probability'],rtc_policy=self.cfg['rtc_policy'])
            try:
                if view==chosen:
                    clean,noisy,meta=generate_matched(base_wave,noise_recipe,self.bank,self.engines)
                    selected_meta=meta
                else:noisy,meta=generate(base_wave,noise_recipe,self.bank,self.engines)
            except Exception as exc:raise RuntimeError('V3.18.6 Noisy/reference failed: '+base['view_id']) from exc
            role=('noisy_a','noisy_b')[view]
            main.append(dict(base,role=role,condition=role,wave=ready(noisy),augmentation=meta))
        assert clean is not None
        # CTC timestamps belong to the real source waveform before ready()'s
        # exceptional short-clip padding. Never apply PCL to repeated content.
        refs={'clean':dict(base,role='clean_ref',condition='clean_ref',wave=ready(clean),augmentation=selected_meta)}
        pairs={};coverage={};timing={}
        from .phonemes import map_segments
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
            pairs['noise']=noise_pairs;coverage['noise_valid']=len(noise_pairs)
        if 'pcl_online_index' in t:
            official=self.rows[t['pcl_online_index']]
            if any(official.get(k)!=row.get(k) for k in ('source_id','group_id','language','label')):
                raise ValueError('Official auxiliary source identity disagrees')
            ow=self.original(official)
            refs['official']=dict(official,occurrence=t['occurrence'],role='official_ref',wave=ready(ow),
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
            pairs['platform']=platform;coverage['platform_valid']=len(platform)
        main[0].update(pcl_refs=refs,pcl_pairs=pairs,pcl_noisy_index=chosen,
            pcl_coverage=coverage,pcl_timing=timing)
        return main


def microbatches(units,size,frame_budget):
    """Budget auxiliary encoder waves as well as classification waves."""
    batch=[];count=0;longest=0
    for unit in units:
        refs=unit[0].get('pcl_refs',{});pairs=unit[0].get('pcl_pairs',{})
        needed=[]
        if pairs.get('noise') or pairs.get('platform'):needed.append('clean')
        if pairs.get('platform'):needed.append('official')
        extra=[refs[key] for key in needed]
        rows=list(unit)+extra
        n=max(max(1,(len(r['wave'])-400)//320+1) for r in rows)
        if len(rows)>size or len(rows)*n>frame_budget:
            raise ValueError('A complete source/PCL unit exceeds microbatch/frame budget; '
                'increase --microbatch/--frame-budget rather than truncating audio')
        total=count+len(rows);length=max(longest,n)
        if batch and (total>size or total*length>frame_budget):
            yield batch;batch=[];count=0;longest=0
        batch.append(unit);count+=len(rows);longest=max(longest,n)
    if batch:yield batch
