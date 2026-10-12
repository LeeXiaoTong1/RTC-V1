"""Transform each source once; share its transformed Offline base across the Noisy pair."""
from w2v_v3184.data import (Waves as BaseWaves,ready,loader,close,microbatches,TicketSampler)
from w2v_v3184.augment import NoiseBank,Engines,recipe,generate
from .views import recipe as view_recipe,augment


class Waves(BaseWaves):
    def __getitem__(self,t):
        # Fixed Dev and old Train probes have no virtual-view ticket. Do not
        # change their input distribution or consume the training view schedule.
        if isinstance(t,int) or 'base_variant' not in t:return super().__getitem__(t)
        if self.split!='train' or t.get('probe') or t.get('scene') or t.get('family'):
            raise ValueError('Base views are only for scheduled training tickets')
        r=self.rows[t['index']]
        rec=view_recipe(self.cfg['seed'],r['id'],t['kind'],t['base_variant'])
        x,base_meta=augment(self.original(r),rec)
        base=dict(r,occurrence=t['occurrence'],base_augmentation=base_meta,
                  view_id=f'{r["id"]}#aug5-{t["base_variant"]}')
        if t['kind']=='online':return [dict(base,role='online',wave=ready(x))]
        if self.bank is None:
            self.bank=NoiseBank(self.cfg['noise_records'][self.split],self.cfg['noise_cache_mib']*1024**2,
                                self.cfg['scene_noise_records'][self.split])
            self.engines=Engines(self.cfg['ffmpeg'])
        result=[]
        for v in (0,1):
            rcp=recipe(self.cfg['seed'],t['occurrence'],r['group_id'],v,t['epoch'],self.cfg['warm_epochs'],
                self.split,stress_probability=self.cfg['rtc_stress_probability'],
                stress_ramp_epochs=self.cfg['rtc_stress_ramp_epochs'],
                scene_probability=self.cfg['scene_probability'],rtc_policy=self.cfg['rtc_policy'])
            try:wave,meta=generate(x,rcp,self.bank,self.engines)
            except Exception as exc:raise RuntimeError('AUG5 Noisy failed: '+base['view_id']) from exc
            result.append(dict(base,role=('noisy_a','noisy_b')[v],condition=('noisy_a','noisy_b')[v],
                               wave=ready(wave),augmentation=meta))
        return result
