"""Six target scenes in both noisy views; preserve old probes and old RTC by default."""
import numpy as np
from w2v_v318.augment import (NoiseBank as LegacyBank, TRAIN_FAMILIES, PANEL_FAMILIES,
    generate as legacy_generate, augmentation_runtime)
from w2v_v3183r2.augment import recipe as prior_recipe, Engines
from w2v_v315.augment import common_inputs
from .common import seed_for
from .noise import CATEGORIES, SceneBank, scene_wave


class NoiseBank:
    def __init__(self,records,cap_bytes,scene_records=None):
        self.legacy=LegacyBank(records,cap_bytes)
        self.scenes=SceneBank(scene_records,min(cap_bytes,32*1024**2)) if scene_records else None

    def sample(self,rng):return self.legacy.sample(rng)


def recipe(seed,occurrence,source,view,epoch,warm_epochs=1,split='train',family=None,probe=False,
           stress_probability=.35,stress_ramp_epochs=2.,scene_probability=.8,rtc_policy='legacy',scene=None):
    r=prior_recipe(seed,occurrence,source,view,epoch,warm_epochs,split,family,probe,
                   stress_probability,stress_ramp_epochs)
    # Old fixed probes/panels remain bitwise reproducible. Scene tickets are explicit.
    eligible=split=='train' and not probe and family is None
    if scene is not None:
        if scene not in CATEGORIES:raise ValueError('Unknown target scene')
        r['category']=scene
    elif eligible:
        pair_rng=np.random.default_rng(seed_for(seed,occurrence,source,'v3184-scenes'))
        categories=pair_rng.choice(CATEGORIES,2,replace=False)
        if pair_rng.random(2)[view]<scene_probability:r['category']=str(categories[view])
    if r.get('category'):
        r['noise_type']='scene:'+r['category']
        rng=np.random.default_rng(seed_for(seed,occurrence,source,view,'v3184-strength'))
        if r['category']=='echo':
            r['echo']=dict(delay_samples=int(rng.integers(640,2401)),gain=float(rng.uniform(.1,.5)))
    if rtc_policy not in ('legacy','three-profile'):raise ValueError('Unknown RTC policy')
    if rtc_policy=='three-profile' and (eligible or scene is not None):
        # Independent marginal profile probabilities for A and B. Sampling without
        # replacement would distort the second view's requested 50/30/20 mixture.
        rng=np.random.default_rng(seed_for(seed,occurrence,source,view,'v3184-rtc'))
        r['family']=str(rng.choice(('webrtc','ffmpeg','light'),p=(.5,.3,.2)))
        r['codec']='opus';r['bitrate']=int(rng.choice((16000,24000,32000)));r.pop('stress',None)
    r['rtc_policy']=rtc_policy if eligible or scene is not None else 'legacy-fixed-probe'
    return r


def generate(wave,r,bank,engines):
    category=r.get('category')
    if category is None:return legacy_generate(wave,r,bank,engines)
    x=np.asarray(wave,np.float32)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid full-wave scene input')
    if category=='echo':
        delay=min(r['echo']['delay_samples'],len(x)-1)
        mixed=x.copy()
        # Use the ORIGINAL source on the right; never an overlapping in-place view.
        mixed[delay:]+=r['echo']['gain']*x[:-delay]
        gain=min(10**(r['input_gain_db']/20),.97/max(float(np.abs(mixed).max()),float(np.abs(x).max()),1e-12))
        mixed=(mixed*gain).astype(np.float32)
        meta=dict(common_gain=gain,noise_scale=0.,pair_eligible=True,measured_input_snr_db=None,
            snr_scope='echo uses delayed-speech gain, not additive-noise SNR',
            echo_delay_samples=delay,echo_gain=r['echo']['gain'])
        hashes=[];sources=[];attempt=0
    else:
        if bank.scenes is None:raise ValueError('Target scene bank is not configured')
        rng=np.random.default_rng(r['seed'])
        for attempt in range(8):
            noise,hashes,sources=scene_wave(bank.scenes,category,len(x),rng)
            _,mixed,meta=common_inputs(x,noise,r['snr_db'],r['input_gain_db'])
            if meta['pair_eligible']:break
        if not meta['pair_eligible']:
            # Do not silently count an unaugmented sample as a target scene.
            raise RuntimeError('No non-silent speech/noise overlap after 8 scene attempts: '+category)
    y=engines(mixed,r)
    if y.shape!=x.shape or not np.isfinite(y).all():raise RuntimeError('Scene RTC changed length or produced nonfinite audio')
    return y,dict(recipe=r,noise_hashes=hashes,noise_original_sources=sources,
                  noise_attempts=attempt+1,**meta)
