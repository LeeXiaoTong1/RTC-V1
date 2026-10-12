"""Legacy noisy A; controlled RTC input/codec diversity on a fraction of noisy B."""
import numpy as np
from scipy.signal import butter, sosfilt
from w2v_v318.augment import (NoiseBank, TRAIN_FAMILIES, PANEL_FAMILIES, generate,
    augmentation_runtime, recipe as legacy_recipe, Engines as LegacyEngines)
from .common import seed_for


def recipe(seed,occurrence,source,view,epoch,warm_epochs=0,split='train',family=None,probe=False,
           stress_probability=.35,stress_ramp_epochs=2.):
    r=legacy_recipe(seed,occurrence,source,view,epoch,warm_epochs,split,family,probe)
    if view!=1 or split!='train' or probe or family or stress_probability==0:return r
    probability=stress_probability*min(1.,(epoch+1)/stress_ramp_epochs)
    rng=np.random.default_rng(seed_for(seed,occurrence,source,'v3183-rtc'))
    if rng.random()>=probability:return r
    mechanism=str(rng.choice(('gain_envelope','bandwidth','soft_limit','codec')))
    r['stress']=dict(mechanism=mechanism,seed=seed_for(seed,occurrence,source,'rtc-envelope'),probability=probability)
    if mechanism=='gain_envelope':r['stress'].update(span_db=float(rng.uniform(2,6)),period=float(rng.uniform(.5,2.)))
    elif mechanism=='bandwidth':r['stress'].update(highpass=int(rng.choice((60,100,180))),lowpass=int(rng.choice((3400,4200,5600))))
    elif mechanism=='soft_limit':r['stress'].update(threshold=float(rng.uniform(.35,.7)))
    else:r.update(codec='opus',bitrate=int(rng.choice((12000,16000,24000,32000,48000))))
    return r


class Engines(LegacyEngines):
    def __call__(self,x,r):
        s=r.get('stress')
        if s:
            y=np.asarray(x,np.float32);kind=s['mechanism']
            # generate() mixes noise FIRST. These transforms act on speech and
            # noise together, before NS/AGC and the existing Opus pipeline.
            if kind=='gain_envelope':
                rng=np.random.default_rng(s['seed']);stride=max(1,int(16000*s['period']))
                knots=np.arange(0,len(y)+stride,stride)
                levels=rng.uniform(-s['span_db'],s['span_db'],len(knots))
                envelope=np.interp(np.arange(len(y)),knots,levels)
                y=y*np.power(10.,envelope/20.)
            elif kind=='bandwidth':
                y=sosfilt(butter(3,[s['highpass'],s['lowpass']],btype='bandpass',fs=16000,output='sos'),y)
            elif kind=='soft_limit':
                threshold=s['threshold'];y=threshold*np.tanh(y/threshold)
            elif kind!='codec':raise ValueError('Unknown RTC stress mechanism')
            x=np.asarray(y,np.float32)
        return super().__call__(x,r)
