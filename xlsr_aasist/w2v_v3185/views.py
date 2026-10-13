"""Shared label-blind transformations; no disk waveform or feature cache.

Speed uses small rational resampling (speed and pitch change together), not
TTS, voice conversion, or pitch-preserving time stretching. Preserve the
original label and source identity; apply an Offline view BEFORE noise/RTC.
"""
import math
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from .common import seed_for

FAMILIES=('original','speed_slow','speed_fast','frequency_response','temporal_context')
SR=16000


def recipe(seed, source, kind, variant, *, family=None, generation=0, dynamic=False):
    if kind not in ('online','offline') or not 0<=variant<5:
        raise ValueError('Expected Online/Offline and one of five views')
    if family is None:family=FAMILIES[variant]
    if family not in FAMILIES or type(generation) is not int or generation<0:
        raise ValueError('Unknown family or invalid generation')
    if type(dynamic) is not bool:raise ValueError('dynamic must be boolean')
    if family=='original' and (dynamic or generation):raise ValueError('Original views cannot be refreshed or dynamic')
    family_index=FAMILIES.index(family)
    # Keep fixed English generation-zero waveforms exactly comparable to AUG5.
    key=seed_for(seed,'aug5-waveform-v1',source,kind,family_index)
    if dynamic or generation:
        key=seed_for(seed,'v3185-waveform-v1',source,kind,variant,family,generation,dynamic)
    rng=np.random.default_rng(key)
    return dict(version=2,seed=key,variant=variant,family=family,generation=generation,dynamic=dynamic,
        rate=int(rng.integers(970,991)) if family_index==1 else int(rng.integers(1010,1031)),
        low_hz=float(rng.uniform(180,600)),high_hz=float(rng.uniform(2500,5000)),
        low_db=float(rng.choice((-1,1))*rng.uniform(1,3)),
        high_db=float(rng.choice((-1,1))*rng.uniform(1,3)),
        keep=float(rng.uniform(.85,.95)),start=float(rng.random()),
        envelope_db=float(rng.uniform(1,3)),phase=float(rng.uniform(0,2*np.pi)))


def augment(wave, r):
    x=np.asarray(wave,np.float32)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid augmentation source')
    # A ZH-real fixed slot may map to any family. ZH-fake dynamic transforms
    # have slot zero but are not original. Dispatch by family, never slot ID.
    v=FAMILIES.index(r['family']);fallback=None
    if v==0:
        y=x.copy()
    elif v in (1,2):
        # Exactly specified rational factors avoid accumulating timing errors.
        rate=int(r['rate']);g=math.gcd(1000,rate)
        y=resample_poly(x,1000//g,rate//g).astype(np.float32)
    elif v==3:
        low=sosfilt(butter(2,r['low_hz'],fs=SR,output='sos'),x)
        high=sosfilt(butter(2,r['high_hz'],btype='highpass',fs=SR,output='sos'),x)
        y=(x+(10**(r['low_db']/20)-1)*low+(10**(r['high_db']/20)-1)*high).astype(np.float32)
    elif v==4:
        keep=max(2*SR,int(round(len(x)*r['keep'])))
        if keep < len(x):
            start=min(len(x)-keep,int(r['start']*(len(x)-keep+1)))
            y=x[start:start+keep].copy()
            # Avoid retaining a nearly silent tail while dropping the speech.
            if float(np.dot(y.astype(np.float64),y)) < .65*float(np.dot(x.astype(np.float64),x)):
                fallback='energy_preserving_envelope'
        else:fallback='short_recording_envelope'
        if fallback:
            phase=np.linspace(0,2*np.pi,len(x),dtype=np.float64)+r['phase']
            y=(x*np.power(10.,r['envelope_db']*np.sin(phase)/20)).astype(np.float32)
    else:raise ValueError('Unknown view')
    if v:
        peak=float(np.abs(y).max())
        if peak>.98:y=y*(.98/peak)
    if not len(y) or not np.isfinite(y).all():raise ValueError('Augmentation produced invalid audio')
    return y,dict(recipe=r,input_samples=len(x),output_samples=len(y),fallback=fallback,
        source_identity_preserved=True,independent_new_recording=False)
