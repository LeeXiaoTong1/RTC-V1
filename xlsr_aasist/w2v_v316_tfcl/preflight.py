"""Check real codec implementations before paying for model loading/training."""
import json
import numpy as np
from w2v_v315.augment import runtime,settings_grid
from .data import Engines


def main():
    measured=runtime();engines=Engines(measured['ffmpeg_path'])
    n=16731;rng=np.random.default_rng(316)
    wave=(.1*np.sin(np.arange(n)*2*np.pi*237/16000)+rng.normal(0,.003,n)).astype(np.float32)
    checks={}
    for family in ('ffmpeg','webrtc','light','g711_mulaw','g711_alaw'):
        r=dict(family=family,settings={} if family.startswith('g711') else settings_grid(family,'train')[0])
        a,b=engines(wave,r),engines(wave.copy(),r)
        if a.shape!=wave.shape or not np.isfinite(a).all() or not np.array_equal(a,b):
            raise RuntimeError(f'{family}: nondeterministic fresh processing or wrong duration')
        checks[family]=dict(samples=len(a),fresh_state_repeat_exact=True,training=not family.startswith('g711'))
    print('V316_TFCL_AUGMENTATION_PREFLIGHT='+json.dumps(dict(runtime=measured,checks=checks)),flush=True)


if __name__=='__main__':main()
