"""Fail before training if either real communication implementation is unusable."""
import json

import numpy as np

from .augment import runtime, Engines, settings_grid


def main():
    measured=runtime()
    engines=Engines(measured['ffmpeg_path'])
    n=16731  # Includes a non-10-ms tail; exercises Opus pre-skip/end trimming.
    rng=np.random.default_rng(315)
    wave=(.1*np.sin(np.arange(n)*2*np.pi*237/16000)+rng.normal(0,.003,n)).astype(np.float32)
    checks={}
    for family in ('ffmpeg','webrtc','light'):
        r=dict(family=family,settings=settings_grid(family,'train')[0])
        a,b=engines(wave,r),engines(wave.copy(),r)
        if a.shape!=wave.shape or not np.isfinite(a).all() or not np.array_equal(a,b):
            raise RuntimeError(f'{family}: repeated fresh full-utterance processing differs or changes duration')
        checks[family]=dict(samples=len(a),fresh_state_repeat_exact=True)
    print('V315_AUGMENTATION_PREFLIGHT='+json.dumps(dict(runtime=measured,checks=checks)),flush=True)


if __name__=='__main__':main()
