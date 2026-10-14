"""Reuse the exact V3.18.5 Noisy endpoint; process one matched clean reference.

Noise draws, common gain, base augmentation, RTC recipe and classification
waveforms remain unchanged. Engines reset WebRTC and codec state per call.
An echo scene is the noisy perturbation; it is absent from the clean arm.
"""
import copy
import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly
from w2v_v3184.augment import NoiseBank, Engines, recipe, generate


def generate_matched(wave, rec, bank, engines):
    noisy, metadata = generate(wave, rec, bank, engines)
    clean_input = (np.asarray(wave, np.float32)*metadata['common_gain']).astype(np.float32)
    clean = engines(clean_input, rec)
    if clean.shape != noisy.shape or not np.isfinite(clean).all():
        raise RuntimeError('Matched RTC reference changed length or is nonfinite')
    metadata = dict(metadata, same_settings=True, shared_base=True,
        fresh_state_per_utterance=True, clean_has_scene_echo=False,
        erased_spans=[], recipe=copy.deepcopy(rec))
    return clean, noisy, metadata


def estimate_delay(reference, received, max_shift=4000, minimum_score=.12):
    """Measure constant processing delay, rather than assuming equal length.

    Correlation uses anti-aliased 4 kHz waveforms and a bounded +/-250 ms
    search. Positive lag means received content appears later. Very weak or
    ambiguous matches are invalid and do not silently become identity maps.
    The remaining local reliability check is separate below.
    """
    a = np.asarray(reference, np.float32)
    b = np.asarray(received, np.float32)
    if len(a)<400 or len(b)<400 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(valid=False, offset=0., score=0., reason='invalid_or_short')
    a = resample_poly(a,1,4).astype(np.float64)
    b = resample_poly(b,1,4).astype(np.float64)
    a -= a.mean(); b -= b.mean()
    norm = float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
    if norm<1e-12:return dict(valid=False,offset=0.,score=0.,reason='silent')
    corr=correlate(b,a,mode='full',method='fft')/norm
    lags=correlation_lags(len(b),len(a),mode='full')
    mask=np.abs(lags)<=max_shift//4
    values=corr[mask]; lag=lags[mask]; at=int(np.argmax(values))
    score=float(values[at]); best=int(lag[at])
    # Nearby lobes of the same waveform are not competing alignments. Widely
    # separated equally likely peaks (periodicity/repetition) are ambiguous.
    far=np.abs(lag-best)>80
    second=float(np.max(values[far])) if np.any(far) else -1.
    valid=score>=minimum_score and score-second>=.01 and abs(best)<max_shift//4
    return dict(valid=bool(valid),offset=float(best*4),score=score,
        alternative_score=second,reason='ok' if valid else 'weak_or_ambiguous')


def reliable_spans(reference, received, spans, reference_offset=0., received_offset=0.):
    """Reject local deletions/silence and unrelated intervals after alignment.

    This is a signal-validity mask, never a detector-confidence gate. Full
    frame/phoneme erasure cannot be inferred from equal output lengths.
    """
    a=np.asarray(reference,np.float32);b=np.asarray(received,np.float32)
    keep=[]
    ea=float(np.mean(a.astype(np.float64)**2)); eb=float(np.mean(b.astype(np.float64)**2))
    global_ratio=max(eb,1e-12)/max(ea,1e-12)
    # Downsample each full waveform once, not once per phone. Using full
    # context also avoids artificial resampling boundaries at every phone.
    low_a=resample_poly(a,1,4).astype(np.float64)
    low_b=resample_poly(b,1,4).astype(np.float64)
    for start,end in spans:
        aa=int(round(start+reference_offset));bb=int(round(end+reference_offset))
        ca=int(round(start+received_offset));cb=int(round(end+received_offset))
        if aa<0 or ca<0 or bb>len(a) or cb>len(b) or min(bb-aa,cb-ca)<80:
            keep.append(False);continue
        n=min(bb-aa,cb-ca);x=a[aa:aa+n].astype(np.float64);y=b[ca:ca+n].astype(np.float64)
        xx=float(np.dot(x,x)); yy=float(np.dot(y,y))
        if xx<=1e-10 or yy<=1e-10 or yy/xx < .01*global_ratio:
            keep.append(False);continue
        # A small search tolerates sub-frame phase/group-delay changes. A
        # single sample-perfect correlation would reject filtered speech.
        if n>128:
            u=low_a[(aa+3)//4:(aa+n)//4]
            v=low_b[(ca+3)//4:(ca+n)//4]
            size=min(len(u),len(v));u=u[:size];v=v[:size]
            # Only the requested +/-20 low-rate lags are computed. A full
            # FFT correlation per phone spends work on lags we discard.
            cc=np.correlate(np.pad(v,(20,20)),u,mode='valid')
            score=float(np.max(cc))/max(float(np.sqrt(np.dot(u,u)*np.dot(v,v))),1e-12)
        else:score=float(np.dot(x,y))/max(np.sqrt(xx*yy),1e-12)
        keep.append(bool(score>.05))
    return keep
