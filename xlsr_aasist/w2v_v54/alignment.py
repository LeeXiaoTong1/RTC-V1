"""The inherited delay/reliability rules, sharing one 4 kHz conversion per pair."""
import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly


def align_pair(reference, received, spans, max_shift=4000, minimum_score=.12):
    a, b = np.asarray(reference, np.float32), np.asarray(received, np.float32)
    reject = lambda reason: (dict(valid=False, offset=0., score=0., reason=reason), [False]*len(spans))
    if min(len(a), len(b)) < 400 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return reject('invalid_or_short')
    low_a, low_b = (resample_poly(x, 1, 4).astype(np.float64) for x in (a, b))
    x, y = low_a-low_a.mean(), low_b-low_b.mean()
    norm = float(np.sqrt(np.dot(x,x)*np.dot(y,y)))
    if norm < 1e-12:
        return reject('silent')
    corr = correlate(y,x,mode='full',method='fft')/norm
    lags = correlation_lags(len(y),len(x),mode='full')
    mask = np.abs(lags) <= max_shift//4
    values, lag = corr[mask], lags[mask]
    at = int(np.argmax(values)); score, best = float(values[at]), int(lag[at])
    far = np.abs(lag-best) > 80
    second = float(np.max(values[far])) if np.any(far) else -1.
    valid = score >= minimum_score and score-second >= .01 and abs(best) < max_shift//4
    delay = dict(valid=bool(valid), offset=float(best*4), score=score,
                 alternative_score=second, reason='ok' if valid else 'weak_or_ambiguous')
    if not valid:
        return delay, [False]*len(spans)
    global_ratio = max(float(np.mean(b.astype(np.float64)**2)),1e-12)/max(float(np.mean(a.astype(np.float64)**2)),1e-12)
    keep = []
    for start,end in spans:
        aa,bb = int(round(start)),int(round(end))
        ca,cb = int(round(start+delay['offset'])),int(round(end+delay['offset']))
        if aa < 0 or ca < 0 or bb > len(a) or cb > len(b) or min(bb-aa,cb-ca) < 80:
            keep.append(False); continue
        n = min(bb-aa,cb-ca)
        x,y = a[aa:aa+n].astype(np.float64),b[ca:ca+n].astype(np.float64)
        xx,yy = float(np.dot(x,x)),float(np.dot(y,y))
        if xx <= 1e-10 or yy <= 1e-10 or yy/xx < .01*global_ratio:
            keep.append(False); continue
        if n > 128:
            u,v = low_a[(aa+3)//4:(aa+n)//4],low_b[(ca+3)//4:(ca+n)//4]
            size = min(len(u),len(v)); u,v = u[:size],v[:size]
            cc = np.correlate(np.pad(v,(20,20)),u,mode='valid')
            score = float(np.max(cc))/max(float(np.sqrt(np.dot(u,u)*np.dot(v,v))),1e-12)
        else:
            score = float(np.dot(x,y))/max(np.sqrt(xx*yy),1e-12)
        keep.append(bool(score > .05))
    return delay,keep
