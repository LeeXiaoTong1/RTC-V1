"""V3.18.5 waveforms, with exact base-augmentation sample coordinates."""
from w2v_v3185.views import FAMILIES, SR, recipe, augment as base_augment


def augment(wave, rec):
    output, metadata = base_augment(wave, rec)
    scale, offset = 1., 0.
    if rec['family'] in ('speed_slow', 'speed_fast'):
        scale = 1000. / int(rec['rate'])
    elif rec['family'] == 'temporal_context' and not metadata['fallback']:
        keep = max(2 * SR, int(round(len(wave) * rec['keep'])))
        start = min(len(wave)-keep, int(rec['start']*(len(wave)-keep+1)))
        offset = -float(start)
    metadata['time_map'] = dict(scale=scale, offset=offset,
        input_samples=len(wave), output_samples=len(output))
    return output, metadata
