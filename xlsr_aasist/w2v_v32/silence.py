"""Rare waveform-local zeroing, bounded in every supervised waveform view."""
import math
import numpy as np

RATE = 16000
MIN_SAMPLES = round(.04*RATE)


def apply_local_silence(wave, parameters, protected_views=()):
    """Return an independent waveform and the actual applied/skip audit record.

    Duration includes both ramps. The same altered waveform supplies the full
    and cropped views. A conservative shortest-view cap guarantees at most 5%
    affected samples even when the entire gap falls inside a crop. An energy
    guard rejects removal of the only useful sound in an otherwise quiet view.
    This does not infer speech activity, truth labels, or codec behavior.
    """
    if np.ndim(wave)!=1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Silence augmentation requires finite nonempty mono audio')
    duration=float(parameters['duration_seconds'])
    position=float(parameters['position'])
    fraction=float(parameters.get('maximum_fraction',.05))
    fade=float(parameters.get('fade_seconds',.005))
    if not all(math.isfinite(v) for v in (duration,position,fraction,fade)):
        raise ValueError('Silence parameters must be finite')
    if not .04<=duration<=.16 or not 0<=position<=1 or not 0<fraction<=.05 or fade!=.005:
        raise ValueError('Silence requires 40-160 ms, fraction <=5%, position [0,1], 5 ms ramps')
    views=[(0,len(wave))]
    for start,count in protected_views:
        if (isinstance(start,bool) or isinstance(count,bool) or
                not isinstance(start,(int,np.integer)) or not isinstance(count,(int,np.integer)) or
                start<0 or count<1 or start+count>len(wave)):
            raise ValueError('Protected views must be nonempty slices of the waveform')
        views.append((int(start),int(count)))
    result=np.array(wave,dtype=np.float32,copy=True)
    metadata=dict(silence_applied=False,silence_skip_reason='',silence_start_sample=None,
                  silence_samples=0,silence_zero_samples=0,
                  silence_requested_seconds=duration,silence_view_fractions=[],
                  silence_remaining_energy_fractions=[])
    length=min(round(duration*RATE),min(math.floor(count*fraction) for _,count in views))
    if length<MIN_SAMPLES:
        metadata['silence_skip_reason']='view_too_short_for_40ms_at_5percent'
        return result,metadata
    start=min(len(wave)-length,int(position*(len(wave)-length+1)))
    ramp=round(fade*RATE)
    envelope=np.zeros(length,dtype=np.float32)
    edge=(.5+.5*np.cos(np.linspace(0,np.pi,ramp))).astype(np.float32)
    envelope[:ramp]=edge
    envelope[-ramp:]=edge[::-1]
    end=start+length
    result[start:end]*=envelope
    fractions=[];energies=[]
    for view_start,count in views:
        view_end=view_start+count
        overlap=max(0,min(end,view_end)-max(start,view_start))
        fractions.append(overlap/count)
        # Unaffected views need no extra audio scan.
        if not overlap:
            energies.append(1.);continue
        before=np.asarray(wave[view_start:view_end],dtype=np.float64)
        after=np.asarray(result[view_start:view_end],dtype=np.float64)
        energy=float(np.dot(before,before))
        kept=float(np.dot(after,after))
        ratio=kept/energy if energy>0 else 1.
        energies.append(ratio)
        if energy<=0 or ratio<.9:
            metadata['silence_skip_reason']='insufficient_remaining_view_energy'
            metadata['silence_remaining_energy_fractions']=energies
            return np.array(wave,dtype=np.float32,copy=True),metadata
    if not np.any(np.asarray(wave[start:end])!=result[start:end]):
        metadata['silence_skip_reason']='selected_region_already_silent'
        return result,metadata
    metadata.update(silence_applied=True,silence_start_sample=start,
                    silence_samples=length,silence_zero_samples=int((envelope==0).sum()),
                    silence_view_fractions=fractions,
                    silence_remaining_energy_fractions=energies)
    return np.ascontiguousarray(result),metadata
