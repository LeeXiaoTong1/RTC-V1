"""Two distinct mechanisms per source; natural continuous noise, no waveform cache."""
import numpy as np
from scipy.signal import butter,sosfilt
from w2v_v315.augment import NoiseBank, common_inputs
from rtc_noisy.simulator import LocalRTC
from .common import seed_for

TRAIN_FAMILIES=('ffmpeg','webrtc','light','bypass')
PANEL_FAMILIES=('g711_mulaw','anlmdn')


def recipe(seed,occurrence,source,view,epoch,warm_epochs=2,split='train',family=None,probe=False):
    base=np.random.default_rng(seed_for(seed,occurrence,source,'families'))
    families=base.choice(TRAIN_FAMILIES,2,replace=False)
    rng=np.random.default_rng(seed_for(seed,occurrence,source,view,'recipe'))
    lower=10. if epoch<warm_epochs or probe else 5.
    return dict(seed=seed_for(seed,occurrence,source,view,'noise'),split=split,
        family=family or str(families[view]),noise_type=str(rng.choice(('continuous','events','mixed'),p=(.6,.2,.2))),
        snr_db=float(rng.uniform(lower,25.)),input_gain_db=float(rng.uniform(-3,3)),
        codec=str(rng.choice(('none','opus'))),bitrate=int(rng.choice((16000,24000,32000))),
        cutoff=int(rng.choice((4800,6000,7200))),nr=int(rng.choice((6,9,12))),
        gain=int(rng.choice((2,4))),ns=int(rng.choice((0,1,2))),target=int(rng.choice((9,12,15))))


def noise_wave(bank,length,kind,rng):
    hashes=set()
    def continuous():
        out=np.zeros(length,np.float32);cursor=0
        while cursor<length:
            x,key=bank.sample(rng);hashes.add(key)
            if not len(x): raise ValueError('Empty noise recording')
            fade=min(cursor,320)
            count=min(len(x),length-cursor+fade)
            start=int(rng.integers(len(x)-count+1));piece=x[start:start+count].copy()
            fade=min(fade,count//4)
            if fade:
                w=np.linspace(0,1,fade,dtype=np.float32)
                out[cursor-fade:cursor]=out[cursor-fade:cursor]*(1-w)+piece[:fade]*w
            n=min(count-fade,length-cursor);out[cursor:cursor+n]=piece[fade:fade+n];cursor+=n
        return out
    def events():
        out=np.zeros(length,np.float32)
        for _ in range(int(rng.integers(1,5))):
            x,key=bank.sample(rng);hashes.add(key);n=min(len(x),length,int(rng.integers(800,24001)))
            a=int(rng.integers(len(x)-n+1));b=int(rng.integers(length-n+1));piece=x[a:a+n].copy()
            fade=min(160,n//4)
            if fade: piece[:fade]*=np.linspace(0,1,fade);piece[-fade:]*=np.linspace(1,0,fade)
            out[b:b+n]+=piece
        return out
    if kind=='continuous':out=continuous()
    elif kind=='events':out=events()
    elif kind=='mixed':
        a,b=continuous(),events();out=a+.5*b*max(np.sqrt(np.mean(a*a)),1e-6)/max(np.sqrt(np.mean(b*b)),1e-6)
    else: raise ValueError('Unknown noise type')
    return out,sorted(hashes)


class Engines:
    def __init__(self,ffmpeg='ffmpeg'):
        self.rtc=LocalRTC(ffmpeg)
    def __call__(self,x,r):
        family=r['family'];y=np.asarray(x,np.float32)
        raw_args=['-f','f32le','-ar','16000','-ac','1','-i','pipe:0']
        if family=='ffmpeg':
            filters=f'afftdn=nr={r["nr"]}:nf=-35:tn=1,dynaudnorm=f=150:g=7:p=0.9:m={r["gain"]}'
            y=np.frombuffer(self.rtc._run([*raw_args,'-af',filters,'-f','f32le','pipe:1'],y.astype('<f4').tobytes()),dtype='<f4').copy()
        elif family=='webrtc':
            from webrtc_audio_processing import AudioProcessingModule
            ap=AudioProcessingModule(enable_ns=True,agc_type=1,enable_vad=False)
            ap.set_stream_format(16000,1);ap.set_ns_level(r['ns']);ap.set_agc_target(r['target'])
            pcm=np.round(np.clip(y,-.999,.999)*32767).astype('<i2');blocks=[]
            for a in range(0,len(pcm),160):
                frame=pcm[a:a+160];out=np.frombuffer(ap.process_stream(np.pad(frame,(0,160-len(frame))).tobytes()),dtype='<i2')
                if len(out)!=160: raise RuntimeError('Invalid WebRTC frame')
                blocks.append(out[:len(frame)].astype(np.float32)/32768.)
            y=np.concatenate(blocks)
        elif family=='light':y=sosfilt(butter(4,r['cutoff'],fs=16000,output='sos'),y).astype(np.float32)
        elif family=='anlmdn':
            y=np.frombuffer(self.rtc._run([*raw_args,'-af','anlmdn=s=0.0001:p=0.002:r=0.006','-f','f32le','pipe:1'],y.astype('<f4').tobytes()),dtype='<f4').copy()
        elif family=='g711_mulaw':
            raw=self.rtc._run([*raw_args,'-ar','8000','-c:a','pcm_mulaw','-f','mulaw','pipe:1'],y.astype('<f4').tobytes())
            y=np.frombuffer(self.rtc._run(['-f','mulaw','-ar','8000','-ac','1','-i','pipe:0','-ar','16000','-f','f32le','pipe:1'],raw),dtype='<f4').copy()
            if abs(len(y)-len(x))>2: raise RuntimeError('G.711 duration mismatch')
            y=np.pad(y,(0,max(0,len(x)-len(y))))[:len(x)]
        elif family!='bypass':raise ValueError('Unknown processing family')
        if r['codec']=='opus' and family not in PANEL_FAMILIES:
            raw=self.rtc._run([*raw_args,'-c:a','libopus','-b:a',str(r['bitrate']),'-application','voip','-frame_duration','20','-vbr','on','-f','ogg','pipe:1'],y.astype('<f4').tobytes())
            y=np.frombuffer(self.rtc._run(['-f','ogg','-i','pipe:0','-ar','16000','-ac','1','-f','f32le','pipe:1'],raw),dtype='<f4').copy()
        if len(y)!=len(x) or not np.isfinite(y).all():raise RuntimeError('Processing changed full duration or returned nonfinite samples')
        return y


def generate(wave,r,bank,engines):
    rng=np.random.default_rng(r['seed'])
    for attempt in range(8):
        noise,hashes=noise_wave(bank,len(wave),r['noise_type'],rng)
        _,mixed,meta=common_inputs(wave,noise,r['snr_db'],r['input_gain_db'])
        if meta['pair_eligible']:break
    y=engines(mixed,r)
    return y,dict(recipe=r,noise_hashes=hashes,noise_attempts=attempt+1,**meta)
