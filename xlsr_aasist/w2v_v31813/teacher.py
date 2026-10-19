"""Pinned, frozen XEUS block 10; used only during local Train preparation."""
from pathlib import Path
import numpy as np
import torch
from w2v_v31810.common import digest, atomic_json, read_json
from .espnet_runtime import ESPNET_REVISION,load_ssl_task

MODEL_ID='espnet/xeus'
REVISION='b651261280bf792df4632ccb2d1ba4eeaabeaee2'
FILES=('model/config.yaml','model/xeus_checkpoint_new.pth')


def assets(cache, endpoint=None, local=False):
    from huggingface_hub import hf_hub_download
    paths=[]
    for name in FILES:
        endpoints=[endpoint] if endpoint or local else [None,'https://hf-mirror.com']
        for index,url in enumerate(endpoints):
            try:
                path=hf_hub_download(MODEL_ID,name,revision=REVISION,cache_dir=str(cache),endpoint=url,local_files_only=local)
                paths.append(Path(path));break
            except Exception:
                if index+1==len(endpoints):raise
                print('Official XEUS download unavailable; retry the same pinned revision via hf-mirror.com',flush=True)
    proof=dict(model=MODEL_ID,revision=REVISION,files={name:digest(path) for name,path in zip(FILES,paths)},
               espnet_revision=ESPNET_REVISION,layer=10,dimension=1024,precision='fp32',sampling_rate=16000)
    receipt=Path(cache)/'v31813_teacher.json'
    if receipt.exists() and read_json(receipt)!=proof:raise ValueError('Pinned XEUS assets changed')
    atomic_json(receipt,proof)
    return paths,proof


class _ReachedLayer(Exception):
    def __init__(self,value):self.value=value


class Teacher:
    def __init__(self, paths, device='cuda:0'):
        SSLTask=load_ssl_task()
        self.device=torch.device(device)
        self.model,_=SSLTask.build_model_from_file(str(paths[0]),str(paths[1]),str(self.device))
        self.model.float().eval().requires_grad_(False)
        if len(self.model.encoder.encoders)!=19:raise ValueError('Unexpected XEUS architecture')
        # FP32 reference extraction must not depend on an optional FlashAttention
        # installation or its implicit half-precision conversion.
        for module in self.model.modules():
            if hasattr(module,'use_flash_attn'):module.use_flash_attn=False
        self.verified=False

    def _encode(self,wave,short):
        x=torch.as_tensor(wave,dtype=torch.float32,device=self.device)[None]
        lengths=torch.tensor([x.shape[1]],dtype=torch.long,device=self.device)
        def stop(_module,_inputs,output):
            value=output[0]
            if isinstance(value,tuple):value=value[0]
            raise _ReachedLayer(value)
        hook=self.model.encoder.encoders[9].register_forward_hook(stop) if short else None
        try:
            with torch.inference_mode():
                try:
                    # Pinned v.202506 returns (final, intermediate_list, lengths),
                    # unlike the earlier fork's encode API.
                    states=self.model.inference_encode(x,lengths,use_mask=False,use_final_output=False)[1]
                    return states[9][0].float().cpu().numpy()
                except _ReachedLayer as reached:return reached.value[0].float().cpu().numpy()
        finally:
            if hook is not None:hook.remove()

    def __call__(self,wave):
        # Single-recording inference preserves the official global waveform
        # normalization; no padding/batch composition changes its statistics.
        if len(wave)<400:return np.empty((0,1024),np.float32)
        if not self.verified:
            probe=wave[:min(len(wave),32000)]
            a=self._encode(probe,False);b=self._encode(probe,True)
            if a.shape!=b.shape or not np.allclose(a,b,atol=2e-5,rtol=2e-5):
                raise ValueError('Truncated XEUS block 10 differs from official full encode')
            self.verified=True
        pieces=[];core=320000;context=16000
        for start in range(0,len(wave),core):
            stop=min(len(wave),start+core);left=max(0,start-context);right=min(len(wave),stop+context)
            if right-left<400:continue
            values=self._encode(wave[left:right],True)
            centers=left+200+320*np.arange(len(values))
            pieces.append(values[(centers>=start)&(centers<stop)&(centers+200<=len(wave))])
        values=np.concatenate(pieces) if pieces else np.empty((0,1024),np.float32)
        if values.ndim!=2 or values.shape[1]!=1024 or not np.isfinite(values).all():
            raise ValueError('Invalid XEUS representations')
        return values
