"""Resumable pinned-model download, followed by strictly local HF loading.

Use individual files: a mirror need not implement the repository-listing API.
The hub owns blob/partial files; changing endpoints reuses the same cache.
Only public model assets are requested, never audio or authentication tokens.
"""
import hashlib
import json
import os
from pathlib import Path

from .phonemes import MODEL_ID, MODEL_REVISION, atomic_json

OFFICIAL='https://huggingface.co'
MIRROR='https://hf-mirror.com'
ENDPOINTS=(OFFICIAL,MIRROR)
FILES=('preprocessor_config.json','config.json','tokenizer_config.json',
       'special_tokens_map.json','vocab.json','pytorch_model.bin')
# Official immutable file page, checked independently of the mirror:
# https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/blob/2c733782da5604684829819a5eb744c193fe9398/pytorch_model.bin
WEIGHT_SHA256='04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551'


class ModelDownloadError(RuntimeError):
    pass


def add_download_arguments(parser):
    parser.add_argument('--hf-endpoint',choices=('auto',)+ENDPOINTS,default='auto',
        help='Preferred public download endpoint; auto honors HF_ENDPOINT, then tries the other endpoint')
    parser.add_argument('--download-retries',type=int,choices=range(1,6),default=2,
        help='Bounded attempts per endpoint; interrupted hub downloads resume automatically')


def endpoint_order(preferred='auto'):
    first=os.environ.get('HF_ENDPOINT',OFFICIAL) if preferred=='auto' else preferred
    first=first.rstrip('/')
    if first not in ENDPOINTS:
        raise ValueError('Use --hf-endpoint '+MIRROR+' or '+OFFICIAL+' for this public model')
    return [first]+[value for value in ENDPOINTS if value!=first]


def snapshot_path(cache):
    return Path(cache)/('models--'+MODEL_ID.replace('/','--'))/'snapshots'/MODEL_REVISION


def verify_file(path,name):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:raise ValueError('Missing/empty model file: '+name)
    # Reject HTML error pages, truncated JSON, incompatible geometry, and LFS
    # pointers. Only the binary hash is an independently pinned byte checksum;
    # JSON hashes below are recorded for local provenance, not an upstream pin.
    if name.endswith('.json'):
        if path.stat().st_size>128*1024:raise ValueError('Unexpectedly large model JSON: '+name)
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or 'auto_map' in data:raise ValueError('Invalid model JSON: '+name)
        expected={
            'preprocessor_config.json':dict(sampling_rate=16000,feature_extractor_type='Wav2Vec2FeatureExtractor',
                                           feature_size=1,do_normalize=True),
            'config.json':dict(model_type='wav2vec2',hidden_size=1024,num_hidden_layers=24,
                               num_attention_heads=16,vocab_size=392,pad_token_id=0,
                               conv_stride=[5,2,2,2,2,2,2],conv_kernel=[10,3,3,3,3,2,2]),
            'tokenizer_config.json':dict(tokenizer_class='Wav2Vec2PhonemeCTCTokenizer'),
            'special_tokens_map.json':dict(pad_token='<pad>',unk_token='<unk>',bos_token='<s>',eos_token='</s>')
        }.get(name,{})
        if any(data.get(key)!=value for key,value in expected.items()):
            raise ValueError('Pinned model configuration differs: '+name)
        if name=='vocab.json' and (len(data)!=392 or set(data.values())!=set(range(392))
                                   or data.get('<pad>')!=0):
            raise ValueError('Pinned phoneme vocabulary differs')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    sha=h.hexdigest()
    if name=='pytorch_model.bin' and sha!=WEIGHT_SHA256:
        raise ValueError('Pinned phoneme weight SHA256 mismatch')
    return dict(bytes=path.stat().st_size,sha256=sha)


def ensure_model(cache,local_files_only=False,preferred='auto',retries=2,*,download=None):
    """Reuse verified cache entries; repair only invalid files, never clear cache."""
    if not 1<=retries<=5:raise ValueError('download retries must be between 1 and 5')
    order=None
    cache=Path(cache).expanduser().resolve();cache.mkdir(parents=True,exist_ok=True)
    snapshot=snapshot_path(cache);records={}
    # Imported only when a missing/invalid file actually needs network access.
    # A complete cache and --local-files-only work without contacting the Hub.
    for index,name in enumerate(FILES,1):
        target=snapshot/name;force=False
        if target.is_file():
            try:
                records[name]=dict(verify_file(target,name),source='verified-cache')
                print(f'[Phoneme model] {index}/{len(FILES)} {name}: cached, verified',flush=True)
                continue
            except (OSError,ValueError) as exc:
                if local_files_only:raise ModelDownloadError('Invalid cached '+name+': '+str(exc)) from None
                force=True
        if local_files_only:
            raise ModelDownloadError('Missing cached '+name+' at revision '+MODEL_REVISION+
                '. Remove --local-files-only and use --hf-endpoint '+MIRROR)
        if order is None:order=endpoint_order(preferred)
        if download is None:
            from huggingface_hub import hf_hub_download
            download=hf_hub_download
        errors=[];success=False
        for endpoint in list(order):
            for attempt in range(1,retries+1):
                print(f'[Phoneme model] {index}/{len(FILES)} {name}: {endpoint} attempt {attempt}/{retries}',flush=True)
                try:
                    value=download(repo_id=MODEL_ID,filename=name,revision=MODEL_REVISION,
                        cache_dir=str(cache),endpoint=endpoint,token=False,local_files_only=False,
                        etag_timeout=10,force_download=force)
                    # Keep the standard snapshot: no second copy of the 1.26 GB weights.
                    if Path(value).absolute()!=target.absolute():
                        raise ValueError('Downloader returned a different model/revision path')
                    records[name]=dict(verify_file(target,name),source=endpoint)
                    success=True;order=[endpoint]+[x for x in order if x!=endpoint]
                    break
                except (OSError,ValueError,RuntimeError) as exc:
                    # Network failures leave .incomplete files for HF to resume.
                    # A completed-but-invalid blob must bypass HF's cache hit.
                    force=force or target.is_file()
                    status=getattr(getattr(exc,'response',None),'status_code',None)
                    reason=type(exc).__name__+((' HTTP '+str(status)) if status is not None else '')
                    errors.append(endpoint+' '+reason)
                    print('[Phoneme model] retryable failure: '+reason,flush=True)
            if success:break
        if not success:
            raise ModelDownloadError('Cannot download/verify '+name+'. Tried '+', '.join(errors)+
                '. Cache/partial files retained in '+str(cache)+
                '. Check server HTTPS/proxy access and rerun the same preparation command.')
    receipt=dict(model_id=MODEL_ID,revision=MODEL_REVISION,snapshot=str(snapshot),files=records,
                 weight_sha256=WEIGHT_SHA256,local_loading_only=True)
    atomic_json(cache/'v3186_model_download.json',receipt)
    print('PHONEME_MODEL_READY='+str(snapshot),flush=True)
    return snapshot
