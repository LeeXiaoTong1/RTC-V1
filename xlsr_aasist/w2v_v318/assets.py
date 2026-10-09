"""Pinned official Omni SSL encoders; never substitute CTC or LLM weights."""

DEFAULT_ARCH = '3b'
MODELS = {
    '1b': dict(encoder_dim=1280, encoder_layers=48, download_bytes=4*1024**3,
               sha256='a3194cc50f7ad02a19c92bf75b2970ce94f026b60ab712bd4b720f1219896587',
               microbatch=8, frame_budget=4800, eval_batch=16),
    '3b': dict(encoder_dim=2048, encoder_layers=60, download_bytes=12256910184,
               sha256='09d2a59f5106afb3508a498c4be51a290dc0096f2065d3e797ebc645ed462659',
               microbatch=4, frame_budget=2400, eval_batch=8),
}
# 3B file identity: official HF commit b34b7fba5ac95adbffd9e60813a4425cb0fc6242.
# https://huggingface.co/facebook/omniASR-W2V-3B/commit/b34b7fba5ac95adbffd9e60813a4425cb0fc6242


def spec(arch):
    if arch not in MODELS: raise ValueError('Choose official Omni W2V 1b or 3b')
    name='omniASR-W2V-'+arch.upper()
    return dict(MODELS[arch],arch=arch,name=name,repo='facebook/'+name,
                schema='rtc_omni_w2v'+arch+'_assets_v1',
                source='https://dl.fbaipublicfiles.com/mms/'+name+'.pt')


def default_assets(arch=DEFAULT_ARCH):
    return 'pretrained/'+spec(arch)['name']+'/assets.json'


def validate_assets(assets,arch):
    expected=spec(arch)
    for key in ('schema','arch','repo','sha256','encoder_dim','encoder_layers'):
        if assets.get(key)!=expected[key]:
            raise ValueError(f'Expected official W2V {arch.upper()} assets: {key} differs; prepare the matching model')
    if not assets.get('checkpoint'):raise ValueError('Missing Omni checkpoint path')
    return expected


def configured_spec(cfg):
    # Existing 1B run configs did not have omni_arch; never reinterpret them as 3B.
    arch=cfg.get('omni_arch','1b')
    item=spec(arch)
    if (cfg.get('encoder_dim')!=item['encoder_dim'] or cfg.get('encoder_layers')!=item['encoder_layers']
            or cfg.get('omni_sha256')!=item['sha256']):
        raise ValueError('Omni architecture/dimensions/weight identity disagree')
    validate_assets(cfg['omni_provenance'],arch)
    return item
