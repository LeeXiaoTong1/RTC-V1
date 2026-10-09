"""Frontend selection, asset reuse, 3B projection and export identity checks."""
from pathlib import Path
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import torch
from torch import nn
from .assets import spec,default_assets,validate_assets,configured_spec
from .common import atomic_json,digest,fingerprint,schema_for
from .config import parser,validate
from .model import Detector,load_model,inject,LoRALinear
from .prepare import prepare
from .test_core import TinyLayout


def manifest(arch,path='official.pt'):
    item=spec(arch)
    return dict({k:item[k] for k in ('schema','repo','source','arch','sha256','encoder_dim','encoder_layers')},checkpoint=str(path))


def config(arch):
    item=spec(arch)
    return dict(omni_arch=arch,omni_sha256=item['sha256'],omni_checkpoint='official.pt',omni_provenance=manifest(arch),
        encoder_dim=item['encoder_dim'],encoder_layers=item['encoder_layers'],device='cpu',
        lora_layers=16,lora_rank=16,lora_alpha=32.,lora_dropout=.05,checkpointing=True,
        local_evidence=True,window=128,hop=64,window_batch=32)


class TestAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_selected_size_defaults_and_explicit_budgets(self):
        for arch,budgets,depth in (('3b',(4,2400,8),60),('1b',(8,4800,16),48)):
            args=parser().parse_args(['--omni-size',arch]);validate(args)
            self.assertEqual((args.microbatch,args.frame_budget,args.eval_batch),budgets)
            self.assertEqual((args.stream_sources,args.epochs,args.warm_epochs),(16,10,2))
            validate(parser().parse_args(['--omni-size',arch,'--lora-layers',str(depth)]))
            with self.assertRaises(ValueError):validate(parser().parse_args(['--omni-size',arch,'--lora-layers',str(depth+1)]))
        args=parser().parse_args(['--microbatch','6','--frame-budget','3000','--eval-batch','4']);validate(args)
        self.assertEqual((args.omni_size,args.microbatch,args.frame_budget,args.eval_batch),('3b',6,3000,4))
        self.assertIn('W2V-3B',default_assets())

    def test_reject_wrong_frontend_assets_and_preserve_legacy_identity(self):
        for key in ('schema','arch','repo','sha256','encoder_dim','encoder_layers'):
            bad=manifest('3b');bad[key]=manifest('1b')[key]
            with self.assertRaises(ValueError):validate_assets(bad,'3b')
        old=config('1b');old.pop('omni_arch')
        self.assertEqual(configured_spec(old)['arch'],'1b')
        mixed=config('3b');mixed['encoder_dim']=1280
        with self.assertRaises(ValueError):configured_spec(mixed)

    def test_existing_official_file_is_referenced_and_rechecked_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);weights=root/'existing.pt';weights.write_bytes(b'fixture')
            catalog=spec('3b');catalog['sha256']=digest(weights)
            with patch('w2v_v318.assets.MODELS',{'3b':catalog}),patch('w2v_v318.prepare.subprocess.run') as curl:
                a=prepare(root/'assets',weights,'3b');b=prepare(root/'assets',arch='3b')
                self.assertEqual(a,b);self.assertEqual(Path(a['checkpoint']),weights.resolve())
                self.assertFalse((root/'assets'/'omniASR-W2V-3B.pt').exists());curl.assert_not_called()
                weights.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):prepare(root/'assets',arch='3b')

    def test_resumed_download_budget_hash_and_partial_rename(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);partial=root/'omniASR-W2V-3B.pt.part';partial.write_bytes(b'fi')
            complete=b'fixture';catalog=spec('3b')
            import hashlib
            catalog.update(download_bytes=len(complete),sha256=hashlib.sha256(complete).hexdigest())
            needed=len(complete)-2+12*1024**3
            def download(argv,check):
                self.assertEqual(argv[argv.index('--continue-at')+1],'-')
                self.assertEqual(argv[-1],catalog['source']);partial.write_bytes(complete)
            with patch('w2v_v318.assets.MODELS',{'3b':catalog}),patch('w2v_v318.prepare.subprocess.run',side_effect=download) as curl:
                with patch('w2v_v318.prepare.shutil.disk_usage',return_value=SimpleNamespace(free=needed-1)):
                    with self.assertRaises(OSError):prepare(root,arch='3b')
                    curl.assert_not_called()
                with patch('w2v_v318.prepare.shutil.disk_usage',return_value=SimpleNamespace(free=needed)):
                    info=prepare(root,arch='3b')
                self.assertFalse(partial.exists());self.assertEqual(Path(info['checkpoint']).read_bytes(),complete)

    def test_hub_loads_selected_3b_architecture_and_rejects_wrong_depth(self):
        hub=Mock();arch=SimpleNamespace(encoder_config=SimpleNamespace(model_dim=2048,num_encoder_layers=60))
        hub.get_arch_config.return_value=arch
        hub.load_custom_model.return_value=SimpleNamespace(encoder_frontend=nn.Identity(),encoder=SimpleNamespace(layers=[None]*60))
        modules={name:ModuleType(name) for name in ('omnilingual_asr','fairseq2','fairseq2.models','fairseq2.models.wav2vec2','fairseq2.nn','fairseq2.nn.batch_layout')}
        modules['fairseq2.models.wav2vec2'].get_wav2vec2_model_hub=lambda:hub
        modules['fairseq2.nn.batch_layout'].BatchLayout=TinyLayout
        with patch.dict('sys.modules',modules),patch('w2v_v318.runtime.version',return_value='0.6'),patch('w2v_v318.model.Detector',return_value=nn.Identity()):
            load_model(config('3b'));hub.get_arch_config.assert_called_once_with('3b')
            self.assertIs(hub.load_custom_model.call_args.args[1],arch)
            self.assertTrue(hub.load_custom_model.call_args.kwargs['mmap'])
            hub.reset_mock();arch.encoder_config.num_encoder_layers=48
            with self.assertRaisesRegex(ValueError,'architecture'):load_model(config('3b'))
            hub.load_custom_model.assert_not_called()

    def test_real_3b_projection_and_last_16_layer_lora_shapes(self):
        cfg=config('3b')
        # Meta tensors exercise all 60 layer indices and 2048-wide projection shapes without allocating 3B weights.
        with torch.device('meta'):
            frontend=nn.Linear(1,2048);encoder=nn.Module()
            encoder.layers=nn.ModuleList([nn.ModuleDict({name:nn.Linear(2048,2048) for name in ('q_proj','k_proj','v_proj','output_proj')}) for _ in range(60)])
            model=Detector(frontend,encoder,cfg,TinyLayout)
        self.assertEqual(set(model.lora_inventory),{str(i) for i in range(44,60)})
        self.assertEqual(sum(p.numel() for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))),4194304)
        self.assertEqual(sum(p.numel() for p in model.head.parameters()),583499)
        self.assertFalse(any(p.requires_grad for n,p in model.encoder.named_parameters() if not n.endswith(('.lora_a','.lora_b'))))
        from .aasist import SSLAASIST
        x=torch.randn(2,20,2048,requires_grad=True);head=SSLAASIST(2048).eval()
        logits=head(x);self.assertEqual(logits.shape,(2,2));logits.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_attention_injection_does_not_touch_fairseq_ffn_output_projection(self):
        layer=nn.Module();dim=32
        layer.self_attn=nn.ModuleDict({name:nn.Linear(dim,dim) for name in ('q_proj','k_proj','v_proj','output_proj')})
        layer.ffn=nn.ModuleDict({'inner_proj':nn.Linear(dim,4*dim),'output_proj':nn.Linear(4*dim,dim)})
        layer.requires_grad_(False);ffn=layer.ffn['output_proj'];before=ffn.weight.detach().clone()
        names=inject(layer,dict(encoder_dim=dim,lora_rank=4,lora_alpha=8.,lora_dropout=0.))
        self.assertEqual(set(names),{'self_attn.'+n for n in ('q_proj','k_proj','v_proj','output_proj')})
        self.assertIs(layer.ffn['output_proj'],ffn);self.assertTrue(torch.equal(before,ffn.weight))
        self.assertFalse(ffn.weight.requires_grad)
        self.assertTrue(all(isinstance(m,LoRALinear) for m in layer.self_attn.values()))

    def test_export_rejects_checkpoint_of_another_frontend(self):
        from .evaluate import selected
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);cfg=config('3b');tag='epoch_1_step_1'
            atomic_json(run/'config.json',cfg)
            state=dict(schema=schema_for(cfg),identity=fingerprint(cfg),best_tag=tag,last_tag=tag,
                history=[dict(tag=tag)],best_model={},model={})
            for model_schema,valid in ((schema_for(cfg),True),('rtc_v318_omni1b_aasist_v1',False)):
                state['schema']=model_schema;torch.save(state,run/'last.pt')
                atomic_json(run/'completed.json',dict(version='3.18',status='complete',best_tag=tag,last_tag=tag,checkpoint_sha256=digest(run/'last.pt')))
                if valid:self.assertEqual(selected(run,'best')[2]['selected'],tag)
                else:
                    with self.assertRaisesRegex(ValueError,'mismatch'):selected(run,'best')


if __name__=='__main__':unittest.main()
