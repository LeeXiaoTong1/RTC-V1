"""Real WAV/feature/SSL training, epoch resume, validation and ZIP export."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from w2v_v39.common import atomic_json,read_json,digest
from w2v_v3161.test_release import ReleaseFixture,NativeFixtureEngine
from .arguments import parser,validate
from .config import configuration,verify_inputs
from .train import run_experiment
from .state import load_resume,load_selected,apply_partial
from .model import load_model
from .data import SourcePlan,TrainingLoader,tensors,validate_batch
from .evaluate import export
from .output import infer


class ReleaseTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_wave_training_resume_and_both_exports_use_fresh_new_architecture(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()), \
             patch('w2v_v317.config.runtime',return_value={'fixture_native_apm':True}), \
             patch('w2v_v316_tfcl.data.Engines',NativeFixtureEngine):
            f=ReleaseFixture(directory)
            # No source LAST or Adam is created. Merely the verified data manifest.
            self.assertFalse((f.source/'last.pt').exists())
            cfg=configuration(parser().parse_args(['--data-run',str(f.source),'--epochs','2',
                '--workers','0','--device','cpu','--no-autotune','--lora-layers','1','--lora-rank','2']))
            cfg.update(feature_dim=8,head_expansion=32,head_blocks=2,tfcl_heads=2,tfcl_bins=21,
                       train_probe_per_group=4,disk_margin_bytes=0,free_reserve_bytes=0)
            verify_inputs(cfg)
            run=f.root/'new_run';run.mkdir();atomic_json(run/'config.json',cfg)
            plan=SourcePlan(f.train,16,cfg['seed'],cfg)
            loader=TrainingLoader(plan,cfg,run)
            try:examples=tensors(next(loader.segment(0,0,1)))
            finally:loader.close()
            validate_batch(examples,16,cfg)
            # Interrupt AFTER saving epoch 1, then resume the same immutable config.
            from .train import save as real_save
            def interrupted(*args,**kw):
                real_save(*args,**kw)
                if args[2]['cursor']==plan.steps:raise RuntimeError('simulated interruption')
            with patch('w2v_v317.train.save',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):run_experiment(cfg,run)
            committed=load_resume(run,cfg);self.assertEqual(committed['cursor'],1)
            run_experiment(cfg,run)
            state=load_resume(run,cfg)
            self.assertEqual(state['cursor'],2);self.assertEqual(len(state['history']),2)
            self.assertFalse(read_json(run/'initialization.json')['detector_checkpoint_loaded'])
            self.assertTrue(all(n.startswith('head.') or n.endswith(('lora_A','lora_B')) for n in state['model']))
            # A clean run must equal interrupted/resumed training tensor for tensor.
            control=f.root/'control';control.mkdir();atomic_json(control/'config.json',cfg)
            run_experiment(cfg,control);reference=load_resume(control,cfg)
            for n,v in reference['model'].items():torch.testing.assert_close(v,state['model'][n],rtol=0,atol=0)
            for kind in ('best','last'):
                checkpoint,meta=load_selected(run,kind)
                self.assertEqual(meta['origin']['version'],'3.17');self.assertFalse(meta['baseline_fallback'])
                target=f.root/('validation_'+kind)
                path=export(run,None,None,target,'cpu',0,kind,True)
                expected=next(e['metrics'] for e in state['history'] if e['tag']==meta['selected'])
                self.assertEqual(read_json(path),expected)
            protocol=f.root/'submission_protocol.txt'
            protocol.write_text('\n'.join(Path(r['audio']).name for r in f.dev)+'\n',encoding='utf-8')
            archive=export(run,protocol,f.root,f.root/'submission','cpu',0,'best')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(),['scores.txt'])
                self.assertEqual(len(z.read('scores.txt').decode().splitlines()),len(f.dev))
            self.assertEqual(read_json(f.root/'submission'/'submission_meta.json')['version'],'3.17')
            # Frozen public weights cannot be changed without invalidating export.
            weight=next(Path(p) for p in cfg['pretrained_fingerprints'] if p.endswith('.safetensors'))
            with weight.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Pinned input'):verify_inputs(cfg)

    def test_sampling_smooths_repetition_but_preserves_group_loss_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            f=ReleaseFixture(directory)
            # Synthetic source identities test quotas without reading these waves.
            rows=[]
            import hashlib
            sizes={('en',1):15,('en',0):75,('zh',1):60,('zh',0):237}
            for (language,label),count in sizes.items():
                template=[r for r in f.train if r['language']==language and r['label']==label][:4]
                for i in range(count):
                    source=f'{language}-{label}-{i}';sha=hashlib.sha256(source.encode()).hexdigest()
                    for original in template:
                        condition=original['condition'];row=dict(original,source_id=source,group_id=sha,
                            source_sha256=sha,audio_sha256=sha,id=source if condition=='offline' else source+condition)
                        rows.append(row)
            plan=SourcePlan(rows,16,31701,{'source_sampling_power':.5})
            self.assertEqual(plan.quotas,{('en',0):4,('en',1):2,('zh',0):7,('zh',1):3})
            self.assertEqual(plan.batches(2,1,2),plan.batches(2)[1:2])
            streams=plan._streams(2);self.assertEqual(len(streams[('en',1)]),plan.steps*2)
            all_rows=[]
            for ticket in plan.batches(0,0,1)[0]:
                row=rows[ticket.original];mass=.25/plan.quotas[(row['language'],row['label'])]
                for role,view_mass in (('offline',.1),('online',.5),('noisy',.4)):
                    all_rows.append(dict(row,role=role,pair_occurrence=ticket.occurrence,
                        source_mass=mass,ce_weight=mass*view_mass))
            validate_batch(all_rows,16)
            all_rows[0]['ce_weight']*=2
            with self.assertRaises(ValueError):validate_batch(all_rows,16)

    def test_cli_rejects_bad_limits_before_any_training(self):
        for args in (['--epochs','5'],['--microbatch','2'],['--lora-rank','0'],['--lora-lr','nan']):
            with self.assertRaises(ValueError):validate(parser().parse_args(args))


if __name__=='__main__':unittest.main()
