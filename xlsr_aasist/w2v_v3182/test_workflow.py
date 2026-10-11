"""End-to-end CPU train/resume/fork/export using real AASIST and synthetic audio."""
import copy
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .common import atomic_json,read_json,digest,partial_state
from .train import run_experiment
from .evaluate import export,selected
from .test_support import make_fixture,tiny_model,NativeBoundary


def save_inputs(run,cfg,train,dev):
    run.mkdir()
    atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
    cfg['saved_manifests']={n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
    atomic_json(run/'config.json',cfg)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_resume_and_shared_warmup_exactly_reproduce_joint_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,train,dev=make_fixture(root/'audio')
            cfg['data_run']='synthetic data fixture';cfg['epochs']=2
            a=root/'parent';save_inputs(a,cfg,train,dev)
            with patch('w2v_v3182.data.Engines',NativeBoundary), patch('w2v_v318.data.Engines',NativeBoundary), redirect_stdout(io.StringIO()):
                ready=run_experiment(cfg,a,train,dev,tiny_model,stop_after_warmup=True)
                self.assertEqual(ready['status'],'warmup_ready')
                with self.assertRaises(ValueError): selected(a,'best')
                warm_hash=digest(a/'warmup.pt')
                # Simulate a killed, uncommitted write before the next resume.
                with (a/'steps.jsonl').open('a') as f:
                    f.write(json.dumps(dict(update=999,epoch=2,step=999))+'\n{"broken":')
                done=run_experiment(cfg,a,train,dev,tiny_model)
                self.assertEqual(done['status'],'complete')
                self.assertEqual(digest(a/'warmup.pt'),warm_hash)
                atomic_json(a/'training_history.json',[])
                recovered=run_experiment(cfg,a,train,dev,tiny_model)
                self.assertEqual(recovered['status'],'complete')
                self.assertEqual(len(read_json(a/'training_history.json')),2)
                b=root/'fork';fork_cfg=copy.deepcopy(cfg)
                fork_cfg.update(warmup_parent=str(a),initialization={'mode':'shared warmup fixture'})
                save_inputs(b,fork_cfg,train,dev)
                fork=run_experiment(fork_cfg,b,train,dev,tiny_model)
                self.assertEqual(fork['status'],'complete')
                pa=torch.load(a/'last.pt',map_location='cpu',weights_only=True)
                pb=torch.load(b/'last.pt',map_location='cpu',weights_only=True)
                for name in pa['model']:
                    self.assertTrue(torch.equal(pa['model'][name],pb['model'][name]),name)
                self.assertEqual(pa['best_metrics'],pb['best_metrics'])
                c=root/'control';control_cfg=copy.deepcopy(fork_cfg);control_cfg['contrastive_weight']=0.
                save_inputs(c,control_cfg,train,dev)
                control_done=run_experiment(control_cfg,c,train,dev,tiny_model)
                self.assertEqual(control_done['status'],'complete')
                self.assertEqual(read_json(c/'training_history.json')[1]['training']['pair_loss'],0.)
                from .compare import compare
                report=compare(c,a)
                self.assertEqual(report['shared_warmup_sha256'],warm_hash)
                self.assertIn('real_at_95_fake',report['groups']['online/en'])
                steps=[json.loads(s) for s in (a/'steps.jsonl').read_text().splitlines()]
                self.assertEqual([s['update'] for s in steps],list(range(1,pa['cursor']+1)))
                history=read_json(a/'training_history.json')
                self.assertEqual(history[0]['training']['pair_loss'],0.)
                self.assertGreater(history[1]['training']['pair_loss'],0.)
                self.assertEqual(history[1]['training']['mean_unique_sources_per_update'],4)
                self.assertTrue((a/'diagnostics'/'curves.html').is_file())
                self.assertIn('real_at_95_fake',(a/'diagnostics'/'groups.csv').read_text(encoding='utf-8-sig'))
                out=root/'validation'
                result=export(a,out,dev=True,device='cpu',workers=0,model_factory=tiny_model,verify=False)
                metrics=read_json(result)
                self.assertIn('matched_95',metrics['raw']['groups']['online/en'])
                cal=read_json(a/'calibration_best.json')
                self.assertGreaterEqual(max(cal['raw_selection']['weighted_f1'],cal['affine_selection']['weighted_f1']),cal['raw_selection']['weighted_f1'])
                before=digest(a/'last.pt')
                protocol=root/'eval.txt'
                protocol.write_text('\n'.join(r['id'] for r in dev if r['condition']=='online')+'\n')
                submission=export(a,root/'submission',protocol=str(protocol),audio_root=str(root/'audio'/'dev'),
                    device='cpu',workers=0,model_factory=tiny_model,verify=False)
                self.assertTrue(submission.is_file());self.assertEqual(before,digest(a/'last.pt'))
                cfg_bad=dict(cfg,contrastive_weight=.2)
                with self.assertRaises(ValueError):run_experiment(cfg_bad,a,train,dev,tiny_model)
                from .warmup import initial_state
                with self.assertRaises(ValueError):initial_state(a,dict(fork_cfg,lora_rank=7))

    def test_calibration_identity_fallback(self):
        from .calibration import choose
        rows=[];logits=[]
        for c in ('online','seen','heldout'):
            for lang in ('en','zh'):
                for label in (0,1):
                    rows.append(dict(condition=c,language=lang,label=label))
                    logits.append([2.,0.] if label==0 else [0.,2.])
        z=np.array(logits*2);rows=rows*2
        with patch('w2v_v3182.calibration.fit',return_value=dict(scale=1.,bias=100.)) as fit:
            selected=choose(rows,z,dict(calibration=list(range(12)),select=list(range(12,24))))
        self.assertEqual(fit.call_args.args[2],list(range(12)))
        self.assertFalse(selected['enabled']);self.assertEqual(selected['scale'],1.);self.assertEqual(selected['bias'],0.)

    def test_resume_log_trims_uncommitted_records(self):
        from .recovery import trim_step_log
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'steps.jsonl'
            p.write_text('{"update":1}\n{"update":2}\n{"update":3}\n{"update":',encoding='utf-8')
            trim_step_log(p,2)
            self.assertEqual([json.loads(s)['update'] for s in p.read_text().splitlines()],[1,2])


if __name__=='__main__': unittest.main()
