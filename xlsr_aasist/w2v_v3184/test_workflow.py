"""Real AASIST training/resume/export with a completed synthetic R2 best fixture."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_v318.test_core import make_fixture,NativeBoundary
from w2v_v3183r2.test_support import tiny_cfg as r2_cfg
from w2v_v3183r2.common import schema_for as r2_schema
from w2v_v3183r2.monitor import panel_rows
from .test_support import tiny_cfg,tiny_model,scene_fixture
from .common import atomic_json,read_json,digest,partial_state,seed_all,fingerprint
from .sampling import Plan,probe_tickets
from .records import dev_partition
from .inference import infer
from .parent import inspect_parent
from .train import run_experiment,save
from .evaluate import export,selected


def save_inputs(run,cfg,train,dev):
    run.mkdir();atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
    cfg['saved_manifests']={n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
    atomic_json(run/'config.json',cfg)


def completed_r2(root):
    cfg,train,dev=make_fixture(root/'audio');cfg.update(r2_cfg())
    parent=root/'parent';save_inputs(parent,cfg,train,dev)
    atomic_json(parent/'dev_partition.json',dev_partition(dev,cfg['dev_pairs'],cfg['seed'],cfg['calibration_fraction']))
    plan=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'])
    atomic_json(parent/'train_probe_tickets.json',probe_tickets(plan,cfg['train_probe_per_group']))
    rows,tickets=panel_rows(cfg);atomic_json(parent/'independent_panel.json',dict(rows=rows,tickets=tickets))
    seed_all(cfg['seed']);model=tiny_model(cfg)
    with torch.no_grad():
        for name,value in model.named_parameters():
            if name.endswith('.lora_b'):value.normal_(0,.002)
        for name,value in model.named_buffers():
            if name.endswith('running_var'):value.fill_(1.1)
            if name.endswith('num_batches_tracked'):value.fill_(5)
    best=partial_state(model);last=copy.deepcopy(best);last['fusion.layer_logits'].fill_(.5)
    logits,_=infer(model,dev,cfg)
    torch.save(dict(schema=r2_schema(cfg),identity=fingerprint(cfg),best_tag='epoch_1_step_3',last_tag='epoch_2_step_3',
        best_model=best,model=last,best_logits=torch.from_numpy(logits),last_logits=torch.from_numpy(-logits),
        history=[{'tag':'epoch_1_step_3'},{'tag':'epoch_2_step_3'}]),parent/'last.pt')
    atomic_json(parent/'completed.json',dict(version='3.18.3-R2',status='complete',best_tag='epoch_1_step_3',
        last_tag='epoch_2_step_3',checkpoint_sha256=digest(parent/'last.pt')))
    return parent,cfg,train,dev,best


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_r2_best_scenes_resume_selection_and_submission(self):
        with tempfile.TemporaryDirectory() as d,patch('w2v_v3184.data.Engines',NativeBoundary),redirect_stdout(io.StringIO()):
            root=Path(d);parent,prior,train,dev,parent_best=completed_r2(root)
            original_hash=digest(parent/'last.pt');_,weights,initialization=inspect_parent(parent)
            self.assertEqual(initialization['tag'],'epoch_1_step_3')
            cfg=tiny_cfg()
            for key in ('dev_pairs','official_dev_protocol','official_dev_root','noise_records'):cfg[key]=prior[key]
            cfg.update(scene_fixture(root/'scenes'))
            cfg['initialization']=dict(initialization,mode='parent-transfer',old_optimizer_loaded=False)
            full=root/'full';save_inputs(full,cfg,train,dev)
            run_experiment(cfg,full,train,dev,tiny_model,baseline_only=True)
            initial=torch.load(full/'last.pt',weights_only=True)
            for key,value in parent_best.items():torch.testing.assert_close(initial['model'][key],value,rtol=0,atol=0)
            self.assertTrue(read_json(full/'initialization_audit.json')['exact_continuation'])
            self.assertEqual(read_json(full/'initialization_audit.json')['bn_statistics_reset'],[])
            self.assertEqual(read_json(full/'scene_baseline.json')['count'],24)
            self.assertFalse((full/'completed.json').exists())
            with self.assertRaises(FileNotFoundError):selected(full,'best')
            run_experiment(cfg,full,train,dev,tiny_model)
            resumed=root/'resumed';save_inputs(resumed,cfg,train,dev)
            def interrupt(*args):
                save(*args)
                if args[2]['epoch']==1:raise RuntimeError('test interruption')
            with patch('w2v_v3184.train.save',interrupt),self.assertRaisesRegex(RuntimeError,'test interruption'):
                run_experiment(cfg,resumed,train,dev,tiny_model)
            with (resumed/'steps.jsonl').open('a') as stream:stream.write('{"update":999}\n{"broken":')
            run_experiment(cfg,resumed,train,dev,tiny_model)
            a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            self.assertEqual(a['best_tag'],b['best_tag']);self.assertEqual(a['scene_baseline'],b['scene_baseline'])
            self.assertTrue(any(not torch.equal(initial['model'][n],v) for n,v in a['model'].items() if n.endswith('.lora_b')))
            self.assertEqual([e['phase'] for e in a['history']],['head_fusion_warmup','joint'])
            self.assertEqual([json.loads(s)['update'] for s in (resumed/'steps.jsonl').read_text().splitlines()],list(range(1,b['cursor']+1)))
            for entry in a['history']:
                self.assertEqual(entry['scenes']['count'],24);self.assertEqual(entry['dev']['count'],len(dev))
                self.assertEqual({k for k in entry['metrics']['groups'] if '/' in k},
                    {'online/en','online/zh','seen/en','seen/zh','heldout/en','heldout/zh'})
            self.assertEqual(digest(parent/'last.pt'),original_hash)
            atomic_json(resumed/'training_history.json',[])
            run_experiment(cfg,resumed,train,dev,tiny_model)
            self.assertEqual(len(read_json(resumed/'training_history.json')),2)
            out=root/'dev-export';export(resumed,out,dev=True,device='cpu',workers=0,model_factory=tiny_model,verify=False)
            self.assertEqual(read_json(out/'validation_meta.json')['rtc_policy'],'legacy')
            protocol=root/'eval.txt';ids=[r['id'] for r in dev if r['condition']=='online'];protocol.write_text('\n'.join(ids)+'\n')
            from .delivery import prepare_request,finish_request
            from .config import parser
            from functools import partial
            args=parser().parse_args(['--submit-best','--upload-submission','--submission-protocol',str(protocol),
                '--submission-audio-root',str(root/'audio'/'dev'),'--submission-out',str(root/'automatic-best')])
            request=prepare_request(args,resumed,cfg)
            with patch('w2v_v3184.delivery.ROOT',root),patch('w2v_v3184.delivery.upload_archive',side_effect=RuntimeError('network fixture')):
                with self.assertRaisesRegex(RuntimeError,'Retry without GPU'):
                    finish_request(resumed,cfg,request,partial(export,model_factory=tiny_model,verify=False))
            self.assertEqual([s.split()[0] for s in (root/'automatic-best'/'scores.txt').read_text().splitlines()],ids)
            with patch('w2v_v3184.delivery.ROOT',root),patch('w2v_v3184.delivery.upload_archive',return_value='https://temp.sh/test/submission.zip'), \
                 patch('w2v_v3184.evaluate.export',side_effect=AssertionError('Inference must not repeat')):
                delivered=finish_request(resumed,cfg,request)
            self.assertEqual(delivered['selected'],b['best_tag']);self.assertTrue(delivered['uploaded'])
            self.assertEqual((root/'automatic-best'/'temp_download_url.txt').read_text().strip(),'https://temp.sh/test/submission.zip')
            self.assertEqual(digest(parent/'last.pt'),original_hash)
            with self.assertRaisesRegex(ValueError,'Expected parent'):inspect_parent(parent,'epoch_999_step_3')
            (parent/'completed.json').unlink()
            with self.assertRaises(FileNotFoundError):inspect_parent(parent)


if __name__=='__main__':unittest.main()
