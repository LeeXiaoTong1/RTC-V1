import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_v3184.test_workflow import completed_r2
from w2v_v318.test_core import NativeBoundary
from w2v_v3183r2.monitor import measure
from w2v_v3183r2.sampling import Plan
from .common import atomic_json,read_json,digest
from .seal_parent import seal,validate_state,authenticate_supervisor
from .parent import inspect_parent


def interrupted_parent(root):
    parent,cfg,train,dev,_=completed_r2(root)
    state=torch.load(parent/'last.pt',weights_only=True)
    ix=read_json(parent/'dev_partition.json')['select'];rows=[dev[i] for i in ix]
    best=torch.full((len(dev),2),-2.);last=torch.zeros_like(best)
    for i,r in enumerate(dev):best[i,r['label']]=2.
    steps=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas']).steps
    first=measure(rows,best.numpy()[ix]);second=measure(rows,last.numpy()[ix])
    state.update(epoch=2,cursor=steps*2,best_metrics=first,best_logits=best,last_logits=last,
                 history=[dict(epoch=1,tag='epoch_1_step_'+str(steps),selection_metrics=first),
                          dict(epoch=2,tag='epoch_2_step_'+str(steps),selection_metrics=second)])
    torch.save(state,parent/'last.pt');(parent/'completed.json').unlink()
    return parent,cfg,state


class SealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_sealed_best_original_unchanged_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d,patch('w2v_v3184.data.Engines',NativeBoundary),redirect_stdout(io.StringIO()):
            root=Path(d);parent,cfg,state=interrupted_parent(root)
            hashes={p.name:digest(p) for p in parent.iterdir() if p.is_file()}
            pointer=root/'pointer';out=seal(parent,verify=False,pointer=pointer)
            self.assertNotEqual(out,parent);self.assertFalse((parent/'completed.json').exists())
            self.assertEqual(pointer.read_text().strip(),str(out))
            for n,h in hashes.items():self.assertEqual(digest(parent/n),h)
            _,values,meta=inspect_parent(out)
            self.assertEqual(meta['tag'],state['best_tag']);self.assertNotEqual(meta['tag'],state['last_tag'])
            self.assertEqual(digest(out/'last.pt'),hashes['last.pt'])
            self.assertEqual(read_json(out/'calibration_best.json')['selected'],state['best_tag'])
            self.assertEqual(seal(parent,verify=False),out)
            (out/'dev_rows.json').write_text('[]')
            with self.assertRaises(Exception):seal(parent,verify=False)

    def test_reject_partial_cursor_wrong_best_and_bad_logits(self):
        with tempfile.TemporaryDirectory() as d,patch('w2v_v3184.data.Engines',NativeBoundary),redirect_stdout(io.StringIO()):
            parent,cfg,state=interrupted_parent(Path(d))
            validate_state(parent,cfg,state)
            broken=copy.deepcopy(state);broken['cursor']+=1
            with self.assertRaisesRegex(ValueError,'cursor/history'):validate_state(parent,cfg,broken)
            broken=copy.deepcopy(state);broken['best_tag']=state['last_tag']
            with self.assertRaisesRegex(ValueError,'best committed'):validate_state(parent,cfg,broken)
            broken=copy.deepcopy(state);broken['best_logits'][0,0]=float('nan')
            with self.assertRaisesRegex(ValueError,'committed scores'):validate_state(parent,cfg,broken)
            broken=copy.deepcopy(state);broken['best_logits'].zero_()
            with self.assertRaisesRegex(ValueError,'disagree'):validate_state(parent,cfg,broken)

    def test_process_identity_bound_to_script_run_log_and_session(self):
        # Synthetic /proc; no signal is ever sent by this test.
        with tempfile.TemporaryDirectory() as d:
            proc=Path(d);entry=proc/'123';root=entry/'cwd';root.mkdir(parents=True);(root/'exp').mkdir()
            run=root/'exp'/'r2';log=root/'exp'/'r2.log'
            for name,value in [('run',run),('log',log)]:
                (root/'exp'/('.latest_v3183r2_'+name)).write_text(str(value))
            args=['bash',str(root/'v3183r2_supervise.sh'),str(log)]
            (entry/'cmdline').write_bytes(('\0'.join(args)+'\0').encode());log.write_text('V3183R2_RUN='+str(run)+'\n')
            with patch('os.getuid',return_value=entry.stat().st_uid,create=True),patch('os.getpgid',return_value=123,create=True):
                self.assertEqual(authenticate_supervisor(root,run,123,proc),args)
                (entry/'cmdline').write_bytes(b'bash\0another_script.sh\0')
                with self.assertRaisesRegex(ValueError,'Refusing to signal'):authenticate_supervisor(root,run,123,proc)
                (entry/'cmdline').write_bytes(('\0'.join(args)+'\0').encode());log.write_text('different-run')
                with self.assertRaisesRegex(ValueError,'not bound'):authenticate_supervisor(root,run,123,proc)


if __name__=='__main__':unittest.main()
