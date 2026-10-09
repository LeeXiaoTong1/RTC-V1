import json
from pathlib import Path
import tempfile
import unittest

import torch
import cleanup_early_checkpoints as cleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.targets = []
        for run in cleanup.TARGETS:
            folder = self.root/'exp'/run
            folder.mkdir(parents=True)
            torch.save({'resume':True},folder/'last.pt')
            torch.save({'standalone':True},folder/'best_model.pt')
            self.targets.append(folder/'last.pt')

    def validator(self,path,schema): return {'sha256':cleanup.sha(path),'schema':schema}

    def plan(self):return cleanup.plan(self.root,self.validator)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_only_two_targets_removed_and_all_other_files_unchanged(self):
        extra=self.root/'exp'/'w2v_v315_important'/'last.pt'
        extra.parent.mkdir()
        torch.save({'best_inside':True},extra)
        self.write(self.root/'data'/'manifest.json',{'original':True})
        before={p:cleanup.sha(p) for p in self.root.rglob('*') if p.is_file()}
        plan=self.plan()
        self.assertEqual({Path(r['path']) for r in plan['candidates']},set(self.targets))
        cleanup.apply(plan)
        for p,h in before.items():
            if p in self.targets:self.assertFalse(p.exists())
            else:self.assertEqual(cleanup.sha(p),h)
        self.assertEqual(self.plan()['candidates'],[])

    def test_nested_json_and_embedded_checkpoint_references_keep_targets(self):
        self.write(self.root/'exp'/'later'/'stages'/'selection.json',{'parent':str(self.targets[0])})
        torch.save({'config':{'source':str(self.targets[1])}},self.root/'exp'/'later'/'best.pt')
        plan=self.plan()
        self.assertFalse(plan['candidates'])
        self.assertEqual(len(plan['retained']),2)

    def test_directory_dependency_protects_last(self):
        self.write(self.root/'exp'/'later'/'config.json',{'source_run':str(self.targets[0].parent)})
        self.assertIn(str(self.targets[0]),{r['path'] for r in self.plan()['retained']})

    def test_inventory_reference_does_not_pin_resume_state(self):
        self.write(self.root/'exp'/'maintenance_old.json',{'inventory':[str(p) for p in self.targets]})
        self.assertEqual(len(self.plan()['candidates']),2)

    def test_missing_best_stops_before_deletion(self):
        (self.targets[0].parent/'best_model.pt').unlink()
        with self.assertRaisesRegex(ValueError,'missing'):self.plan()
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_changed_target_cancels_all_deletions(self):
        plan=self.plan()
        self.targets[1].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Candidate changed'):cleanup.apply(plan)
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_unrecognized_checkpoint_fails_closed(self):
        (self.root/'exp'/'unknown.pt').write_bytes(b'not a readable checkpoint')
        with self.assertRaisesRegex(ValueError,'legacy checkpoint'):self.plan()
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_changed_best_and_outside_allowlist_are_rejected(self):
        plan=self.plan()
        plan['candidates'][0]['path']=str(self.targets[0].parent/'best_model.pt')
        with self.assertRaisesRegex(ValueError,'allowlist'):cleanup.apply(plan)
        plan=self.plan()
        (self.targets[0].parent/'best_model.pt').write_bytes(b'changed best')
        with self.assertRaisesRegex(ValueError,'changed'):cleanup.apply(plan)
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_actual_standalone_best_architecture_and_missing_tensor(self):
        from transformers import Wav2Vec2BertConfig
        from w2v_v3.model import Detector
        cfg=Wav2Vec2BertConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
            intermediate_size=32,feature_projection_input_dim=160,conv_depthwise_kernel_size=7,
            num_conv_pos_embedding_groups=2)
        model=Detector.from_config(cfg.to_dict(),dict(input_dim=16,projection=8,expansion=32,
            blocks=4,kernels=(3,7),merge_kernel=3,dropout=.1),checkpointing=False)
        schema=next(iter(cleanup.TARGETS.values()))
        state=dict(schema=schema,kind='weights',model=model.state_dict(),**model.architecture())
        best=self.targets[0].parent/'best_model.pt'
        torch.save(state,best)
        self.assertEqual(cleanup.validate_best(best,schema)['sha256'],cleanup.sha(best))
        state['model'].pop(next(iter(state['model'])))
        torch.save(state,best)
        with self.assertRaisesRegex(ValueError,'complete architecture'):cleanup.validate_best(best,schema)


if __name__=='__main__':unittest.main()
