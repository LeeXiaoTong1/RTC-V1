import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import cleanup_for_v318 as cleanup


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.cache=self.root/cleanup.TRAIN_CACHE
        self.wave=self.cache/'audio'/'ab'/'generated_noisy_a.wav'
        self.wave.parent.mkdir(parents=True)
        self.wave.write_bytes(b'generated wave')
        self.write(self.wave.with_suffix('.json'),{'metadata':'keep'})
        self.write(self.cache/'config.json',{'format':'rtc_v33_full_condition_cache_v1','role':'train'})
        self.manifest=self.cache/'manifest.jsonl'
        self.manifest.write_text(json.dumps({'audio':'audio/ab/generated_noisy_a.wav','condition':'noisy_a'})+'\n')
        self.complete()
        self.rolling=self.root/cleanup.ROLLING_CACHE
        self.rolling.mkdir(parents=True)
        cfg={'rolling_cache_bytes':6000}
        self.write(self.rolling.parent/'config.json',cfg)
        self.write(self.rolling/'owner.json',dict(format='rtc_v315_bounded_pairs_v1',cap_bytes=6000,
            identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()))
        self.pair=self.rolling/('a'*64+'.npz')
        self.pair.write_bytes(b'regenerable paired array')
        self.best=self.rolling.parent/'last.pt'
        torch.save({'selected_best_inside':True},self.best)

    def write(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value),encoding='utf-8')

    def complete(self):
        self.write(self.cache/'complete.json',dict(config_sha256=cleanup.sha(self.cache/'config.json'),
            manifest_sha256=cleanup.sha(self.manifest)))

    def plan(self):
        candidates,metadata=cleanup.cache_plan(self.root,{})
        return dict(root=str(self.root),candidates=candidates,metadata=metadata,checked_files={},verified_best={})

    def test_only_payloads_deleted_all_metadata_and_selected_weights_survive(self):
        original={p:cleanup.sha(p) for p in self.root.rglob('*') if p.is_file()}
        plan=self.plan()
        result=cleanup.apply(plan,self.root/'receipt.jsonl')
        self.assertEqual(result['files_deleted'],2)
        for p,h in original.items():
            if p in (self.wave,self.pair):self.assertFalse(p.exists())
            else:self.assertEqual(cleanup.sha(p),h)
        self.assertEqual(self.plan()['candidates'],[])

    def test_current_input_path_or_root_is_never_deleted(self):
        for protected in ({self.wave:set()},{self.cache:set()}):
            with self.assertRaisesRegex(ValueError,'Current V3.18 input'):cleanup.cache_plan(self.root,protected)
        self.assertTrue(self.wave.exists())

    def test_foreign_cache_and_changed_manifest_rejected(self):
        cfg=json.loads((self.cache/'config.json').read_text())
        cfg['role']='dev';self.write(self.cache/'config.json',cfg);self.complete()
        with self.assertRaisesRegex(ValueError,'ownership'):self.plan()
        self.assertTrue(self.wave.exists())

    def test_manifest_path_escape_and_unknown_rolling_content_rejected(self):
        original=self.manifest.read_text()
        self.manifest.write_text(json.dumps({'audio':'../../official.wav','condition':'noisy_a'})+'\n')
        self.complete()
        with self.assertRaisesRegex(ValueError,'Unexpected generated'):self.plan()
        self.manifest.write_text(original);self.complete()
        (self.rolling/'valuable.pt').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'Unexpected rolling'):self.plan()

    def test_file_and_manifest_changes_stop_before_any_unlink(self):
        plan=self.plan()
        self.pair.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'target changed'):cleanup.apply(plan,self.root/'receipt.jsonl')
        self.assertTrue(self.wave.exists())
        plan=self.plan()
        self.manifest.write_text('changed')
        with self.assertRaisesRegex(ValueError,'metadata changed'):cleanup.apply(plan,self.root/'receipt.jsonl')
        self.assertTrue(self.wave.exists())

    def test_nonallowlisted_checkpoint_rejected(self):
        plan=self.plan()
        plan['candidates'].append(dict(path=str(self.best),identity=cleanup.early.stamp(self.best),kind='checkpoint'))
        with self.assertRaisesRegex(ValueError,'outside allowlist'):cleanup.apply(plan,self.root/'receipt.jsonl')
        self.assertTrue(self.best.exists());self.assertTrue(self.wave.exists())

    def test_retained_candidate_transitively_protects_another_candidate(self):
        runs=list(cleanup.early.TARGETS)
        targets=[]
        for run in runs:
            path=self.root/'exp'/run/'last.pt';path.parent.mkdir(parents=True)
            torch.save({},path);torch.save({},path.parent/'best_model.pt');targets.append(path)
        torch.save({'parent_checkpoint':str(targets[1])},targets[0])
        self.write(self.root/'exp'/'later'/'config.json',{'parent_checkpoint':str(targets[0])})
        candidates,kept,_,_=cleanup.checkpoint_plan(self.root,lambda p,s:{'sha256':cleanup.sha(p)})
        self.assertFalse(candidates)
        self.assertEqual({r['path'] for r in kept},{str(p) for p in targets})

    def test_checkpoint_failure_does_not_discard_safe_cache_plan(self):
        run=next(iter(cleanup.RESUMES))
        p=self.root/'exp'/run/'last.pt';p.parent.mkdir(parents=True);torch.save({},p)
        candidates,kept,_,_=cleanup.checkpoint_plan(self.root)
        self.assertFalse(candidates);self.assertEqual(len(kept),1)
        self.assertEqual(len(self.plan()['candidates']),2)

    def test_aasist_best_is_checked_against_real_architecture(self):
        from w2v_aasist.tests import tiny_detector
        model=tiny_detector()
        path=self.root/'aasist_best.pt'
        torch.save(dict(schema='rtc_w2v_aasist_full_v1',kind='weights',model=model.state_dict(),
            model_config=model.backbone.config.to_dict()),path)
        self.assertEqual(cleanup.validate_selected(path,'rtc_w2v_aasist_full_v1')['sha256'],cleanup.sha(path))

    def test_v313_best_survives_via_verified_starting_checkpoint(self):
        run=self.root/'exp'/'v313_test';run.mkdir()
        base=self.root/'base.pt';base.write_bytes(b'base')
        start=self.root/'start.pt';start.write_bytes(b'start')
        cfg=dict(base_checkpoint=str(base),base_checkpoint_sha256=cleanup.sha(base),
                 starting_checkpoint=str(start),starting_checkpoint_sha256=cleanup.sha(start))
        state=dict(schema='rtc_v313_partial_state_v1',config=cfg,model=None,selected='baseline',
            identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
        torch.save(state,run/'best.pt')
        self.write(run/'completed.json',dict(version='3.13',status='complete',selected='baseline',
            checkpoint_sha256=cleanup.sha(run/'best.pt'),base_checkpoint_sha256=cleanup.sha(base),
            starting_checkpoint_sha256=cleanup.sha(start),baseline_fallback=True))
        info=cleanup.validate_selected(run/'best.pt','rtc_v313_partial_state_v1')
        self.assertEqual(set(info['dependencies']),{str(base),str(start)})
        start.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'dependency changed'):
            cleanup.validate_selected(run/'best.pt','rtc_v313_partial_state_v1')


if __name__=='__main__':unittest.main()
