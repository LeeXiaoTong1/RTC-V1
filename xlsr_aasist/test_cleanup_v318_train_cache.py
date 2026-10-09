import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cleanup_v318_train_cache as cleanup


class TrainPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.a=self.root/cleanup.V33;self.b=self.root/cleanup.ROLLING
        self.write(self.a/'config.json',{'format':'rtc_v33_full_condition_cache_v1','role':'train'})
        cfg={'rolling_cache_bytes':123}
        self.write(self.b.parent/'config.json',cfg)
        self.write(self.b/'owner.json',{'format':'rtc_v315_bounded_pairs_v1','cap_bytes':123,
            'identity':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()})
        self.wave=self.a/'audio'/'aa'/('a'*24+'_noisy_a.wav')
        self.wave.parent.mkdir(parents=True);self.wave.write_bytes(b'generated')
        self.pair=self.b/('b'*64+'.npz');self.pair.write_bytes(b'generated pair')
        self.write(self.wave.with_suffix('.json'),{'recipe':'preserve'})
        self.best=self.b.parent/'last.pt';self.best.write_bytes(b'best inside')
        self.other=self.a/'original.wav';self.other.write_bytes(b'not a generated name')

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_deletes_only_owned_payloads_preserves_all_metadata_and_models(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        plan=cleanup.inventory(self.root,{})
        self.assertEqual(cleanup.apply(plan),2)
        for p,b in before.items():
            if p in (self.wave,self.pair):self.assertFalse(p.exists())
            else:self.assertEqual(p.read_bytes(),b)
        self.assertEqual(cleanup.inventory(self.root,{})['candidates'],[])

    def test_current_input_parent_child_and_exact_match_block_deletion(self):
        for p in (self.wave,self.a,self.a.parent):
            with self.assertRaisesRegex(ValueError,'input intersects'):cleanup.inventory(self.root,{p:{'input'}})
        self.assertTrue(self.wave.exists())

    def test_owner_mismatch_and_changed_payload_fail_closed(self):
        plan=cleanup.inventory(self.root,{})
        self.wave.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Payload changed'):cleanup.apply(plan)
        self.assertTrue(self.pair.exists())
        self.write(self.b/'owner.json',{})
        with self.assertRaisesRegex(ValueError,'ownership'):cleanup.inventory(self.root,{})

    def test_allowlist_never_deletes_checkpoint(self):
        plan=cleanup.inventory(self.root,{})
        plan['candidates'].append({'path':str(self.best),'identity':cleanup.identity(self.best)})
        with self.assertRaisesRegex(ValueError,'allowlist'):cleanup.apply(plan)
        self.assertTrue(self.wave.exists());self.assertTrue(self.best.exists())


if __name__=='__main__':unittest.main()
