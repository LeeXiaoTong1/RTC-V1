import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .data import Waves, pair_info, valid_frames
from . import calibration
from .test_integration import fixture
from w2v_v3183r2.common import GROUPS


def row(wave, **kwargs):
    return dict(id='offline/en/Dev_1.wav', source_id='s', group_id='g', language='en', label=1,
        split='train', role='noisy', meta_side='A', wave=wave, original_samples=len(wave),
        pcl_valid_frames=(len(wave)-400)//320+1, v5_origin_split='dev', v5_pcl_eligible=False,
        input_processing=dict(pcl_excluded_spans=[]), **kwargs)


class PairDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_relative_delay_and_original_prefix_mask(self):
        rng = np.random.default_rng(54)
        x = rng.normal(0,.1,6400).astype(np.float32)
        clean, noisy = row(x), row(np.r_[np.zeros(320,np.float32), x[:-320]])
        info = pair_info(clean,noisy)
        self.assertTrue(info['delay']['valid'])
        self.assertEqual(info['shift_frames'],1.)
        self.assertFalse(info['noisy_valid'][0])
        self.assertGreater(info['noisy_valid'].sum(),4)
        short = row(np.tile(x[:1600],4))
        short['original_samples']=1600
        self.assertEqual(int(valid_frames(short).sum()),4)
        short['augmentation']=dict(rtc_activity=dict(suppressed_spans=[[320,800]]))
        self.assertLess(int(valid_frames(short).sum()),4)
        short['role']='clean_ref'
        short['augmentation']['clean_activity']=dict(suppressed_spans=[])
        self.assertEqual(int(valid_frames(short).sum()),4)

    def test_dev_early_pcl_return_still_gets_new_pair_without_phones(self):
        x = np.random.default_rng(7).normal(0,.1,6400).astype(np.float32)
        item=row(x+.01); item['pcl_refs']=dict(clean=row(x)); item['pcl_pairs']={}
        dataset = object.__new__(Waves); dataset.cfg=dict(repair_arm='paired')
        with patch('w2v_v54.data.ParentWaves.__getitem__',return_value=[item]), patch.object(dataset,'_phones',side_effect=AssertionError('Dev must not need CTC')):
            result=dataset[dict(index=0,pcl_enabled=False)]
        self.assertTrue(result[0]['repair_pair']['delay']['valid'])
        self.assertEqual(result[0]['pcl_pairs'],{})
        self.assertFalse(result[0]['v5_pcl_eligible'])

    def test_bad_alignment_keeps_ce_and_rejects_aux_without_wrong_class_reference(self):
        x=np.random.default_rng(8).normal(0,.1,6400).astype(np.float32)
        clean,noisy=row(x),row(np.zeros_like(x))
        info=pair_info(clean,noisy)
        self.assertFalse(info['noisy_valid'].any())
        self.assertEqual(len(noisy['wave']),6400)
        clean['label']=0
        with self.assertRaises(ValueError):pair_info(clean,noisy)

    def test_balanced_scale_preparation_pause_resume_and_cache_reuse(self):
        cfg,units,model,_=fixture()
        cfg.update(repair_scale_sources_per_group=1,repair_scale_recipes=2,workers=0,
            sampling_seed=55,seed=55,repair_min_frames=4)
        rows=[]; all_tickets=[]; mapping={}
        for unit in units:
            if unit[0]['role']=='noisy' and unit[0]['meta_side']=='A':
                i=len(rows);r=unit[0]
                r['pcl_refs']['clean']['pcl_valid_frames']=(len(r['pcl_refs']['clean']['wave'])-400)//320+1
                rows.append(r);mapping[i]=unit
                all_tickets.append(dict(index=i,kind='offline',meta_side='A',pcl_enabled=True))
        class Plan:
            def batches(self, epoch):return [all_tickets]
        class Dataset:
            def __init__(self,rows,settings):pass
            def __getitem__(self,ticket):return copy.deepcopy(mapping[ticket['index']])
        class Loader:
            def __init__(self,dataset,sampler):self.dataset,self.sampler=dataset,sampler
            def __iter__(self):
                for batch in self.sampler.batches:yield [self.dataset[t] for t in batch]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, patch.object(calibration,'Waves',Dataset), patch.object(calibration,'close'), patch.object(calibration,'loader',side_effect=lambda d,c,batch_sampler:Loader(d,batch_sampler)):
            root=Path(directory); run=root/'final'; run.mkdir()
            initial=copy.deepcopy(model.state_dict()); model.repair.scale_ready.fill_(False)
            (root/'STOP_REQUESTED').touch()
            self.assertFalse(calibration.calibrate(model,cfg,rows,Plan(),run))
            self.assertFalse(bool(model.repair.scale_ready))
            (root/'STOP_REQUESTED').unlink()
            calibration.calibrate(model,cfg,rows,Plan(),run)
            expected=model.repair.frame_scale.clone()
            with patch.object(model,'encode',side_effect=AssertionError('completed calibration repeated')):
                model.repair.scale_ready.fill_(False)
                calibration.calibrate(model,cfg,rows,Plan(),run)
            torch.testing.assert_close(model.repair.frame_scale,expected,rtol=0,atol=0)
            for name,value in initial.items():
                if not name.startswith('repair.frame_scale') and not name.startswith('repair.scale_ready'):
                    torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
            import json
            report=json.loads((run/'pair_preflight.json').read_text())
            self.assertEqual(report['pairs'],8)
            self.assertEqual(report['coverage']['origin_dev'],8)
            from .diagnostics import probe
            measured=probe(model,cfg,run,'unchanged_probe')
            self.assertTrue(all(r['reference_drift']==0 for r in measured['records']))
            self.assertEqual(measured['encoder_views'],16)  # Four saved contexts, not different batches.


if __name__=='__main__':unittest.main()
