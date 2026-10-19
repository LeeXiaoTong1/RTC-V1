from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
from torch import nn
from w2v_v31810.common import atomic_json,read_json,digest,fingerprint
from w2v_v3187.phonemes import (SCHEMA,MODEL_ID,MODEL_REVISION,ALGORITHM,inventory,lineage,record_name,digest_json)
from .phonemes import build_pairs
from .data import Waves
from .teacher import Teacher
from .diagnostics import compare


def phone_fixture(root):
    rows=[]
    for lang in ('en','zh'):
        for label in (0,1):
            off=f'{lang}{label}-offline';on=f'{lang}{label}-online'
            group=None
            for condition,id in (('offline',off),('online',on)):
                path=root/(id+'.wav');wave=np.random.default_rng(len(rows)).normal(0,.1,4240).astype(np.float32)
                sf.write(path,wave,16000,subtype='PCM_16');sha=digest(path)
                if group is None:group=sha
                rows.append(dict(id=id,source_id=off,group_id=group,audio=str(path),audio_sha256=sha,
                    output_samples=len(wave),full_length=True,canonical_source=off,language=lang,label=label,split='train',condition=condition))
    records={}
    for row in rows:
        rec=dict(schema=SCHEMA,identity=lineage(row),model_id=MODEL_ID,model_revision=MODEL_REVISION,algorithm=ALGORITHM,
            segments=[dict(instance=0,token='a',start_sample=0,end_sample=3000,support_start_sample=200,support_end_sample=2960,posterior=.9)],
            quality={})
        file=root/record_name(row);atomic_json(file,rec);records[row['id']]=dict(path=record_name(row),sha256=digest(file),segments=1)
    items=inventory(rows);atomic_json(root/'manifest.json',dict(schema=SCHEMA,model_id=MODEL_ID,model_revision=MODEL_REVISION,
        algorithm=ALGORITHM,inventory=items,inventory_sha256=digest_json(items),complete=True,records=records))
    return rows


class Block(nn.Module):
    def __init__(self,i):super().__init__();self.i=i;self.calls=0
    def forward(self,x,mask):self.calls+=1;return x+(self.i+1)*.001,mask


class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Module();self.encoder.encoders=nn.ModuleList([Block(i) for i in range(19)])
    def inference_encode(self,wave,lengths,use_mask=False,use_final_output=False):
        assert use_mask is False and use_final_output is False
        n=(wave.shape[1]-400)//320+1;x=torch.zeros(1,n,1024);mask=None;states=[]
        for layer in self.encoder.encoders:x,mask=layer(x,mask);states.append(x)
        return x,states,lengths


class PreparationTests(unittest.TestCase):
    def test_real_pcm_phone_manifest_and_original_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=phone_fixture(root)
            pairs=build_pairs(rows,root/'manifest.json',root/'pairs.json')
            self.assertEqual(len(pairs['pairs']),4)
            self.assertEqual(pairs,build_pairs(rows,root/'manifest.json',root/'pairs.json'))
            settings=dict(seed=1,raw_audio_cache_mib=1)
            dataset=Waves(rows,settings);p=pairs['pairs'][rows[1]['id']]
            unit=dataset[dict(index=1,kind='online',epoch=1,occurrence='test',official_pcl=p)]
            refs=unit[0]['original_pcl']['refs']
            self.assertEqual([r['condition'] for r in refs],['offline','online'])
            self.assertTrue(all('augmentation' not in r for r in refs))
            changed=read_json(root/'pairs.json');changed['pairs'][rows[1]['id']]['offline']='bad';atomic_json(root/'pairs.json',changed)
            with self.assertRaises(ValueError):build_pairs(rows,root/'manifest.json',root/'pairs.json')

    def test_teacher_official_tuple_layout_truncates_and_cleans_hook(self):
        teacher=Teacher.__new__(Teacher);teacher.device=torch.device('cpu');teacher.model=TeacherModel();teacher.verified=False
        result=teacher(np.ones(6400,np.float32));self.assertTrue(teacher.verified)
        self.assertEqual(result.shape,((6400-400)//320+1,1024))
        np.testing.assert_allclose(result,.055,atol=1e-7)
        # First full pass validates the shortcut. Later calls stop after layer 10.
        before=[m.calls for m in teacher.model.encoder.encoders];teacher(np.ones(6400,np.float32))
        after=[m.calls for m in teacher.model.encoder.encoders]
        self.assertEqual([b-a for a,b in zip(before,after)],[1]*10+[0]*9)
        self.assertFalse(teacher.model.encoder.encoders[9]._forward_hooks)
        self.assertEqual(teacher(np.ones(300,np.float32)).shape,(0,1024))

    def test_diagnostics_do_not_change_deployed_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);n=8;labels=np.tile([0,0,1,1],2)
            common=dict(ids=np.array([str(i) for i in range(n)]),canonical_source=np.array([str(i) for i in range(n)]),
                language=np.array(['en']*4+['zh']*4),panel=np.array(['official_online']*n),label=labels)
            a=np.array([2,-.1,.1,-2]*2);b=np.array([2,-.2,-.1,-2]*2)
            np.savez(root/'a.npz',**common,logits=np.stack((a,np.zeros(n)),1))
            np.savez(root/'b.npz',**common,logits=np.stack((b,np.zeros(n)),1))
            result=compare(root/'a.npz',root/'b.npz')
            self.assertEqual(result['corrected'],2);self.assertEqual(result['regressed'],0)
            self.assertEqual(result['deployed_threshold'],.5);self.assertFalse(result['threshold_fit'])

if __name__=='__main__':unittest.main()
