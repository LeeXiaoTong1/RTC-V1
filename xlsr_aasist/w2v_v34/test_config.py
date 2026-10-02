"""Verify the submitted winner, fallback semantics, and unchanged input recipe."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from w2v_aasist.runtime import atomic_json, sha256
from w2v_v33 import SCHEMA
from w2v_v33.control import quality
from . import config, source


def fixture(root, arm='candidate', tag='epoch_1_step_1209'):
    run = root/'experiment'; folder = run/arm
    folder.mkdir(parents=True)
    baseline = root/'original.pt'; baseline.write_bytes(b'protected original')
    fingerprint = root/'existing_input.json'; fingerprint.write_text('{}')
    data = {str(fingerprint): sha256(fingerprint)}
    cfg = dict(version='3.3', algorithm='w2v-BERT 2.0 + MultiConv; V3.3', full_noisy=True,
        source_run='/retired/v3', arm=arm, arms=[arm], pair_weight=.02,
        baseline=str(baseline), baseline_sha256=sha256(baseline),
        trainable_layers=4, joint_epochs=1, encoder_lr=5e-8, head_lr=2e-6,
        joint_head_lr=2e-6, source_batch=16, seed=1234, cka_weight=.01,
        short_loss_weight=.3, short_min_seconds=3., short_max_seconds=6.,
        pair_warmup_fraction=.1, noisy_weight=.5, noisy_weight_start=.5,
        processing_enabled=False, short_rawboost_probability=.25,
        processing_silence_probability=.05, aux_max_tokens=256,
        legacy_full_cache=str(root/'removed_cache'),
        legacy_train_caches=[str(root/'removed_cache')],
        train_noisy_cache_v33=str(root/'current_cache'), train_caches=[str(root/'current_cache')],
        preparation_fingerprints=data, gpu_activation_gib=18., gpu_reserve_gib=8.,
        checkpointing=True, prefetch_factor=2)
    atomic_json(run/'config.json', cfg)
    arm_cfg = dict(cfg)
    if arm == 'control': arm_cfg['pair_weight'] = 0.
    atomic_json(folder/'config.json', arm_cfg)
    dev = dict(clean_f1=.98, noisy_f1=.95, weighted_f1=.959,
        groups={name+'/en': {'recall': [.98, .8]} for name in ('offline','online','seen','heldout')})
    selected = {'tag': tag, **quality(dev)}
    anchor = dict(selected, tag='baseline')
    atomic_json(folder/'completed.json', dict(selection=dict(best_safe=selected, anchor=anchor)))
    checkpoint = folder/'best_model.pt'
    state = dict(schema=SCHEMA, kind='weights', tag=tag, config=arm_cfg, dev=dev,
        model={'tiny.weight': torch.ones(2)}, model_config={'num_hidden_layers':24},
        baseline_sha256=cfg['baseline_sha256'], data_fingerprints=data)
    torch.save(state, checkpoint)
    atomic_json(run/'comparison.json', dict(status='complete', selected_arm=arm, checkpoint=str(checkpoint),
        arms={arm: dict(selected=selected, anchor=anchor, checkpoint=str(checkpoint))}))
    output = root/(run.name+'_submission'); output.mkdir()
    archive = output/'submission.zip'; archive.write_bytes(b'original exported submission')
    meta = output/'submission_meta.json'
    atomic_json(meta, dict(checkpoint_sha256=sha256(checkpoint), checkpoint_tag=tag,
        protocol_sha256='a'*64, zip_sha256=sha256(archive), count=20000,
        score='P(fake)', threshold=.5, input_policy='full utterance', eval_amp='none'))
    return run, meta, checkpoint, cfg, state


def modify_json(path, update):
    value = json.loads(path.read_text(encoding='utf-8'))
    update(value)
    atomic_json(path, value)


class SourceTests(unittest.TestCase):
    def test_selected_export_is_bound_by_hash_tag_and_all_selection_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run,meta,checkpoint,cfg,state=fixture(root)
            before={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            resolved=source.resolve_source(run,meta)
            self.assertEqual(resolved['checkpoint'],str(checkpoint.resolve()))
            self.assertEqual(resolved['checkpoint_sha256'],sha256(checkpoint))
            self.assertEqual(resolved['config'],state['config'])
            p=resolved['provenance']
            self.assertFalse(p['baseline_fallback']);self.assertTrue(p['submission_zip_verified'])
            self.assertEqual(p['source_data_fingerprints'],state['data_fingerprints'])
            self.assertEqual(p['submission_metadata']['count'],20000)
            self.assertEqual(before,{str(p):sha256(p) for p in root.rglob('*') if p.is_file()})
            self.assertFalse(Path(cfg['legacy_full_cache']).exists())

    def test_default_metadata_location_matches_v33_export_script(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run,meta,*_=fixture(root)
            with patch.object(source,'DEFAULT_SUBMISSION_ROOT',root):
                result=source.resolve_source(run)
            self.assertEqual(result['provenance']['submission_meta'],str(meta.resolve()))

    def test_missing_metadata_requires_original_export_file(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run,meta,*_=fixture(root)
            meta.unlink()
            with self.assertRaisesRegex(FileNotFoundError,'--submission-meta'):
                source.resolve_source(run,meta)

    def test_missing_zip_retains_metadata_binding_but_records_no_zip_verification(self):
        with tempfile.TemporaryDirectory() as td:
            run,meta,*_=fixture(Path(td)); (meta.parent/'submission.zip').unlink()
            result=source.resolve_source(run,meta)
            self.assertFalse(result['provenance']['submission_zip_verified'])

    def test_changed_export_or_metadata_is_rejected(self):
        cases=({'checkpoint_sha256':'0'*64},{'checkpoint_tag':'wrong'}, {'threshold':.6},
               {'score':'P(real)'},{'eval_amp':'bf16'}, {'count':0},{'zip_sha256':'bad'})
        for changes in cases:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as td:
                run,meta,*_=fixture(Path(td))
                modify_json(meta,lambda d:d.update(changes))
                with self.assertRaises(ValueError): source.resolve_source(run,meta)
        with tempfile.TemporaryDirectory() as td:
            run,meta,*_=fixture(Path(td));(meta.parent/'submission.zip').write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError,'submission.zip'):
                source.resolve_source(run,meta)

    def test_changed_checkpoint_bytes_are_rejected_against_original_export(self):
        with tempfile.TemporaryDirectory() as td:
            run,meta,checkpoint,_,state=fixture(Path(td));state['model']['tiny.weight'].add_(1)
            torch.save(state,checkpoint)
            with self.assertRaisesRegex(ValueError,'SHA256'):
                source.resolve_source(run,meta)

    def test_even_matching_export_cannot_adopt_foreign_or_modified_checkpoint(self):
        cases=({'schema':'rtc_w2v_multiconv_v3'},{'kind':'training'}, {'tag':'wrong'})
        for changes in cases:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as td:
                run,meta,checkpoint,_,state=fixture(Path(td));state.update(changes)
                torch.save(state,checkpoint)
                modify_json(meta,lambda d:d.update(checkpoint_sha256=sha256(checkpoint)))
                with self.assertRaisesRegex(ValueError,'schema/kind/tag'):
                    source.resolve_source(run,meta)

    def test_completed_comparison_must_match_actual_selected_winner(self):
        for field,value in (('status','training'),('selected_arm','unknown'),('checkpoint','other.pt')):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as td:
                run,meta,*_=fixture(Path(td))
                modify_json(run/'comparison.json',lambda d:d.update({field:value}))
                with self.assertRaises(ValueError):source.resolve_source(run,meta)
        with tempfile.TemporaryDirectory() as td:
            run,meta,*_=fixture(Path(td))
            modify_json(run/'candidate'/'completed.json',lambda d:d['selection']['best_safe'].update(tag='stale'))
            with self.assertRaisesRegex(ValueError,'completed arm selection'):
                source.resolve_source(run,meta)

    def test_checkpoint_metrics_and_config_must_match_saved_selection_and_recipe(self):
        for changed in ('dev','config','data_fingerprints'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as td:
                run,meta,checkpoint,_,state=fixture(Path(td))
                if changed=='dev': state['dev']['weighted_f1']=.999
                elif changed=='config':state['config']['pair_weight']=.03
                else: state['data_fingerprints']={'other':'b'*64}
                torch.save(state,checkpoint)
                modify_json(meta,lambda d:d.update(checkpoint_sha256=sha256(checkpoint)))
                with self.assertRaises(ValueError):source.resolve_source(run,meta)
        with tempfile.TemporaryDirectory() as td:
            run,meta,*_=fixture(Path(td))
            modify_json(run/'config.json',lambda d:d.update(short_loss_weight=.5))
            with self.assertRaisesRegex(ValueError,'root recipe'):source.resolve_source(run,meta)


class ConfigTests(unittest.TestCase):
    def test_defaults_choose_exact_submitted_run_and_all_encoder_blocks(self):
        args=config.parser().parse_args([])
        self.assertEqual(Path(args.source_run).name,'w2v_v33_20261002_004018_198a')
        self.assertEqual((args.trainable_layers,args.encoder_lr,args.head_lr,args.layer_decay,args.epochs),
                         (24,5e-7,2e-6,.9,2))
        self.assertIsNone(args.submission_meta)

    def test_candidate_data_and_loss_recipe_are_inherited_without_rebuilding_old_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,meta,_,old,_=fixture(root)
            args=config.parser().parse_args(['--source-run',str(run),'--submission-meta',str(meta)])
            with patch.object(config,'BASELINE_SHA256',old['baseline_sha256']):
                new=config.configuration(args)
            for key in ('pair_weight','pair_warmup_fraction','cka_weight','noisy_weight',
                        'short_loss_weight','source_batch','seed','train_caches','train_noisy_cache_v33',
                        'short_rawboost_probability','processing_silence_probability','aux_max_tokens',
                        'short_min_seconds','short_max_seconds','processing_enabled'):
                self.assertEqual(new[key],old[key],key)
            self.assertEqual(new['version'],'3.4');self.assertEqual(new['joint_epochs'],2)
            self.assertEqual(new['source_data_fingerprints'],old['preparation_fingerprints'])
            self.assertEqual(new['arm'],'candidate');self.assertEqual(new['arms'],['candidate'])
            self.assertEqual(new['prefetch_factor'],1)
            self.assertEqual((new['gpu_activation_gib'],new['gpu_reserve_gib']),(18.,8.))
            self.assertFalse(Path(old['legacy_full_cache']).exists())

    def test_selected_control_baseline_does_not_silently_enable_pair_loss(self):
        with tempfile.TemporaryDirectory() as td:
            run,meta,_,old,_=fixture(Path(td),arm='control',tag='baseline')
            args=config.parser().parse_args(['--source-run',str(run),'--submission-meta',str(meta)])
            with patch.object(config,'BASELINE_SHA256',old['baseline_sha256']):
                new=config.configuration(args)
            self.assertEqual(new['pair_weight'],0.);self.assertEqual(new['expected_warm_tag'],'baseline')
            self.assertTrue(new['source_provenance']['baseline_fallback'])
            self.assertEqual(new['arm'],'control');self.assertEqual(new['arms'],['control'])

    def test_invalid_optimization_flags_fail_before_source_io(self):
        cases=(['--encoder-lr','nan'],['--head-lr','0'],['--layer-decay','nan'],
               ['--layer-decay','0'],['--layer-decay','1.01'],['--workers','-1'],['--smoke-steps','-1'])
        for flags in cases:
            with self.subTest(flags=flags),patch.object(config,'resolve_source',side_effect=AssertionError('source I/O')):
                with self.assertRaises(ValueError):config.configuration(config.parser().parse_args(flags))

    def test_original_protected_checkpoint_is_still_required(self):
        with tempfile.TemporaryDirectory() as td:
            run,meta,_,old,_=fixture(Path(td))
            Path(old['baseline']).write_bytes(b'changed original')
            args=config.parser().parse_args(['--source-run',str(run),'--submission-meta',str(meta)])
            with patch.object(config,'BASELINE_SHA256',old['baseline_sha256']):
                with self.assertRaisesRegex(ValueError,'Protected original'):config.configuration(args)


if __name__=='__main__':unittest.main()
