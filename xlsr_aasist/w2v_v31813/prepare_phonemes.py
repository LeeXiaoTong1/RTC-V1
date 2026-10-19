"""Prepare compact Train phoneme metadata once; never create audio/features.

Downloads only the pinned public phoneme model into --model-cache. All source
audio and inference remain local. Approximate CTC intervals are not advertised
as manually annotated phoneme boundaries. No evaluation audio is accepted.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time

import numpy as np

from w2v_v3187.phonemes import (SCHEMA, MODEL_ID, MODEL_REVISION, SAMPLE_RATE, ALGORITHM,
    digest_file, digest_json, read_json, atomic_json, lineage, inventory,
    record_name, validate_record, bind_manifest, greedy_tokens, ctc_align, segment_support, SHORT_REASON)
from .source import resolve_data_run
from w2v_v3186.phoneme_download import ensure_model,add_download_arguments,ModelDownloadError


class Recognizer:
    def __init__(self, model_cache, device, local_files_only=False, *, hf_endpoint='auto', download_retries=2):
        import torch
        from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModelForCTC
        self.torch = torch; self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA phoneme preparation but CUDA is unavailable')
        local_model=ensure_model(model_cache,local_files_only,hf_endpoint,download_retries)
        options = dict(local_files_only=True, trust_remote_code=False)
        self.extractor = AutoFeatureExtractor.from_pretrained(str(local_model), **options)
        # Only ID->phone decoding is needed. No text phonemization and no
        # espeak/phonemizer system dependency is required for this task.
        self.tokenizer = AutoTokenizer.from_pretrained(str(local_model), do_phonemize=False, **options)
        self.model = AutoModelForCTC.from_pretrained(str(local_model), use_safetensors=False, **options).to(self.device).eval()
        self.model.requires_grad_(False)
        self.blank = int(self.model.config.pad_token_id)
        self.ignore = set(self.tokenizer.all_special_ids)
        for name in ('word_delimiter_token_id', 'phone_delimiter_token_id'):
            value = getattr(self.tokenizer, name, None)
            if isinstance(value, int): self.ignore.add(value)
        self.vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        stride = 1; receptive = 1
        for kernel, step in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            receptive += (int(kernel) - 1) * stride; stride *= int(step)
        self.stride, self.receptive = stride, receptive
        if self.extractor.sampling_rate != SAMPLE_RATE or stride != 320 or receptive != 400:
            raise ValueError('Pinned phoneme model unexpectedly changed frame geometry')

    def __call__(self, wave):
        torch = self.torch; result = []; dropped_boundary = 0; decoded = 0
        core = ALGORITHM['chunk_samples']; context = ALGORITHM['context_samples']
        # Bounded chunks avoid quadratic attention memory for rare long files.
        # Context overlaps, but retained ownership intervals do not. A phone
        # crossing a chunk boundary is masked instead of being duplicated.
        for start in range(0, len(wave), core):
            stop = min(start + core, len(wave)); left = max(0, start - context); right = min(len(wave), stop + context)
            chunk = wave[left:right]
            if len(chunk) < self.receptive: chunk = np.pad(chunk, (0, self.receptive - len(chunk)))
            values = self.extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors='pt')
            values = {k: v.to(self.device) for k, v in values.items()}
            with torch.inference_mode():
                # Use one FP32 preparation policy, independent of detector AMP.
                # This does not promise bitwise equality across GPU generations.
                logits = self.model(**values).logits[0].float()
                log_probs = logits.log_softmax(-1).cpu().numpy()
            tokens = greedy_tokens(log_probs, self.blank, self.ignore); decoded += len(tokens)
            support = ctc_align(log_probs, tokens, self.blank)
            segments = segment_support(support, self.vocab, right - left, self.stride, self.receptive)
            for s in segments:
                s = dict(s)
                for key in ('start_sample', 'end_sample', 'support_start_sample', 'support_end_sample'): s[key] += left
                if s['start_sample'] < start or s['end_sample'] > stop:
                    dropped_boundary += 1; continue
                s['instance'] = len(result); result.append(s)
        return result, dict(decoded_tokens=decoded, kept_segments=len(result),
                            chunk_boundary_dropped=dropped_boundary,
                            approximate_boundaries=True, recognition_input='original_train_audio')


def _read_verified_wave(row):
    import soundfile as sf
    path = Path(row['audio']).expanduser().resolve()
    if digest_file(path) != row['audio_sha256']:
        raise ValueError('Original waveform SHA256 changed: ' + str(path))
    wave, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != SAMPLE_RATE or wave.shape != (row['output_samples'], 1) or not np.isfinite(wave).all():
        raise ValueError('Require original verified 16 kHz mono waveform: ' + str(path))
    return wave[:, 0]


def prepare(parent_run, output_dir, model_cache, device='cuda:0', local_files_only=False,
            recognizer_factory=None, *, hf_endpoint='auto', download_retries=2):
    if recognizer_factory is None:
        from functools import partial
        recognizer_factory=partial(Recognizer,hf_endpoint=hf_endpoint,download_retries=download_retries)
    parent = resolve_data_run(parent_run); root = Path(output_dir).expanduser().resolve()
    rows_path = parent / 'train_rows.json'; rows = read_json(rows_path); items = inventory(rows)
    model_cache = Path(model_cache).expanduser().resolve(); model_cache.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    # Explicit paths keep the only large download away from the system disk.
    os.environ['HF_HOME'] = str(model_cache)
    os.environ['HF_HUB_CACHE'] = str(model_cache)
    os.environ['TRANSFORMERS_CACHE'] = str(model_cache)
    owner = dict(schema=SCHEMA, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                 algorithm=ALGORITHM, inventory_sha256=digest_json(items), inventory=items,
                 data_role='original_train_offline_and_online_only',
                 storage='phoneme_json_only_no_audio_or_detector_features')
    if (root / 'owner.json').exists():
        if read_json(root / 'owner.json') != owner:
            raise ValueError('Existing phoneme cache belongs to a different Train inventory/algorithm')
    else: atomic_json(root / 'owner.json', owner)
    completed = None
    if (root / 'manifest.json').is_file():
        bind_manifest(root / 'manifest.json', rows)
        completed = read_json(root / 'manifest.json')
    recognizer = None; entries = {}; counts = Counter(); started = time.monotonic()
    for i, row in enumerate(sorted(rows, key=lambda r: r['id'])):
        target = root / record_name(row)
        # Reverify original bytes even on resumed preparation. This is a single
        # preparation-time scan, not a per-epoch training IO requirement.
        if target.is_file():
            target_hash=digest_file(target)
            if completed is not None and target_hash != completed['records'][row['id']]['sha256']:
                raise ValueError('Completed phoneme metadata checksum changed: ' + row['id'])
            if digest_file(row['audio']) != row['audio_sha256']:
                raise ValueError('Original waveform SHA256 changed: ' + row['id'])
            rec = validate_record(read_json(target), row); counts['reused'] += 1
        else:
            wave = _read_verified_wave(row)
            if len(wave)<400:
                # Do not pad artificial audio and hallucinate CTC phone spans.
                # Still verify bytes and publish a hashed empty source record.
                segments=[]
                quality=dict(reason=SHORT_REASON,decoded_tokens=0,kept_segments=0,
                             approximate_boundaries=True,recognition_input='original_train_audio',
                             recognizer_executed=False)
            else:
                if recognizer is None: recognizer = recognizer_factory(model_cache, device, local_files_only)
                segments, quality = recognizer(wave)
            rec = dict(schema=SCHEMA, identity=lineage(row), model_id=MODEL_ID,
                       model_revision=MODEL_REVISION, algorithm=ALGORITHM,
                       segments=segments, quality=quality)
            validate_record(rec, row); atomic_json(target, rec); counts['prepared'] += 1
            target_hash=digest_file(target)
        entries[row['id']] = dict(path=record_name(row), sha256=target_hash,
                                 segments=len(rec['segments']))
        counts['segments'] += len(rec['segments'])
        counts['empty'] += int(not rec['segments'])
        counts['originals_below_receptive_field'] += int(row['output_samples']<400)
        counts[row['language'] + '/' + str(row['label']) + '/' + row['condition']] += len(rec['segments'])
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f'Phonemes {i + 1}/{len(rows)} prepared={counts["prepared"]} reused={counts["reused"]} empty={counts["empty"]}', flush=True)
    if not counts['segments'] and counts['originals_below_receptive_field']!=len(rows):
        raise ValueError('Recognizer yielded no usable phoneme intervals; cache cannot train PCL')
    manifest = dict(owner, complete=True, records=entries, parent_run=str(parent),
                    train_rows_sha256=digest_file(rows_path), summary=dict(counts),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    code_sha256={name: digest_file(Path(__file__).parent / name)
                                 for name in ('phonemes.py', 'prepare_phonemes.py')})
    # Publishing completeness is the final atomic step. Interrupted preparation
    # leaves resumable valid records but cannot be bound by training.
    if completed is None:
        atomic_json(root / 'manifest.json', manifest)
    elif entries != completed['records']:
        raise ValueError('Completed phoneme cache changed during idempotent preparation')
    binding = bind_manifest(root / 'manifest.json', rows)
    print('PHONEME_MANIFEST=' + binding['path'], flush=True)
    return binding


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run','--parent-run',dest='parent_run',required=True,
                   help='Run containing verified Train manifests; no detector checkpoint is loaded')
    p.add_argument('--output-dir', required=True, help='Compact metadata folder on the data disk')
    p.add_argument('--model-cache', required=True, help='Pinned Hugging Face model download cache on the data disk')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--local-files-only', action='store_true', help='Never download; require the pinned model in --model-cache')
    add_download_arguments(p)
    return p


def main():
    args = parser().parse_args()
    try:
        prepare(args.parent_run,args.output_dir,args.model_cache,args.device,args.local_files_only,
                hf_endpoint=args.hf_endpoint,download_retries=args.download_retries)
    except ModelDownloadError as exc:
        print('PHONEME_MODEL_DOWNLOAD_FAILED='+str(exc),file=sys.stderr,flush=True)
        raise SystemExit(2) from None


if __name__ == '__main__': main()
