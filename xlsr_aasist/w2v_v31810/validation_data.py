"""Fixed validation synthesis: one worker pool, no PCL or refreshed variants."""
import numpy as np
from w2v_v3184.data import Waves as RawWaves, close
from w2v_v3189.data import prepared, microbatches
from w2v_v3189.augment import recipe, generate, Engines, NoiseBank
from w2v_v3189.activity import settings
from .workers import loader
from .common import observation


class FixedWaves(RawWaves):
    def __getitem__(self, ticket):
        index = ticket['index']; row = self.rows[index]; x = self.original(row)
        base = dict(row, row_index=index, occurrence=ticket['occurrence'])
        if ticket['kind']=='online': return [prepared(base, x, role='online')]
        if self.bank is None:
            # All validation views explicitly name a scene, so the legacy bank
            # is never sampled. Retain its constructor contract with this bank.
            scenes = self.cfg['scene_noise_records']['dev']
            self.bank = NoiseBank(scenes, self.cfg.get('noise_cache_mib', 64)*1024**2, scenes)
            self.engines = Engines(self.cfg['ffmpeg'])
        family = ticket['family']
        rec = recipe(self.cfg['seed'], ticket['occurrence'], row['canonical_source'], 0, 1, 0,
                     'dev', family=family, probe=False, stress_probability=0,
                     scene_probability=1, rtc_policy='legacy', scene=ticket['scene'],
                     activity_policy=self.cfg.get('rtc_activity_policy', 'mixed'))
        # scene+three-profile used to override an explicit heldout family.
        # Never allow that silent evaluation contamination here.
        rec['family'] = family
        if family in ('anlmdn', 'g711_mulaw'):
            rec['codec'] = 'none'; rec['activity'] = settings(0, 'off')
        else:
            rec['codec'] = 'opus'
        y, meta = generate(x, rec, self.bank, self.engines)
        return [prepared(base, y, role='noisy', augmentation=meta)]


def infer_fixed(model, rows, cfg, tickets, split='dev'):
    cfg = dict(cfg, workers=cfg.get('eval_workers', cfg.get('workers', 0)))
    dataset = FixedWaves(rows, cfg, split)
    batch_size = cfg['eval_batch']
    batches = loader(dataset, cfg, batch_sampler=[tickets[a:a+batch_size] for a in range(0, len(tickets), batch_size)])
    logits = []; observed = []
    try:
        with observation(model):
            for units in batches:
                expected = [r for unit in units for r in unit]
                positions = {id(row): i for i, row in enumerate(expected)}
                output = [None]*len(expected)
                for micro in microbatches(units, cfg['microbatch'], cfg['frame_budget']):
                    examples = [r for unit in micro for r in unit]
                    values = model([r['wave'] for r in examples]).float().cpu().numpy()
                    for row, value in zip(examples, values): output[positions[id(row)]] = value
                if any(value is None for value in output): raise ValueError('Validation microbatch dropped a recording')
                logits.extend(output)
                observed.extend({k:v for k,v in row.items() if k not in ('wave', 'augmentation')} for row in expected)
    finally:
        close(batches)
    return np.stack(logits), observed
