"""V4 waveform generation, retaining the already computed base for PCL."""
import copy
from w2v_v31810.data import Waves as OriginalWaves, make_recipe
from w2v_v3189.data import prepared
from w2v_v3189.views import recipe as view_recipe, augment
from w2v_v3189.augment import generate


class Waves(OriginalWaves):
    def __getitem__(self, ticket):
        if isinstance(ticket, int):
            row = self.rows[ticket]
            return [prepared(row, self.original(row), row_index=ticket)]
        row = self.rows[ticket['index']]; wave = self.original(row)
        metadata = {key: copy.deepcopy(ticket[key]) for key in (
            'occurrence', 'canonical_source', 'meta_side', 'condition_block', 'task_kind',
            'task_direction', 'condition_rank', 'domain_override', 'nuisance_seed',
            'heldout_condition', 'batch_group_quotas', 'source_cycle', 'calibration') if key in ticket}
        if 'base_variant' in ticket:
            if self.split != 'train': raise ValueError('Dynamic views are Train only')
            rec = view_recipe(ticket.get('recipe_seed', self.cfg['seed']), row['id'], ticket['kind'],
                ticket['base_variant'], family=ticket.get('base_family'),
                generation=ticket.get('base_generation', 0), dynamic=ticket.get('base_dynamic', False))
            wave, base_meta = augment(wave, rec)
            metadata.update(base_augmentation=base_meta, view_id=ticket['base_view_id'])
        base = dict(row, **metadata)
        if ticket['kind'] == 'online': return [prepared(base, wave, role='online')]
        if ticket['kind'] != 'offline': raise ValueError('Only Online or Offline-source tickets are supported')
        self._bank(); rec = make_recipe(ticket, row, self.cfg, self.split)
        try: noisy, meta = generate(wave, rec, self.bank, self.engines)
        except Exception as exc: raise RuntimeError('V4.1 augmentation failed: '+str(row['id'])) from exc
        out = prepared(base, noisy, role='noisy', condition='noisy', augmentation=meta,
                       noise_recipe_view=int(ticket.get('noise_recipe_view', 0)))
        if 'pcl_noise' in ticket: out['_base_wave'] = wave
        return [out]
