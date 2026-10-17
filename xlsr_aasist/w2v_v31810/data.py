"""One full Online/Noisy waveform per origin; no PCL or reference encoding."""
import copy
from w2v_v3184.data import Waves as BaseWaves, close, TicketSampler
from w2v_v3189.data import ready, prepared, microbatches
from w2v_v3189.views import recipe as view_recipe, augment
from w2v_v3189.augment import NoiseBank, Engines, recipe, generate
from w2v_v3189.activity import settings
from .workers import loader


def make_recipe(ticket, row, cfg, split='train'):
    """Force declared domains AFTER the old marginal policy is sampled.

    All non-domain controls are matched by a task/rank key independent of class,
    language, source and A/B role. Noise material/offset has its own source seed.
    This retains the existing RTC engine, activity and single-codec path.
    """
    domain = ticket.get('domain_override')
    seed = ticket.get('recipe_seed', cfg['seed'])
    view = int(ticket.get('noise_recipe_view', 0))
    scene = domain['scene'] if domain else ticket.get('scene')
    family = domain['family'] if domain else ticket.get('family')
    rec = recipe(seed, ticket['occurrence'], row['group_id'], view,
        ticket.get('epoch', 0), cfg.get('warm_epochs', 1), split,
        None if domain else family, ticket.get('probe', False),
        cfg.get('rtc_stress_probability', .35), cfg.get('rtc_stress_ramp_epochs', 2.),
        cfg.get('scene_probability', 1.), cfg.get('rtc_policy', 'three-profile'), scene,
        activity_policy=cfg.get('rtc_activity_policy', 'mixed'))
    if domain:
        if split != 'train' or family not in ('webrtc', 'ffmpeg', 'light'):
            raise ValueError('Training domains must use the three declared processing families')
        nuisance_seed = ticket['nuisance_seed']
        nuisance = recipe(nuisance_seed, 'task-rank-common', 'task-rank-common',
            (ticket.get('step', 0) + ticket.get('condition_rank', 0)) % 2,
            ticket.get('epoch', 0), cfg.get('warm_epochs', 1), split,
            None, False, cfg.get('rtc_stress_probability', .35),
            cfg.get('rtc_stress_ramp_epochs', 2.), 1., cfg.get('rtc_policy', 'three-profile'), scene,
            activity_policy=cfg.get('rtc_activity_policy', 'mixed'))
        for key in ('snr_db', 'input_gain_db', 'codec', 'bitrate', 'cutoff', 'nr', 'gain', 'ns', 'target', 'activity'):
            rec[key] = copy.deepcopy(nuisance[key])
        if 'stress' in nuisance: rec['stress'] = copy.deepcopy(nuisance['stress'])
        else: rec.pop('stress', None)
        if scene == 'echo': rec['echo'] = copy.deepcopy(nuisance['echo'])
        rec['family'] = family
        rec['domain_contract'] = dict(scene=scene, family=family, nuisance_seed=nuisance_seed,
            matched_controls='SNR/gain/codec/profile sampled without class, language, source or meta-side')
    elif family:
        # Explicit fixed diagnostics must not be overwritten by three-profile.
        rec['family'] = family
        if family in ('anlmdn', 'g711_mulaw'):
            rec['codec'] = 'none'; rec.pop('stress', None); rec['activity'] = settings(0, 'off')
    return rec


class Waves(BaseWaves):
    def _bank(self):
        if self.bank is None:
            self.bank = NoiseBank(self.cfg['noise_records'][self.split],
                self.cfg.get('noise_cache_mib', 128) * 1024**2,
                self.cfg.get('scene_noise_records', {}).get(self.split))
            if getattr(self.bank, 'scenes', None) is not None:
                self.bank.scenes.cap = self.cfg.get('scene_cache_mib', 32) * 1024**2
            self.engines = Engines(self.cfg['ffmpeg'])

    def __getitem__(self, ticket):
        if isinstance(ticket, int):
            row = self.rows[ticket]
            return [prepared(row, self.original(row), row_index=ticket)]
        row = self.rows[ticket['index']]
        wave = self.original(row)
        metadata = {key: copy.deepcopy(ticket[key]) for key in (
            'occurrence', 'canonical_source', 'meta_side', 'condition_block', 'task_kind',
            'task_direction', 'condition_rank', 'domain_override', 'nuisance_seed',
            'heldout_condition', 'batch_group_quotas', 'source_cycle', 'calibration') if key in ticket}
        if 'base_variant' in ticket:
            if self.split != 'train': raise ValueError('Dynamic views are Train only')
            rec = view_recipe(ticket.get('recipe_seed', self.cfg['seed']), row['id'], ticket['kind'],
                ticket['base_variant'], family=ticket.get('base_family'),
                generation=ticket.get('base_generation', 0), dynamic=ticket.get('base_dynamic', False))
            wave, base = augment(wave, rec)
            metadata.update(base_augmentation=base, view_id=ticket['base_view_id'])
        base = dict(row, **metadata)
        if ticket['kind'] == 'online': return [prepared(base, wave, role='online')]
        if ticket['kind'] != 'offline': raise ValueError('Only Online or Offline-source tickets are supported')
        self._bank(); rec = make_recipe(ticket, row, self.cfg, self.split)
        try: noisy, meta = generate(wave, rec, self.bank, self.engines)
        except Exception as exc:
            raise RuntimeError('V3.18.10 augmentation failed: '+str(row['id'])+' '+str(rec.get('domain_contract', {}))) from exc
        return [prepared(base, noisy, role='noisy', condition='noisy', augmentation=meta,
            noise_recipe_view=int(ticket.get('noise_recipe_view', 0)))]
