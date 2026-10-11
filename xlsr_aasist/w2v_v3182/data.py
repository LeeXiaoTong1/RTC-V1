"""Three full-wave views from one verified source, with deterministic fresh noise."""
from w2v_v318.data import (Waves as BaseWaves, ready, loader, close, TicketSampler,
                           microbatches as pack_batches, read_wave, worker_init, passthrough)
from .augment import NoiseBank, Engines, recipe, generate


def microbatches(units, size, frame_budget):
    # Pairing is maintained by the logical objective, so a physical batch may
    # split a triplet. This uses all four default slots instead of only three.
    yield from pack_batches(([row] for unit in units for row in unit), size, frame_budget)


class Waves(BaseWaves):
    def __getitem__(self, ticket):
        if isinstance(ticket, int) or ticket['kind'] != 'source':
            return super().__getitem__(ticket)
        original = self.rows[ticket['index']]
        x = self.original(original)
        shared = dict(occurrence=ticket['occurrence'], source_mass=ticket['source_mass'],
                      source_id=original['source_id'], source_sha256=original['source_sha256'],
                      group_id=original['group_id'])
        output = [dict(original, **{**shared, 'role':'offline', 'wave':ready(x),
                                    'ce_weight':.1*ticket['source_mass']})]
        if ticket['online_index'] is not None:
            online = self.rows[ticket['online_index']]
            if any(online[k] != original[k] for k in ('source_sha256','group_id','language','label')):
                raise ValueError('Official Online pair identity changed')
            output.append(dict(online, **{**shared, 'role':'online', 'wave':ready(self.original(online)),
                                          'ce_weight':.45*ticket['source_mass']}))
        if self.bank is None:
            self.bank = NoiseBank(self.cfg['noise_records'][self.split], self.cfg['noise_cache_mib']*1024**2)
            self.engines = Engines(self.cfg['ffmpeg'])
        for v in range(1 if ticket['online_index'] is not None else 2):
            rec = recipe(self.cfg['seed'], ticket['occurrence'], original['group_id'], v,
                         ticket['epoch'], self.cfg['warm_epochs'], self.split, probe=ticket.get('probe', False))
            try:
                wave, meta = generate(x, rec, self.bank, self.engines)
            except Exception as exc:
                raise RuntimeError(f'V3.18.2 augmentation failed for {original["id"]}: {rec}') from exc
            role = ('noisy_a','noisy_b')[v]
            output.append(dict(original, **{**shared, 'role':role, 'condition':role,
                'wave':ready(wave), 'augmentation':meta, 'ce_weight':.45*ticket['source_mass']}))
        return output
