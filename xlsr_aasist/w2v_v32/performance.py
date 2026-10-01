"""Wall-clock diagnostics kept outside deterministic checkpoint metrics."""
import json
from pathlib import Path
import time
from w2v_aasist.runtime import atomic_json


class TimedLoader:
    def __init__(self, iterable):
        self.iterable = iterable
        self.last_wait = 0.

    def __iter__(self):
        iterator = iter(self.iterable)
        while True:
            started = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                return
            self.last_wait = time.perf_counter()-started
            yield item


def record(run, step, epoch, examples, wait, elapsed, stats):
    sources = stats['ordinary_sources']+stats['noisy_sources']
    row = dict(global_step=step, epoch=epoch, data_wait_seconds=wait,
               compute_seconds=elapsed, wall_seconds=wait+elapsed,
               sources=sources, encoded_views=len(examples),
               encoder_calls=stats['encoder_forward_microbatches'],
               mean_physical_batch=len(examples)/max(1,stats['encoder_forward_microbatches']),
               audio_seconds=sum(e['audio_seconds'] for e in examples),
               sources_per_second=sources/max(1e-9,wait+elapsed),
               **{k:v for k,v in stats.items() if k.startswith(('gpu_', 'activation_'))})
    path = Path(run)/'performance.jsonl'
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    atomic_json(Path(run)/'performance_latest.json',row)
    return row
