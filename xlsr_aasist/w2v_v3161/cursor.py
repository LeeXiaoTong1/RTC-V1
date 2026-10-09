"""Continue the source sampling stream even when V3.16 stopped mid-epoch."""


def source_cursor(cfg, steps):
    epoch = cfg['sampling_epoch_offset']
    step = cfg.get('source_last_step', steps)
    if epoch < 1 or not 1 <= step <= steps:
        raise ValueError('Source LAST epoch/step is outside the current source plan')
    cursor = (epoch-1)*steps+step
    if cursor != cfg['source_committed_updates']:
        raise ValueError('Source LAST cursor does not match the current sampling plan')
    return cursor


def segments(cursor, updates, steps):
    """Consecutive source segments spanning exactly the requested updates."""
    while updates:
        epoch, start = divmod(cursor, steps)
        count = min(updates, steps-start)
        yield epoch, start, start+count
        cursor += count
        updates -= count


def completed_tag(cursor, steps):
    epoch, step = divmod(cursor, steps)
    return f'epoch_{epoch if step == 0 else epoch+1}_step_{step or steps}'
