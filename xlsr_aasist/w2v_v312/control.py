"""Training progress is distinct from permission to replace the protected best."""


def stopping(history, value, cursor, epoch_steps, baseline, cfg):
    values = [entry['metrics']['weighted_f1'] for entry in history] + [value['weighted_f1']]
    best, stale = float('-inf'), 0
    for score in values:
        if score > best + cfg['progress_min_delta']:
            best, stale = score, 0
        else:
            stale += 1
    catastrophic = value['weighted_f1'] < baseline['weighted_f1']-cfg['catastrophic_weighted_drop']
    minimum_reached = cursor >= min(cfg['epochs'], cfg['minimum_epochs']) * epoch_steps
    stalled = minimum_reached and stale >= cfg['patience']
    reason = 'catastrophic_regression' if catastrophic else ('candidate_progress_stalled' if stalled else 'continue')
    return dict(stop=catastrophic or stalled, reason=reason, checks_without_progress=stale,
                best_candidate_weighted=best, minimum_epochs_reached=minimum_reached,
                selection_guards_relaxed=False)
