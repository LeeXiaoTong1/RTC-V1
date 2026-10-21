"""V5.2 view weights, independent of language/class and sample counts."""
CE_WEIGHTS = dict(online=.6, clean_rtc=.2, noisy=.2)


def classification_weights(cfg):
    if cfg.get('classification_ce_weights') != CE_WEIGHTS:
        raise ValueError('V5.2 requires Online60/CleanRTC20/Noisy20; start a new run')
    return dict(CE_WEIGHTS)
