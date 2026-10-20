"""Fixed view-level CE allocation; language/class groups stay balanced."""

CE_WEIGHTS = {"online": .25, "clean_rtc": .25, "noisy": .5}


def classification_weights(cfg):
    """Require an explicit objective, never reinterpret an older saved run."""
    if cfg.get("classification_ce_weights") != CE_WEIGHTS:
        raise ValueError(
            "V3.19.1 requires classification_ce_weights="
            "{online: 0.25, clean_rtc: 0.25, noisy: 0.5}; "
            "start a new run/suite from the V3.18.9 parent best, "
            "do not resume an experiment with the old CE allocation")
    return dict(CE_WEIGHTS)
