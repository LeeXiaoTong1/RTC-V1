"""Fixed before training; no selection from Progress/Eval scores."""
import hashlib
import itertools
import json

SCHEMA = "rtc_noisy_optimization_v2"
BANDS = ((5., 10.), (10., 15.), (15., 20.), (20., 25.))
ALL_SETTINGS = tuple(itertools.product((6, 12, 18), (2, 4, 8), (16000, 24000, 32000)))
# Each marginal setting occurs twice in held-out and seven times in training.
# The combination, not necessarily each individual knob, is held out.
HELDOUT_SETTINGS = ((6, 2, 16000), (12, 4, 24000), (18, 8, 32000),
                    (6, 4, 32000), (12, 8, 16000), (18, 2, 24000))
TRAIN_SETTINGS = tuple(x for x in ALL_SETTINGS if x not in HELDOUT_SETTINGS)


def digest_json(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_seed(seed, *parts):
    return int(digest_json([int(seed), *parts])[:16], 16)


def plan_definition():
    return {"schema": SCHEMA, "train_settings": [list(x) for x in TRAIN_SETTINGS],
            "heldout_settings": [list(x) for x in HELDOUT_SETTINGS],
            "snr_bands": [list(x) for x in BANDS],
            "setting_order": ["noise_reduction", "max_gain", "bitrate"]}


PLAN_ID = digest_json(plan_definition())


def settings_for(role):
    if role in ("train", "dev_seen"):
        return TRAIN_SETTINGS
    if role == "dev_heldout":
        return HELDOUT_SETTINGS
    raise ValueError(f"Unknown cache role: {role}")


def setting_key(row):
    return tuple(row[k] for k in ("noise_reduction", "max_gain", "bitrate"))


def group_coefficients(ordinary_count, real_view_count, reference_count, beta):
    """Counts control only the internal composition of the OTHER input group."""
    if not 0 <= beta <= 1:
        raise ValueError("noisy_ce_weight must lie in [0,1]")
    counts = (ordinary_count, real_view_count, reference_count)
    if any(n <= 0 for n in counts):
        raise ValueError("All three other-input groups must be nonempty")
    total = sum(counts)
    return tuple((1. - beta) * n / total for n in counts) + (float(beta),)
