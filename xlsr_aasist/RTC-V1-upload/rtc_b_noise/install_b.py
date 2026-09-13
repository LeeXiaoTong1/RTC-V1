"""Install B into an existing official xlsr_aasist directory; keep a backup."""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil


def patched_text(source):
    if "# RTC_B_ENV_NOISE_V1" in source:
        return source
    init_anchor = "        self.cut = 64600"
    item_anchor = "        audio = Tensor(pad_audio(audio, self.cut))"
    for anchor in (init_anchor, item_anchor):
        if source.count(anchor) != 1:
            raise ValueError("Source layout differs from the official baseline. No files changed. Missing/ambiguous: " + anchor.strip())
    ast.parse(source)
    source = "# RTC_B_ENV_NOISE_V1\nfrom .env_noise import NoiseAugment\n" + source
    source = source.replace(init_anchor, init_anchor + "\n        self.env_noise = NoiseAugment.from_env() if self.use_rawboost else None")
    source = source.replace(item_anchor,
        "        if self.env_noise is not None:\n"
        "            audio = self.env_noise(audio[:self.cut], sample_rate)\n" + item_anchor)
    ast.parse(source)
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    target = a.repo.expanduser().resolve() / "utils" / "data_utils.py"
    source = target.read_text(encoding="utf-8")
    result = patched_text(source)
    module = target.parent / "env_noise.py"
    payload = Path(__file__).with_name("env_noise.py").read_bytes()
    if module.exists() and module.read_bytes() != payload:
        raise SystemExit("Existing utils/env_noise.py differs; no files changed")
    if result == source:
        if not module.exists():
            module.write_bytes(payload)
        print("B already installed. Enable it with RTC_B_NOISE_MANIFEST.")
        return
    backup = target.with_name("data_utils.py.before_b_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    shutil.copy2(target, backup)
    module.write_bytes(payload)
    target.write_text(result, encoding="utf-8")
    print("Installed:", module)
    print("Modified:", target)
    print("Backup:", backup)
    print("Default is OFF. No model, training arguments, or inference files changed.")


if __name__ == "__main__":
    main()
