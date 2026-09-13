"""Run portable logic checks or actual FFmpeg/PyTorch checks in the sdd env."""
import argparse
from pathlib import Path
import sys
import unittest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rtc", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--tests", action="store_true", help="Requires PyTorch; fails if any test is skipped")
    p.add_argument("--logic", action="store_true", help="Only the portable plan/rotation tests")
    args = p.parse_args()
    if args.rtc:
        import numpy as np
        from rtc_noisy.simulator import LocalRTC, RTCSettings
        from rtc_noisy_v2.plan import ALL_SETTINGS, HELDOUT_SETTINGS, TRAIN_SETTINGS
        rtc = LocalRTC(args.ffmpeg)
        print(rtc.version)
        waveform = .05*np.random.RandomState(7).normal(size=70321).astype(np.float32)
        for setting in ALL_SETTINGS:
            output = rtc(waveform, RTCSettings(*setting))
            if output.shape != waveform.shape or not np.isfinite(output).all():
                raise RuntimeError(f"RTC check failed for {setting}")
        a = rtc(waveform, RTCSettings(*HELDOUT_SETTINGS[0]))
        b = rtc(waveform, RTCSettings(*HELDOUT_SETTINGS[0]))
        if not np.array_equal(a, b):
            raise RuntimeError("RTC is not deterministic")
        print(f"RTC passed: 70321 -> 70321 for all {len(TRAIN_SETTINGS)} train + {len(HELDOUT_SETTINGS)} held-out settings")
    if args.tests or args.logic:
        folder = Path(__file__).resolve().parent / "tests"
        suite = unittest.defaultTestLoader.discover(str(folder), pattern="test_rtc_noisy_v2_logic.py")
        if args.tests:
            import torch
            print("PyTorch:", torch.__version__)
            suite.addTests(unittest.defaultTestLoader.discover(str(folder), pattern="test_rtc_noisy_v2_torch.py"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or (args.tests and result.skipped):
            sys.exit(1)
    if not (args.rtc or args.tests or args.logic):
        p.error("Choose --rtc, --tests, or --logic")


if __name__ == "__main__":
    main()
