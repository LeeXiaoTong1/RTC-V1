"""Local DSP approximation, NOT WebRTC or a reproduction of a real platform.

Process the entire utterance with fresh state: FFT noise reduction, dynamic
level normalization, then Opus VoIP encode/decode. Crop only AFTER processing.
"""
import shutil
import subprocess
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RTCSettings:
    noise_reduction: int = 12
    max_gain: int = 4
    bitrate: int = 24000

    def as_dict(self):
        return asdict(self)

    @classmethod
    def sample(cls, rng):
        # Draw independently: severe noise must not imply a specific codec.
        return cls(int(rng.choice([6, 12, 18])), int(rng.choice([2, 4, 8])),
                   int(rng.choice([16000, 24000, 32000])))


class LocalRTC:
    def __init__(self, ffmpeg="ffmpeg"):
        self.ffmpeg = shutil.which(ffmpeg)
        if self.ffmpeg is None:
            raise RuntimeError("FFmpeg is missing. Install ffmpeg and run check_rtc_noisy.py --rtc")
        self.version = subprocess.check_output([self.ffmpeg, "-version"], text=True).splitlines()[0]

    def _run(self, arguments, audio_bytes):
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-threads", "1", "-filter_threads", "1", *arguments]
        result = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=120)
        if result.returncode:
            raise RuntimeError("Local RTC failed: " + result.stderr.decode(errors="replace"))
        return result.stdout

    def __call__(self, waveform, settings=None):
        cfg = settings or RTCSettings()
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim != 1 or not x.size or not np.isfinite(x).all():
            raise ValueError("Expected finite, nonempty, mono audio")
        filters = (f"afftdn=nr={cfg.noise_reduction}:nf=-35:tn=1,"
                   f"dynaudnorm=f=150:g=7:p=0.9:m={cfg.max_gain}")
        encoded = self._run([
            "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0", "-af", filters,
            "-c:a", "libopus", "-b:a", str(cfg.bitrate), "-application", "voip",
            "-frame_duration", "20", "-vbr", "on", "-f", "ogg", "pipe:1"],
            x.astype("<f4").tobytes())
        decoded = self._run(["-f", "ogg", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
                             "-f", "f32le", "pipe:1"], encoded)
        y = np.frombuffer(decoded, dtype="<f4").copy()
        # Ogg Opus carries pre-skip/end trimming. Do not manually shift a second
        # time, or silently pad/truncate a codec that produced a wrong length.
        if len(x) != len(y) or not np.isfinite(y).all():
            raise RuntimeError(f"Invalid RTC output; samples {len(x)} -> {len(y)}")
        return y
