"""V3.18 FFmpeg execution with terminal isolation and actionable failures."""
import os
import signal
import subprocess

from rtc_noisy.simulator import LocalRTC


def stderr_text(value):
    if isinstance(value, bytes):
        value = value.decode(errors='replace')
    return (value or '<empty>')[-4000:]


class IsolatedRTC(LocalRTC):
    def _run(self, arguments, audio_bytes):
        cmd = [self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-threads', '1', '-filter_threads', '1', *arguments]
        context = f'ffmpeg={self.ffmpeg!r} arguments={arguments!r} input_bytes={len(audio_bytes)}'
        try:
            result = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=120,
                                    start_new_session=(os.name == 'posix'))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'Local RTC failed: timeout_seconds=120 {context} '
                               f'stderr={stderr_text(exc.stderr)!r}') from exc
        except OSError as exc:
            raise RuntimeError(f'Local RTC failed: launch_error={exc!r} {context}') from exc
        if result.returncode:
            reason = 'none'
            if os.name == 'posix' and result.returncode < 0:
                try:
                    reason = signal.Signals(-result.returncode).name
                except ValueError:
                    reason = str(-result.returncode)
            raise RuntimeError(f'Local RTC failed: returncode={result.returncode} signal={reason} '
                               f'{context} stderr={stderr_text(result.stderr)!r}')
        return result.stdout
