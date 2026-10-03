"""Read-only V3.5 viewer: keep reference/epoch Dev results above one live line."""
import argparse
import math
from pathlib import Path
import re
import sys
import time

from live_progress import alive, log_state, read_state
from v33_console import Display, Events as ExistingEvents, LogTail, format_record

ROOT = Path(__file__).resolve().parent
TAG = re.compile(r'^(?:reference|baseline|epoch_\d+(?:_step_\d+)?)$')


class Events(ExistingEvents):
    def record(self, value):
        if not isinstance(value, dict) or not TAG.fullmatch(str(value.get('tag', ''))):
            return []
        tag = value['tag']
        if not isinstance(value.get('dev'), dict) or self._key(tag) in self.seen:
            return []
        self.seen.add(self._key(tag))
        rates = value.get('learning_rates', {})
        encoder = [float(rate) for name, rate in rates.items() if name.startswith('encoder.')]
        head = [float(rate) for name, rate in rates.items() if name.startswith('head.')]
        compact = {}
        if encoder:
            compact.update(encoder_min=min(encoder), encoder_max=max(encoder))
        if head:
            compact['head'] = max(head)
        lines = format_record(dict(value, learning_rates=compact or rates))
        if 'offline/en' not in value['dev'].get('groups', {}):
            lines = [line for line in lines if not line.startswith('  offline/en')]
        if tag == 'reference':
            lines[0] = lines[0].replace(' reference ', ' protected reference (comparison only) ')
        if value.get('weight_source') or value.get('weights_kind'):
            from v33_console import safe_text
            lines.append('  Evaluated weights: ' + safe_text(value.get('weight_source', value.get('weights_kind'))))
        return lines

    def saved(self, tag):
        if not self.run or not TAG.fullmatch(tag):
            return []
        value = read_state(self.run/(tag+'.json'))
        return self.record(value) if value else []

    def report(self, now=None, force=False):
        now = time.monotonic() if now is None else now
        if not self.run or (not force and now-self.report_clock < 2):
            return []
        self.report_clock = now
        path = self.run/'report.md'
        try:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size)
            if signature == self.report_signature:
                return []
            content = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            return []
        output, missing = [], False
        self.committed_tags = set()
        for line in content.splitlines():
            fields = line.split('|')
            tag = fields[1].strip() if len(fields) > 2 else ''
            if TAG.fullmatch(tag):
                self.committed_tags.add(tag)
                output.extend(self.saved(tag))
                missing = missing or self._key(tag) not in self.seen
        if not missing:
            self.report_signature = signature
        return output

    def consume(self, raw):
        if raw.startswith('V35_RUN='):
            raw = 'V32_RUN='+raw.split('=', 1)[1]
        elif raw.startswith('V35_EVENT='):
            raw = 'V32_EVENT='+raw.split('=', 1)[1]
        if raw.startswith('Dev reference '):
            return self.report(force=True)
        return super().consume(raw)


def watch(log, state, run=None, once=False, interval=1., stream=None):
    output = Display(stream or sys.stdout)
    if not output.tty and not once:
        raise SystemExit('Open the viewer in a terminal. For pipes/files use --once.')
    events, tail = Events(run), LogTail(log)
    output.messages(['V3.5 console: reference and completed Dev results stay above the live line.',
                     'ETA is for the current phase. Ctrl+C closes only this viewer.', 'LOG='+str(log)])
    try:
        while True:
            while True:
                before = tail.offset
                for line in tail.read():
                    output.messages(events.consume(line))
                if tail.offset-before < 2*1024*1024:
                    break
            output.messages(events.report(force=once))
            value = read_state(state) or log_state(log) or {
                'label': 'Waiting for the background job to publish progress', 'updated_at': time.time()}
            if value.get('status', 'running') == 'running' and alive(value.get('owner_pid')) is False:
                value = dict(value, status='stopped', label='Launcher stopped; inspect saved results and log', total=None)
            output.messages(events.phase(value.get('label', '')))
            output.progress(value)
            if once or value.get('status') in ('complete', 'failed', 'stopped'):
                output.messages(events._flush_dev())
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        output.clear_line()
        if output.tty:
            output.messages(['Viewer closed; no signal was sent to the background training job.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=1.)
    args = p.parse_args()
    if not math.isfinite(args.interval) or args.interval < .1:
        p.error('interval must be at least 0.1 seconds')
    log = args.log
    if log is None:
        log = Path((ROOT/'exp'/'.latest_v35_log').read_text(encoding='utf-8').strip())
    watch(log, Path(str(log)+'.progress.json'), run=args.run_dir,
          once=args.once, interval=args.interval)


if __name__ == '__main__':
    main()
