"""Read-only fixed-screen V3.19 progress and committed metrics (stdlib only)."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import textwrap
import time
from v3184_progress import Steps,duration

TRAIN_PATTERN = re.compile(r'\[Train\] V3\.19 epoch (\d+)/(\d+).*updates=(\d+)')
INFERENCE_PATTERN = re.compile(r'^V3\.19 (.+): (\d+)/(\d+)(?: elapsed=([\d.]+)m)?')


def ascii_text(value):
    return ''.join(c if 32<=ord(c)<127 else ' ' for c in str(value))


class JsonFiles:
    def __init__(self):self.cache={}
    def read(self,path,default):
        previous=self.cache.get(path)
        try:
            stat=path.stat();stamp=(stat.st_mtime_ns,stat.st_size)
            if previous and previous[0]==stamp:return previous[1]
            value=json.loads(path.read_text(encoding='utf-8'))
            self.cache[path]=(stamp,value);return value
        except (OSError,ValueError):return previous[1] if previous else default


class Dashboard:
    def __init__(self,log):
        self.log=Path(log);self.offset=0;self.pending=b'';self.run=None
        self.steps=Steps();self.files=JsonFiles();self.history=[];self.links={}
        self.epoch=0;self.epochs=5;self.total=0;self.exit_code=None;self.error=''
        self.selections={};self.fallback_step=0;self.fallback_seconds=0.
        self.phase_counts=None
        self.phase='Startup: checking inputs';self.changed=time.monotonic()
    def stage(self,value):
        if value!=self.phase:
            self.phase=value;self.changed=time.monotonic();self.phase_counts=None
    def line(self,line):
        if line.startswith('V319_RUN='):self.run=Path(line.split('=',1)[1].strip())
        match=TRAIN_PATTERN.search(line)
        inference=INFERENCE_PATTERN.search(line)
        if match:
            self.epoch,self.epochs,self.total=map(int,match.groups());self.stage('Training')
            self.fallback_step=0;self.fallback_seconds=0.
        elif re.search(r'\bSTEP \d+/\d+',line):
            step=re.search(r'\bSTEP (\d+)/(\d+)',line)
            self.fallback_step,self.total=map(int,step.groups())
            timing=re.search(r'compute_s/update=([\d.]+).*wait_s/update=([\d.]+)',line)
            if timing:self.fallback_seconds=sum(map(float,timing.groups()))
            self.stage('Training')
        elif inference:
            label,current,total,minutes=inference.groups()
            if 'submission inference' in label:phase='Submission inference'
            elif 'final-only audit' in label:phase='Evaluation: final-only audit'
            elif 'selection panel' in label:phase='Evaluation: scene selection'
            elif 'fixed Dev' in label:phase='Evaluation: fixed Dev'
            else:phase='Evaluation'
            if label.startswith('parent '):phase='Initial '+phase[0].lower()+phase[1:]
            self.stage(phase)
            self.phase_counts=(int(current),int(total),60*float(minutes or 0))
        elif line.startswith('V319_PROFILE'):self.stage('Startup: profiling physical batches')
        elif line.startswith('[Startup]'):
            self.stage('Startup: loading encoder' if 'loading' in line.lower() else 'Startup: verifying inputs')
        elif line.startswith('[Initial scenes]'):self.stage('Initialization: evaluating six scenes')
        elif line.startswith('[Eval]'):self.stage('Evaluation: full Dev')
        elif line.startswith('[Eval phase]'):
            name=line.split(']',1)[1].strip()
            self.stage({'dev':'Evaluation 1/4: full Dev','train_probe':'Evaluation 2/4: Train probe',
                        'mechanisms':'Evaluation 3/4: RTC mechanisms','six_scenes':'Evaluation 4/4: six scenes'}.get(name,'Evaluation'))
        elif line.startswith(('[Complete]','V319_SELECTED=')):self.stage('Training complete: preparing outputs')
        elif line.startswith('[Submission]'):self.stage('Exporting best submission')
        elif line.startswith('SUBMISSION_ZIP='):self.stage('Preparing submission delivery')
        elif line.startswith('[Report]'):
            self.stage('Uploading report' if 'uploading' in line else 'Packaging report')
        elif line.startswith('SUBMISSION_TEMP_URL='):self.links['submission_url']=line.split('=',1)[1].strip()
        elif line.startswith('REPORT_TEMP_URL='):self.links['report_url']=line.split('=',1)[1].strip()
        elif re.match(r'\w*(?:Error|Exception):',line):self.error=ascii_text(line)
    def poll(self):
        if self.log.is_file():
            if self.log.stat().st_size<self.offset:self.offset=0;self.pending=b''
            with self.log.open('rb') as stream:
                stream.seek(self.offset)
                for block in iter(lambda:stream.read(65536),b''):
                    self.offset+=len(block);lines=(self.pending+block).split(b'\n');self.pending=lines.pop()
                    for raw in lines:self.line(raw.decode('utf-8',errors='replace').rstrip('\r'))
        if self.run is not None:
            self.steps.poll(self.run/'steps.jsonl')
            self.history=self.files.read(self.run/'training_history.json',[])
            self.links.update(self.files.read(self.run/'artifact_links.json',{}))
            cfg=self.files.read(self.run/'config.json',{})
            self.epochs=cfg.get('epochs',self.epochs)
            done=self.files.read(self.run/'completed.json',{})
            self.selections=done.get('selections',{})
            if not self.selections:
                self.selections={'best_weighted':'starting_parent','best_guarded':'starting_parent'}
                for item in self.history:
                    for kind in item.get('promoted',[]):
                        if kind in self.selections:self.selections[kind]=item['tag']
        marker=Path(str(self.log)+'.exit')
        if marker.is_file() and (not self.log.exists() or self.log.stat().st_size<=self.offset):
            try:self.exit_code=int(marker.read_text().strip())
            except (OSError,ValueError):return
            if self.pending:self.line(self.pending.decode('utf-8',errors='replace'));self.pending=b''
            self.stage('Complete' if self.exit_code==0 else 'Stopped with an error')
    def progress(self):
        if self.phase!='Training':
            if self.phase_counts is not None:
                current,total,elapsed=self.phase_counts
                text=f'{self.phase} | {current}/{total} {100*current/max(1,total):.1f}%'
                if current and elapsed:text+=' | ETA '+duration(max(0,total-current)*elapsed/current)
                return text
            return self.phase+' | '+duration(time.monotonic()-self.changed)
        row=self.steps.latest
        step=int(row['step']) if row and int(row['epoch'])==self.epoch else self.fallback_step
        fraction=min(1.,step/max(1,self.total));filled=int(fraction*12)
        bar='['+'#'*filled+'-'*(12-filled)+']'
        text=f'{bar} {step}/{self.total} {fraction*100:.1f}%'
        if step:
            seconds=self.steps.seconds/max(1,self.steps.count) if row and int(row['epoch'])==self.epoch else self.fallback_seconds
            text+=f' | {seconds:.2f}s/step | ETA {duration((self.total-step)*seconds)}'
        return text


def pct(value):return f'{100*value:6.2f}' if isinstance(value,(int,float)) and math.isfinite(value) else '    NA'


def epoch_number(entry):
    if isinstance(entry.get('epoch'),int):return entry['epoch']
    match=re.match(r'epoch_(\d+)_',entry.get('tag',''))
    return int(match.group(1)) if match else 0


def frame(state,width=80,height=24,panel='dev',epoch=None,page=1):
    width=max(1,width-1);limit=max(1,height-1)
    lines=[f'V3.19 | Epoch {state.epoch}/{state.epochs}',state.progress()]
    tail=[]
    for kind in ('submission','report'):
        url=state.links.get(kind+'_url')
        if url:tail.extend(textwrap.wrap(kind.upper()+': '+url,width=width,break_on_hyphens=False))
        elif state.exit_code is not None and state.links.get(kind+'_status')=='failed':tail.append(kind.upper()+': upload failed')
    if state.exit_code not in (None,0):
        tail.append('ERROR: '+(state.error or 'See training log for details'))
    if len(tail)>max(1,limit-2):
        # A very small terminal cannot fit two long URLs. Keep their exact text
        # in the already saved link file rather than scrolling away the status.
        tail=['Links: run/download_links.txt']
    if state.history:
        best=state.selections.get('best_weighted')
        if best is None:best=next((e['tag'] for e in reversed(state.history) if 'best_weighted' in e.get('promoted',[])),None)
        summary=['Epoch  Clean   Noisy Weighted BestW']
        history_count=min(len(state.history),5,max(1,limit-len(lines)-len(tail)-4))
        for item in state.history[-history_count:]:
            v=item['metrics']
            summary.append(f'{epoch_number(item):>5} {pct(v["clean_f1"])} {pct(v["noisy_f1"])} {pct(v["weighted_f1"])}'
                           +('   *' if item.get('tag')==best else ''))
        if best=='starting_parent':summary.append('BestW: retained starting parent')
        lines+=['']+summary
        chosen=next((e for e in reversed(state.history) if epoch_number(e)==epoch),None) if epoch else state.history[-1]
        if chosen is not None:
            metrics=chosen.get('metrics',{}) if panel=='dev' else chosen.get('panel',{})
            data=metrics.get('groups',{}) if panel=='dev' else metrics.get('families',{})
            if panel=='dev':data={name:value for name,value in data.items() if '/' in name}
            heading='Full Dev' if panel=='dev' else 'Held-out RTC / scenes'
            heading+=f' epoch {epoch_number(chosen)} (%)'
            rows=[]
            for name,v in sorted(data.items()):
                extra=(metrics.get('matched',{}).get(name) or v.get('matched') or {}).get('real_recall')
                rows.append(f'{name:12s} '+ ' '.join(pct(x) for x in [*v.get('recall',[None,None]),v.get('macro_f1'),v.get('auc'),v.get('eer'),extra]))
            slots=max(0,limit-len(lines)-len(tail)-3)
            if slots:
                # Reserve one line for a page notice only when actually needed.
                size=slots-1 if len(rows)>slots and slots>1 else slots
                pages=max(1,math.ceil(len(rows)/max(1,size)));number=min(page,pages)
                lines+=['',heading,'Group         FakeR  RealR     F1    AUC    EER  R@99F']
                lines+=rows[(number-1)*size:number*size]
                if pages>1 and slots>1:lines.append(f'Page {number}/{pages}; --panel {panel} --page N')
        elif epoch:lines.append(f'Epoch {epoch} has not completed')
    elif state.exit_code is None:lines.append('No completed epoch yet')
    # Keep artifact URLs in full. All other fields stay within the screen width.
    body=[ascii_text(line)[:width] for line in lines[:max(0,limit-len(tail))]]
    return '\n'.join((body+[ascii_text(line)[:width] for line in tail])[:limit])


def watch(log,*,once=False,panel='dev',epoch=None,page=1,stream=None,sleep=time.sleep):
    stream=stream or sys.stdout;state=Dashboard(log);last='';printed_at=0.;previous_epochs=-1
    tty=stream.isatty()
    while True:
        state.poll();size=shutil.get_terminal_size((100,28))
        value=frame(state,size.columns,size.lines,panel,epoch,page);now=time.monotonic()
        if value!=last and (tty or once or state.exit_code is not None or len(state.history)!=previous_epochs or now-printed_at>=30):
            stream.write(('\033[H\033[2J' if tty else '')+value+'\n');stream.flush()
            last=value;printed_at=now;previous_epochs=len(state.history)
        if once or state.exit_code is not None:return state.exit_code or 0
        sleep(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',nargs='?');parser.add_argument('--once',action='store_true')
    parser.add_argument('--panel',choices=('dev','rtc'),default='dev')
    parser.add_argument('--epoch',type=int);parser.add_argument('--page',type=int,default=1)
    args=parser.parse_args()
    if args.page<1 or (args.epoch is not None and args.epoch<1):parser.error('Epoch/page must be positive')
    root=Path(__file__).resolve().parent
    log=Path(args.log) if args.log else Path((root/'exp'/'.latest_v319_log').read_text().strip())
    try:return watch(log,once=args.once,panel=args.panel,epoch=args.epoch,page=args.page)
    except KeyboardInterrupt:return 0  # Closing the read-only viewer never signals training.


if __name__=='__main__':raise SystemExit(main())
