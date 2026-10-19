"""Request a checkpoint at the next completed update; never kill another job."""
import argparse
import time
from pathlib import Path
from w2v_v31810.common import ROOT,read_json,atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run');a=p.parse_args()
    run=Path(a.run or (ROOT/'exp/.latest_v31813_run').read_text().strip()).resolve()
    if read_json(run/'config.json')['version']!='3.18.13':raise ValueError('Not a V3.18.13 run')
    atomic_json(run/'stop_requested.json',dict(requested_at=time.time()))
    print('Stop requested. Wait for PAUSED_RUN before launching another training job. Run: '+str(run))

if __name__=='__main__':main()
