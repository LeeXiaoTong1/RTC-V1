"""Read live V3.17 diagnostics without loading weights or using the GPU."""
import argparse
import os
from pathlib import Path
import time
from w2v_v39.common import ROOT
from w2v_v317_monitor.history import collect,console_summary
from w2v_v317_monitor.render import write_artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run');parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=30.)
    parser.add_argument('--upload-temp',action='store_true',help='Upload one current report snapshot; not allowed with --watch')
    parser.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    args=parser.parse_args()
    if args.interval<5:parser.error('interval must be at least five seconds')
    if args.watch and args.upload_temp:parser.error('Use --upload-temp for one report snapshot, without --watch')
    run=Path(args.run or os.environ.get('V317_MONITOR_RUN') or (ROOT/'exp'/'.latest_v317_run').read_text(encoding='utf-8').strip()).resolve()
    printed=None
    print('Read-only diagnostics: Ctrl+C stops this observer only. RUN='+str(run),flush=True)
    try:
        while True:
            report=collect(run);target=write_artifacts(report,run/'diagnostics')
            current=console_summary(report)
            if current!=printed:
                print(current,flush=True);print('CURVES='+str(target),flush=True);printed=current
            if not args.watch or report['completed']:
                if args.upload_temp:
                    from w2v_v317_monitor.archive import export_report
                    export_report(run,args.download_dir,True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:print('Observer closed; training was not signalled.',flush=True)


if __name__=='__main__':main()
