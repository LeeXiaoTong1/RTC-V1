"""Export observations and curves only; never include model or audio files."""
import os
from pathlib import Path
import zipfile
from w2v_aasist.launch import upload_archive


def export_report(run,destination,upload=False):
    run=Path(run).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    path=destination/(run.name+'_report.zip');tmp=path.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(run.rglob('*')):
                if not p.is_file() or p.is_symlink():continue
                relative=p.resolve().relative_to(run)
                allowed=p.suffix in ('.json','.jsonl','.md','.log') or (
                    relative.parts[0]=='diagnostics' and p.suffix in ('.html','.csv','.txt'))
                if allowed:z.write(p,relative.as_posix())
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)
    print('REPORT_ZIP='+str(path),flush=True)
    if upload:
        url=upload_archive(path);(run/'temp_download_url.txt').write_text(url+'\n',encoding='utf-8')
    return path
