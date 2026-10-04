"""Replay compact candidates with the actual deployment module before promotion."""
import numpy as np
import torch

from w2v_v36.fit import _save_dev_scores
from w2v_v36.metrics import evaluate
from .fit import logits_from_state,select_candidate,save_fit_report
from .patch import LanguageCorrectedClassifier


def validate_deployment(result,bundle,weight,bias,cfg,out):
    device=torch.device(cfg.get('device','cpu'))
    chunk=1024
    x=bundle['x']
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        for candidate in result['candidates']:
            if candidate.get('status')!='converged':continue
            state=candidate['patch']
            module=LanguageCorrectedClassifier(state,x.shape[1]).to(device).eval()
            measured=np.empty((len(x),2),dtype=np.float32)
            with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
                for start in range(0,len(x),chunk):
                    stop=min(start+chunk,len(x))
                    hidden=torch.from_numpy(np.array(x[start:stop],dtype=np.float32,copy=True)).to(device)
                    measured[start:stop]=module(hidden).cpu().numpy()
            reference=logits_from_state(x,state=state,chunk_rows=chunk)
            finite=bool(np.isfinite(measured).all() and np.isfinite(reference).all())
            drift=float(np.max(np.abs(measured.astype(np.float64)-reference))) if finite else None
            audit=dict(device=str(device),max_absolute_logit_difference=drift,
                       finite_logits=finite,
                       decision_differences=int(np.count_nonzero(measured.argmax(1)!=reference.argmax(1))) if finite else None,
                       rtol=3e-5,atol=1e-3,threshold=.5)
            candidate['deployment_replay']=audit
            if not finite or not np.allclose(measured,reference,rtol=3e-5,atol=1e-3):
                candidate.update(status='rejected',reason='Torch deployment replay differs from fitted FP32 logits',metrics=None)
            else:
                candidate['metrics']=evaluate(bundle['rows'],measured)
                result['dev_score_files'][candidate['name']]=_save_dev_scores(out,candidate['name'],bundle['rows'],measured)
            display_drift=f'{drift:.7g}' if drift is not None else 'nonfinite'
            print(f'[Replay] {candidate["name"]} device={device} max_logit_delta={display_drift} '
                  f'decision_changes={audit["decision_differences"]}',flush=True)
            del module
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf32
    selected=select_candidate(result['baseline'],result['candidates'],cfg)
    original=dict(weight=weight.tolist(),bias=bias.tolist(),language_state=None,student_state=None)
    result['selected']=selected
    result['selected_patch']=next((c['patch'] for c in result['candidates'] if c['name']==selected),original)
    result['outcome']={'baseline':'Neither candidate passed deployment replay and all fixed Dev guards; exact baseline retained.',
        'head_only_control':'Head-only correction selected; no evidence of a language-debias improvement.',
        'language_debias':'Language candidate selected after deployment replay; this does not prove a causal language mechanism.'}[selected]
    result['deployment_metric_policy']='Final selection uses actual Torch candidate module on cached Dev; no encoder rerun or fitting on Dev.'
    save_fit_report(out,result)
    return result
