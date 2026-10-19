"""Read-only fixed-threshold and equal-Fake-miss diagnostics; never fit a model."""
import numpy as np


def compare(before,after):
    with np.load(before,allow_pickle=False) as a,np.load(after,allow_pickle=False) as b:
        for name in ('ids','canonical_source','language','label','panel'):
            if not np.array_equal(a[name],b[name]):raise ValueError('Diagnostic source identity/order changed')
        y=a['label'];d0=a['logits'][:,0]-a['logits'][:,1];d1=b['logits'][:,0]-b['logits'][:,1]
        report={};flips=[]
        for panel in sorted(set(a['panel'])):
            for lang in ('en','zh'):
                keep=(a['panel']==panel)&(a['language']==lang);real=keep&(y==1);fake=keep&(y==0)
                budget=int((d0[fake]<0).sum());values=np.sort(d1[fake])
                # Largest threshold respecting the parent's number of Fake misses.
                # Ties remain together; never deploy this diagnostic threshold.
                threshold=float(values[min(budget,len(values)-1)])
                report[panel+'/'+lang]=dict(parent_real_errors=int((d0[real]>=0).sum()),
                    new_real_errors=int((d1[real]>=0).sum()),parent_fake_errors=budget,new_fake_errors=int((d1[fake]<0).sum()),
                    matched_fake_budget=dict(real_errors=int((d1[real]>=threshold).sum()),fake_errors=int((d1[fake]<threshold).sum()),
                        threshold=threshold,diagnostic_only=True,used_for_selection=False))
        p0=np.where(d0>=0,0,1);p1=np.where(d1>=0,0,1)
        for i in np.flatnonzero(p0!=p1):
            flips.append(dict(id=str(a['ids'][i]),source=str(a['canonical_source'][i]),panel=str(a['panel'][i]),
                language=str(a['language'][i]),label=int(y[i]),corrected=bool(p1[i]==y[i])))
        return dict(groups=report,corrected=sum(r['corrected'] for r in flips),regressed=sum(not r['corrected'] for r in flips),
                    flips=flips,deployed_threshold=.5,threshold_fit=False)
