"""Fixed-.5 errors and paired-source changes; no fitted calibration."""
from collections import defaultdict
import numpy as np
from w2v_v31810.validation import binary_metrics

PANELS = ('official_online','regular_noisy','heldout_noisy')


def report(rows, logits):
    groups = {}; panels = {}
    for panel in PANELS:
        selected = [i for i,r in enumerate(rows) if r['validation_panel']==panel]
        panels[panel] = binary_metrics([rows[i] for i in selected],np.asarray(logits)[selected])
        for lang in ('en','zh'):
            indices = [i for i in selected if rows[i]['language']==lang]
            groups[panel+'/'+lang] = binary_metrics([rows[i] for i in indices],np.asarray(logits)[indices])
    return dict(panels=panels,groups=groups,
        score=.3*panels['official_online']['macro_f1']+.35*panels['regular_noisy']['macro_f1']+.35*panels['heldout_noisy']['macro_f1'])


def guard(baseline, candidate):
    checks = {'en_real_improves':candidate['groups']['official_online/en']['real_false_positive'] < baseline['groups']['official_online/en']['real_false_positive']}
    for panel in PANELS[1:]:
        checks[panel+'_macro_preserved'] = candidate['panels'][panel]['macro_f1'] >= baseline['panels'][panel]['macro_f1']-1e-12
        for lang in ('en','zh'):
            key = panel+'/'+lang
            checks[key+'_fake_preserved'] = candidate['groups'][key]['fake_false_negative'] <= baseline['groups'][key]['fake_false_negative']+1e-12
            checks[key+'_real_preserved'] = candidate['groups'][key]['real_false_positive'] <= baseline['groups'][key]['real_false_positive']+1e-12
    return dict(passed=all(checks.values()),checks=checks)


def paired_changes(rows, before, after):
    y = np.array([r['label'] for r in rows]); bp=np.asarray(before).argmax(1); ap=np.asarray(after).argmax(1)
    # argmax selects Fake=0 on a tie, exactly P(fake)>=.5.
    sources=defaultdict(dict)
    for i,r in enumerate(rows):
        sources[r['canonical_source']].setdefault(r['validation_panel'],[]).append(i)
    counts=defaultdict(lambda:dict(views=0,sources=set(),corrected=0,regressed=0))
    for source,panels in sources.items():
        online=panels.get('official_online',[])
        if not online: continue
        on_wrong=any(bp[i]!=y[i] for i in online)
        for panel in PANELS[1:]:
            for i in panels.get(panel,[]):
                if bp[i]==y[i]: cohort='noisy_initially_correct'
                else: cohort='both_wrong' if on_wrong else 'online_correct_noisy_wrong'
                key=f'{panel}/{rows[i]["language"]}/{rows[i]["label"]}/{cohort}'
                c=counts[key];c['views']+=1;c['sources'].add(source)
                c['corrected']+=int(bp[i]!=y[i] and ap[i]==y[i])
                c['regressed']+=int(bp[i]==y[i] and ap[i]!=y[i])
    return {k:dict(v,sources=len(v['sources'])) for k,v in counts.items()}


def bootstrap(rows, before, after, iterations=500, seed=31812):
    """Paired cluster bootstrap; group strata keep EN/ZH and Real/Fake fixed."""
    strata=defaultdict(dict)
    for i,r in enumerate(rows):
        strata[r['language'],r['label']].setdefault(r['canonical_source'],[]).append(i)
    rng=np.random.default_rng(seed); changes=defaultdict(list)
    base=np.asarray(before);new=np.asarray(after)
    # Compute only needed confusion counts, avoiding repeated AUC/EER sorting.
    def values(indices,z):
        selected=[rows[i] for i in indices]; yy=np.array([r['label'] for r in selected]);pred=z[indices].argmax(1)
        output={}
        for panel in PANELS:
            mask=np.array([r['validation_panel']==panel for r in selected])
            f1=[]
            for label in (0,1):
                tp=np.sum(mask&(yy==label)&(pred==label));fp=np.sum(mask&(yy!=label)&(pred==label));fn=np.sum(mask&(yy==label)&(pred!=label))
                f1.append(2*tp/max(1,2*tp+fp+fn))
            output[panel+'_macro_delta']=np.mean(f1)
            for lang in ('en','zh'):
                lm=mask&np.array([r['language']==lang for r in selected])
                for label,name in ((0,'fake_fnr'),(1,'real_fpr')):
                    m=lm&(yy==label);output[f'{panel}/{lang}_{name}_delta']=np.sum(m&(pred!=yy))/max(1,np.sum(m))
        return output
    for _ in range(iterations):
        indices=[]
        for members in strata.values():
            units=list(members.values())
            for j in rng.integers(0,len(units),size=len(units)):indices.extend(units[j])
        a=values(indices,base);b=values(indices,new)
        for key in a:changes[key].append(float(b[key]-a[key]))
    intervals={k:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975))) for k,v in changes.items()}
    supported=intervals['official_online/en_real_fpr_delta']['high']<0
    for panel in PANELS[1:]:
        supported &= intervals[panel+'_macro_delta']['low']>=-1e-12
        for lang in ('en','zh'):
            for metric in ('fake_fnr','real_fpr'):
                supported &= intervals[f'{panel}/{lang}_{metric}_delta']['high']<=1e-12
    return dict(iterations=iterations,unit='canonical source; paired across conditions',
        intervals=intervals,supported=bool(supported),
        warning='Pointwise descriptive 95% intervals, not a multiple-testing corrected guarantee; Audit is historically reused Dev.')
