"""Fixed threshold, fixed sources, equal language weight after Dev transfer."""
from w2v_v31810.validation import evaluate_protocol as base_evaluate, PANEL_WEIGHTS


def language_balanced(result):
    for panel,metrics in result['panels'].items():
        metrics['language_macro_f1']=sum(result['groups'][panel+'/'+lang]['macro_f1'] for lang in ('en','zh'))/2
    result['pooled_legacy_score']=result['score']
    result['score']=sum(PANEL_WEIGHTS[p]*result['panels'][p]['language_macro_f1'] for p in PANEL_WEIGHTS) if result['split']=='select' else None
    result['selection_aggregation']='Equal EN/ZH macro-F1; fixed 0.30 Online, 0.35 Regular, 0.35 Heldout'
    result['comparable_to_old_selects']=False
    return result


def evaluate_protocol(model,protocol,cfg,*args,**kwargs):
    # Changing the training random seed must not regenerate the retained Dev.
    fixed=dict(cfg,seed=protocol.get('synthesis_seed',protocol['seed']))
    from .validation_data import infer_fixed
    kwargs.setdefault('infer_fn',infer_fixed)
    return language_balanced(base_evaluate(model,protocol,fixed,*args,**kwargs))
