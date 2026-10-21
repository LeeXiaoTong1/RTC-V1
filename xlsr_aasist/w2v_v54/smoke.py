"""Exercise actual warm/joint GPU paths, then restore the untouched start."""
import copy
import time
from .common import partial_state,apply_partial,capture_rng,restore_rng,to_cpu,atomic_json,fingerprint
from .schedule import schedule,warmup_steps
from .step import train_step
from w2v_v3191.precision import scaler_for


def check(model,optimizer,units,cfg,steps,path):
    weights=partial_state(model)
    opt=to_cpu(optimizer.state_dict())
    scaler=scaler_for(optimizer,cfg);scale=copy.deepcopy(scaler.state_dict())
    rng,modes=capture_rng(),[(m,m.training) for m in model.modules()]
    joint=model.joint
    names=('_v3189_joint_update','_v3189_update','_v3189_joint','_v54_warm_steps','_v54_total_updates')
    attributes={k:getattr(optimizer,k) for k in names if hasattr(optimizer,k)}
    warm=warmup_steps(cfg,steps*cfg['epochs'])
    indices=sorted({0,min(warm,steps*cfg['epochs']-1)})
    checks=[];started=time.monotonic()
    try:
        for index in indices:
            schedule(optimizer,cfg,index//steps,index%steps,steps)
            for attempt in range(4):
                result=train_step(model,optimizer,units,cfg,audit=True)
                if not result['amp']['skipped_update']:break
            else:raise FloatingPointError('GPU startup check failed four AMP attempts')
            checks.append(dict(stage=result['repair_stage'],loss=result['loss'],
                encoded_views=result['encoded_views'],backend_windows=result['backend_windows'],
                gradient_audit=result['repair_gradient_audit'],repair_update=result.get('repair_update'),
                amp=result['amp'],hard_capture_valid=result['hard_examples']['validated']))
        report=dict(status='passed',config=fingerprint(cfg),device=cfg['device'],checks=checks,
            seconds=time.monotonic()-started,committed_training_updates=0,
            scope='Actual first training batch; temporary warm/joint updates restored including RNG and Adam state; not a performance experiment')
    except BaseException as error:
        atomic_json(path,dict(status='failed',config=fingerprint(cfg),error=str(error),checks=checks))
        raise
    finally:
        apply_partial(model,weights)
        optimizer.load_state_dict(opt)
        scaler.load_state_dict(scale)
        optimizer.zero_grad(set_to_none=True)
        for name in names:
            if name in attributes:setattr(optimizer,name,attributes[name])
            elif hasattr(optimizer,name):delattr(optimizer,name)
        model.set_training_stage(joint)
        for module,mode in modes:module.training=mode
        restore_rng(rng)
    atomic_json(path,report)
    return report
