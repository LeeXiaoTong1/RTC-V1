"""One bounded A-microbatch audit, before backward; never changes .grad."""
import torch


def contributions(model, losses, sequences, factor):
    candidates = dict(repair_output=model.repair.out.weight, gate_output=model.repair.gate[-1].weight,
        classifier=model.head.aasist.out_layer.weight, layer_mix=model.fusion.layer_logits)
    # Raw H tests the stop-gradient boundary directly, without differentiating
    # all 3B parameters merely to establish the same graph separation.
    candidates.update({'raw_feature_'+str(i): x for i,x in enumerate(sequences) if x.requires_grad})
    selected = {k:v for k,v in candidates.items() if v.requires_grad}
    keys,tensors = list(selected),list(selected.values())
    result,values = {},[]
    for name,loss in losses.items():
        grads = torch.autograd.grad(loss*factor,tensors,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(tensors)
        values.append(torch.stack([g.detach().float().square().sum().sqrt() if g is not None else loss.new_zeros(()) for g in grads]))
    for name,norms in zip(losses,torch.stack(values).cpu().tolist()):
        result[name] = dict(zip(keys,norms))
    # Honest scope: these are selected tensors in ONE microbatch, not whole
    # module norms or the gradient of the complete logical batch.
    ratios = {name: {k: v/max(result['ce'][k],1e-12) for k,v in row.items()} for name,row in result.items() if name != 'ce'}
    return dict(norms=result,relative_to_ce=ratios,outer_factor=factor,
        scope='First A microbatch with LF; selected full tensors, unscaled pre-clip gradients; no parameter mutation')
