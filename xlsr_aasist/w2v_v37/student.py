"""Train-only nonlinear language distillation; no second encoder at inference.

The original detector supplies all inference features. A small tanh branch
learns continuous language-teacher embeddings on official Train only. This is
an approximation to language orthogonalization, not a paper reproduction.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from live_progress import Phase
from w2v_v36.fit import source_coefficients

SCHEMA = 'v37_language_student_v1'


def _normalize(array):
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


class LanguageStudent(nn.Module):
    def __init__(self, state):
        super().__init__()
        if state.get('schema') != SCHEMA:
            raise ValueError('Unsupported language student schema')
        for name in ('input_mean', 'input_scale', 'w1', 'b1', 'w2', 'b2'):
            value = torch.as_tensor(state[name], dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite language student state: '+name)
            self.register_buffer(name, value)
        d = self.input_mean.numel()
        h = self.b1.numel()
        g = self.b2.numel()
        if (self.input_mean.shape != (d,) or self.input_scale.shape != (d,)
                or not bool((self.input_scale > 0).all()) or self.w1.shape != (h,d)
                or self.b1.shape != (h,) or self.w2.shape != (g,h) or self.b2.shape != (g,)
                or state.get('output_dim') != g):
            raise ValueError('Language student dimensions/scales differ')

    def forward(self, x):
        z = (x.float()-self.input_mean)/self.input_scale
        z = F.linear(torch.tanh(F.linear(z,self.w1,self.b1)),self.w2,self.b2)
        return F.normalize(z,p=2,dim=-1,eps=1e-8)


def predict_student(x, state, chunk_rows=4096):
    model = LanguageStudent(state).eval()
    if len(x.shape) != 2 or x.shape[1] != model.input_mean.numel() or chunk_rows < 1:
        raise ValueError('Language student input dimensions/chunk differ')
    output = np.empty((len(x), model.b2.numel()), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(x),chunk_rows):
            stop = min(start+chunk_rows,len(x))
            batch = torch.from_numpy(np.array(x[start:stop],dtype=np.float32,copy=True))
            output[start:stop] = model(batch).numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite predicted language embeddings')
    return output


def fit_student(x, teacher_g, rows, cfg):
    """Fit on the supplied Train partition; never receives Dev or tuning rows."""
    x = np.asarray(x,dtype=np.float32)
    target = np.asarray(teacher_g,dtype=np.float32)
    if (x.ndim != 2 or target.ndim != 2 or len(x) != len(rows) or len(target) != len(rows)
            or len(rows) == 0 or not np.isfinite(x).all() or not np.isfinite(target).all()
            or np.any(np.linalg.norm(target,axis=1) < 1e-8)):
        raise ValueError('Finite aligned nonzero Train language targets required')
    if any(r.get('split','train') != 'train' for r in rows):
        raise ValueError('Language student may use official Train only')
    coefficients = source_coefficients(rows,'group_balanced')
    mean = np.einsum('n,nd->d',coefficients,x,dtype=np.float64)
    variance = np.zeros(x.shape[1],dtype=np.float64)
    for start in range(0,len(x),4096):
        delta = x[start:start+4096].astype(np.float64)-mean
        variance += np.einsum('n,nd,nd->d',coefficients[start:start+4096],delta,delta)
    scale = np.sqrt(np.maximum(variance,1e-6)).astype(np.float32)
    mean = mean.astype(np.float32)
    epochs = int(cfg.get('student_epochs',30))
    batch_size = int(cfg.get('student_batch_rows',4096))
    hidden = int(cfg.get('student_hidden',64))
    learning_rate = float(cfg.get('student_learning_rate',.002))
    decay = float(cfg.get('student_weight_decay',.0001))
    if min(epochs,batch_size,hidden) < 1 or not np.isfinite([learning_rate,decay]).all() or learning_rate <= 0 or decay < 0:
        raise ValueError('Invalid language student optimization settings')
    seed = int(cfg.get('seed',3701))
    # Fork the RNG so auxiliary fitting cannot perturb other experiment stages.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(x.shape[1],hidden),nn.Tanh(),nn.Linear(hidden,target.shape[1]))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=decay)
    generator = np.random.default_rng(seed)
    z = torch.from_numpy((x-mean)/scale)
    goal = torch.from_numpy(_normalize(target).astype(np.float32))
    coeff = torch.from_numpy(coefficients.astype(np.float32))
    tracker = Phase('V3.7 Train-only language branch',epochs)
    trace = []
    for epoch in range(epochs):
        order = generator.permutation(len(rows))
        total = 0.
        for start in range(0,len(order),batch_size):
            ids = torch.as_tensor(order[start:start+batch_size],dtype=torch.long)
            prediction = F.normalize(model(z[ids]),p=2,dim=-1,eps=1e-8)
            per_row = (prediction-goal[ids]).square().sum(-1)
            loss = (per_row*coeff[ids]).sum()*(len(rows)/len(ids))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Language student loss became nonfinite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            total += float((per_row.detach()*coeff[ids]).sum())
        trace.append(total)
        tracker.update(epoch+1,loss=total)
        if epoch == 0 or (epoch+1)%5 == 0 or epoch+1 == epochs:
            print(f'[Language] epoch={epoch+1}/{epochs} weighted_embedding_error={total:.6f}',flush=True)
    state = dict(schema=SCHEMA,input_mean=mean.tolist(),input_scale=scale.tolist(),
        w1=model[0].weight.detach().tolist(),b1=model[0].bias.detach().tolist(),
        w2=model[2].weight.detach().tolist(),b2=model[2].bias.detach().tolist(),output_dim=target.shape[1],
        diagnostics=dict(rows=len(rows),epochs=epochs,hidden=hidden,loss_trace=trace,
                         supervision='frozen generic LID embeddings; official Train only'))
    fitted = predict_student(x,state)
    state['diagnostics']['train_weighted_cosine'] = float(coefficients @ (fitted*_normalize(target)).sum(1))
    return state
