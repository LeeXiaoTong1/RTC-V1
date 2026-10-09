"""Concise permanent metrics; inference counters go to a separate detail file."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from w2v_v313.train import infer as full_infer
from w2v_v39.metrics import GROUPS


def infer(model, rows, cfg, label, run):
    with (Path(run)/'details.log').open('a', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            return full_infer(model, rows, cfg, label)[0]


def print_metrics(tag, metrics):
    def p(x): return 'NA' if x is None else f'{100*x:.3f}'
    print(f'\n[Dev] V3.17 {tag} Clean={p(metrics["clean_f1"])} '
          f'Noisy={p(metrics["noisy_f1"])} Weighted={p(metrics["weighted_f1"])}', flush=True)
    for name in GROUPS:
        g = metrics['groups'][name]
        print(f'  {name:10s} Recall(fake/real)={p(g["recall"][0])}/{p(g["recall"][1])} '
              f'F1={p(g["macro_f1"])} AP={p(g["ap"])} AUC={p(g["auc"])} EER={p(g["eer"])}', flush=True)
