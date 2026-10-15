"""AASIST with sample-local normalization and shared-pass temporal statistics.

Graph and readout equations retain the pinned SSL-AASIST implementation. The
64d temporal statistic uses the genuine pre-graph temporal nodes e_T, before
graph top-k selection; no second encoder/backend pass or duplicated windows.
"""
import torch
from torch.nn import functional as F
from w2v_v318.aasist import SSLAASIST as OriginalAASIST
from .normalization import replace_batchnorm, stable_std


class SSLAASIST(OriginalAASIST):
    normalization = 'groupnorm-conv-layernorm-graph-v1'

    def __init__(self, input_dim=128):
        super().__init__(input_dim)
        replace_batchnorm(self)

    def evidence(self, x):
        return self._evidence(x, False)

    def evidence_with_temporal(self, x):
        return self._evidence(x, True)

    def _evidence(self, x, temporal):
        x = self.LL(x).transpose(1, 2).unsqueeze(1)
        x = self.selu(self.first_bn(F.max_pool2d(x, (3, 3))))
        x = self.selu(self.first_bn1(self.encoder(x)))
        w = self.attention(x)
        e_S = (x * F.softmax(w, dim=-1)).sum(dim=-1).transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = (x * F.softmax(w, dim=-2)).sum(dim=-2).transpose(1, 2)
        # Each input batch has a single exact window length. No temporal
        # padding enters convolution, normalization, graph or this statistic.
        temporal_std = stable_std(e_T, dim=1) if temporal else None
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        aug_T, aug_S, aug_master = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + aug_T, out_S1 + aug_S, master1 + aug_master

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        aug_T, aug_S, aug_master = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + aug_T, out_S2 + aug_S, master2 + aug_master

        out_T = torch.maximum(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.maximum(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.maximum(self.drop_way(master1), self.drop_way(master2))
        evidence = torch.cat((out_T.abs().amax(1), out_T.mean(1), out_S.abs().amax(1),
                              out_S.mean(1), master.squeeze(1)), dim=1)
        return (evidence, temporal_std) if temporal else evidence
