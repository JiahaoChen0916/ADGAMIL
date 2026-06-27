# models/DTFD/network.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import RRTEncoder


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum("bgf,cf->bcg", [features, tweight])
    return cam_maps


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0, n_robust=0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = nn.Dropout(p=self.droprate)

        self.apply(initialize_weights)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class DimReduction(nn.Module):
    def __init__(
        self,
        n_channels,
        m_dim=512,
        dropout=False,
        act="relu",
        n_robust=0,
        rrt=False,
        rrt_convk=15,
        rrt_moek=3,
        rrt_as=False,
        rrt_md=False
    ):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.rrt = RRTEncoder(
            attn="rrt",
            pool="none",
            n_layers=2,
            epeg_k=rrt_convk,
            crmsa_k=rrt_moek,
            moe_mask_diag=rrt_md,
            init=True,
            rrt_window_num=16
        ) if rrt else nn.Identity()

        self.relu1 = nn.ReLU(inplace=True) if act.lower() == "relu" else nn.GELU()
        self.drop = nn.Dropout(0.25)
        self.dropout = dropout
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        if self.dropout:
            x = self.drop(x)

        if isinstance(self.rrt, RRTEncoder):
            x, _ = self.rrt(x)
        else:
            x = self.rrt(x)

        return x


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0, n_robust=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x):
        AA = self.attention(x)          # [1, N]
        afeat = torch.mm(AA, x)         # [1, L]
        pred = self.classifier(afeat)   # [1, C]
        return pred


class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1, n_robust=0):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        self.apply(initialize_weights)

    def forward(self, x, isNorm=True):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)

        if isNorm:
            A = F.softmax(A, dim=1)

        return A


class DTFD(nn.Module):
    """
    Subtype version of DTFD-MIL.

    Unified output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(
        self,
        lr=None,
        weight_decay=None,
        steps=None,
        criterion=None,
        rrt=False,
        epeg_k=15,
        crmsa_k=3,
        input_dim=1024,
        inner_dim=512,
        n_classes=2,
        group=8,
        distill="MaxMinS"
    ) -> None:
        super().__init__()

        self.classifier = Classifier_1fc(inner_dim, n_classes, 0.25)
        self.attention = Attention(inner_dim)
        self.dimReduction = DimReduction(
            input_dim,
            inner_dim,
            rrt=rrt,
            dropout=0.25,
            rrt_convk=epeg_k,
            rrt_moek=crmsa_k
        )
        self.UClassifier = Attention_with_Classifier(L=inner_dim, num_cls=n_classes, droprate=0.25)

        self.group = group
        self.distill = distill
        self.criterion = criterion

    def _prepare_instances(self, x, attn_mask=None):
        """
        x: [N, C] or [1, N, C]
        """
        if x.dim() == 3:
            x = x.squeeze(0)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.squeeze(0)
            valid_mask = attn_mask.bool()
            x = x[valid_mask]

        if x.size(0) == 0:
            x = x.new_zeros((1, x.size(-1)))

        return x

    def _split_groups(self, n_instances):
        feat_index = list(range(n_instances))
        index_chunk_list = np.array_split(np.array(feat_index), min(self.group, n_instances))
        index_chunk_list = [sst.tolist() for sst in index_chunk_list if len(sst) > 0]
        return index_chunk_list

    def _subbag_forward(self, tmidFeat):
        """
        tmidFeat: [n, inner_dim]
        """
        tAA = self.attention(tmidFeat).squeeze(0)                      # [n]
        tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)           # [n, d]
        tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)   # [1, d]
        tPredict = self.classifier(tattFeat_tensor)                   # [1, C]

        patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  # [C, n]
        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)                         # [n, C]
        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)

        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
        topk_idx_max = sort_idx[:1].long()
        topk_idx_min = sort_idx[-1:].long()
        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
        af_inst_feat = tattFeat_tensor

        if self.distill == "MaxMinS":
            pseudo_feat = MaxMin_inst_feat
        elif self.distill == "MaxS":
            pseudo_feat = max_inst_feat
        elif self.distill == "AFS":
            pseudo_feat = af_inst_feat
        else:
            raise NotImplementedError(f"Unsupported distill mode: {self.distill}")

        return tPredict, pseudo_feat, tAA, tattFeat_tensor

    def _forward_single(self, x, label=None, attn_mask=None):
        """
        x: [N, C] or [1, N, C]
        """
        x = self._prepare_instances(x, attn_mask=attn_mask)   # [N_valid, C]
        x = self.dimReduction(x)                              # [N_valid, d]

        index_chunk_list = self._split_groups(x.shape[0])

        slide_pseudo_feat = []
        slide_sub_logits = []
        sub_attn_list = []

        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(
                x, dim=0, index=torch.LongTensor(tindex).to(x.device)
            )                                                # [n_sub, d]

            tPredict, pseudo_feat, tAA, tattFeat_tensor = self._subbag_forward(subFeat_tensor)
            slide_sub_logits.append(tPredict)
            slide_pseudo_feat.append(pseudo_feat)
            sub_attn_list.append(tAA)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)   # [k, d] or [2k, d]
        sub_logits = torch.cat(slide_sub_logits, dim=0)           # [num_groups, C]

        logits = self.UClassifier(slide_pseudo_feat)              # [1, C]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        aux_loss = None
        if label is not None:
            if label.dim() == 0:
                label = label.view(1)
            group_labels = label.repeat(sub_logits.size(0))
            aux_loss = F.cross_entropy(sub_logits, group_labels)

        results_dict = {
            "instance_loss": aux_loss,
            "sub_logits": sub_logits,
            "pseudo_features": slide_pseudo_feat,
            "features": None
        }

        return logits, Y_prob, Y_hat, None, results_dict

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        x: [B, N, C] or [N, C]
        attn_mask: [B, N] or [N]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if label is not None and label.dim() == 0:
            label = label.unsqueeze(0)

        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)

        logits_list, prob_list, hat_list = [], [], []
        aux_losses = []

        for b in range(x.size(0)):
            xb = x[b]
            lb = None if label is None else label[b]
            mb = None if attn_mask is None else attn_mask[b]

            logits_b, prob_b, hat_b, _, results_b = self._forward_single(
                xb, label=lb, attn_mask=mb
            )

            logits_list.append(logits_b)
            prob_list.append(prob_b)
            hat_list.append(hat_b)

            if results_b["instance_loss"] is not None:
                aux_losses.append(results_b["instance_loss"])

        logits = torch.cat(logits_list, dim=0)
        Y_prob = torch.cat(prob_list, dim=0)
        Y_hat = torch.cat(hat_list, dim=0)

        total_aux_loss = None
        if len(aux_losses) > 0:
            total_aux_loss = torch.stack(aux_losses).mean()

        results_dict = {
            "instance_loss": total_aux_loss,
            "features": None
        }

        return logits, Y_prob, Y_hat, None, results_dict
