# models/CLAM/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        drop_p = _get_dropout_p(dropout)

        module = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        if drop_p > 0:
            module.append(nn.Dropout(drop_p))
        module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        drop_p = _get_dropout_p(dropout)

        attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]

        if drop_p > 0:
            attention_a.append(nn.Dropout(drop_p))
            attention_b.append(nn.Dropout(drop_p))

        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class CLAM_SB(nn.Module):
    """
    Subtype version of CLAM-SB.

    Unified output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        n_features=1024
    ):
        super(CLAM_SB, self).__init__()

        self.size_dict = {
            "small": [n_features, 512, 256],
            "big": [n_features, 512, 384]
        }
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        drop_p = _get_dropout_p(dropout)
        if drop_p > 0:
            fc.append(nn.Dropout(drop_p))

        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def _safe_k(self, n_instances):
        return min(self.k_sample, max(1, n_instances))

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k = self._safe_k(h.size(0))
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)

        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)

        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k = self._safe_k(h.size(0))
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        p_targets = self.create_negative_targets(k, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)

        return instance_loss, p_preds, p_targets

    def _forward_single(self, h, label=None, instance_eval=False, attention_only=False):
        """
        h: [N, C]
        """
        A, h = self.attention_net(h)   # A:[N,1], h:[N,H]
        A = torch.transpose(A, 1, 0)   # [1, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)        # [1, N]

        total_inst_loss = None
        all_preds = []
        all_targets = []

        if instance_eval and label is not None:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label.view(-1), num_classes=self.n_classes).squeeze(0)

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_targets.extend(targets.detach().cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_targets.extend(targets.detach().cpu().numpy())
                    else:
                        continue

                total_inst_loss += instance_loss

            if self.subtyping and len(self.instance_classifiers) > 0:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        M = torch.mm(A, h)                  # [1, H]
        logits = self.classifiers(M)        # [1, C]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        results_dict = {
            "instance_loss": total_inst_loss,
            "inst_preds": all_preds,
            "inst_labels": all_targets,
            "features": M
        }

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
        attn_mask=None
    ):
        """
        h: [B, N, C] or [N, C]
        attn_mask: [B, N] or [N], True/1 for valid patches
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)

        if label is not None and label.dim() == 0:
            label = label.unsqueeze(0)

        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)

        B = h.size(0)
        logits_list, prob_list, hat_list, A_list = [], [], [], []
        inst_losses, inst_preds_all, inst_targets_all, feats_all = [], [], [], []

        for b in range(B):
            hb = h[b]
            if attn_mask is not None:
                valid_mask = attn_mask[b].bool()
                hb = hb[valid_mask]
            if hb.size(0) == 0:
                hb = h[b][:1]

            lb = None if label is None else label[b].view(1)

            out = self._forward_single(
                hb,
                label=lb,
                instance_eval=instance_eval,
                attention_only=attention_only
            )

            if attention_only:
                A_list.append(out)
                continue

            logits_b, prob_b, hat_b, A_b, results_b = out
            logits_list.append(logits_b)
            prob_list.append(prob_b)
            hat_list.append(hat_b)
            A_list.append(A_b)

            if results_b["instance_loss"] is not None:
                inst_losses.append(results_b["instance_loss"])
            inst_preds_all.extend(results_b["inst_preds"])
            inst_targets_all.extend(results_b["inst_labels"])
            feats_all.append(results_b["features"])

        if attention_only:
            return A_list[0] if len(A_list) == 1 else A_list

        logits = torch.cat(logits_list, dim=0)
        Y_prob = torch.cat(prob_list, dim=0)
        Y_hat = torch.cat(hat_list, dim=0)

        total_inst_loss = None
        if len(inst_losses) > 0:
            total_inst_loss = torch.stack(inst_losses).mean()

        results_dict = {
            "instance_loss": total_inst_loss,
            "inst_preds": inst_preds_all,
            "inst_labels": inst_targets_all,
            "features": torch.cat(feats_all, dim=0) if len(feats_all) > 0 else None
        }

        A_raw = A_list[0] if len(A_list) == 1 else A_list
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    """
    Subtype version of CLAM-MB.
    """

    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        n_features=1024
    ):
        nn.Module.__init__(self)

        self.size_dict = {
            "small": [n_features, 512, 256],
            "big": [n_features, 512, 384]
        }
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        drop_p = _get_dropout_p(dropout)
        if drop_p > 0:
            fc.append(nn.Dropout(drop_p))

        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def _forward_single(self, h, label=None, instance_eval=False, attention_only=False):
        A, h = self.attention_net(h)      # A:[N,C], h:[N,H]
        A = torch.transpose(A, 1, 0)      # [C, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)

        total_inst_loss = None
        all_preds = []
        all_targets = []

        if instance_eval and label is not None:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label.view(-1), num_classes=self.n_classes).squeeze(0)

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_targets.extend(targets.detach().cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_targets.extend(targets.detach().cpu().numpy())
                    else:
                        continue

                total_inst_loss += instance_loss

            if self.subtyping and len(self.instance_classifiers) > 0:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        M = torch.mm(A, h)    # [C, H]
        logits = torch.empty(1, self.n_classes, dtype=torch.float32, device=h.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        results_dict = {
            "instance_loss": total_inst_loss,
            "inst_preds": all_preds,
            "inst_labels": all_targets,
            "features": M.unsqueeze(0)
        }

        return logits, Y_prob, Y_hat, A_raw, results_dict
