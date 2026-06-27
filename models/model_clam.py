# models/model_clam.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dynamic_graph import DynamicGraphEmbedding


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        modules = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            modules.append(nn.Dropout(0.25))
        modules.append(nn.Linear(D, n_classes))
        self.attention_net = nn.Sequential(*modules)

    def forward(self, x):
        A = self.attention_net(x)
        return A, x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        if dropout:
            self.attention_a.add_module("dropout", nn.Dropout(0.25))
            self.attention_b.add_module("dropout", nn.Dropout(0.25))
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class CLAM_SB(nn.Module):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        num_neighbors=5,
    ):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]

        self.fc = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if gate:
            self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.enable_instance_eval = (instance_loss_fn is not None) and (k_sample is not None) and (k_sample > 0)

        self.dynamic_graph = DynamicGraphEmbedding(
            input_dim=size[1],
            hidden_dim=size[1],
            num_neighbors=num_neighbors,
        )

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device, dtype=torch.long)

    @staticmethod
    def _masked_softmax(attn_logits, attn_mask=None, dim=1):
        if attn_mask is None:
            return F.softmax(attn_logits, dim=dim)

        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        mask = attn_mask.unsqueeze(-1)

        fill_value = torch.finfo(attn_logits.dtype).min
        masked_logits = attn_logits.masked_fill(~mask, fill_value)
        attn = F.softmax(masked_logits, dim=dim)
        attn = attn * mask.to(attn.dtype)
        denom = attn.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return attn / denom

    def _encode_instances(self, h, attn_mask=None):
        B, N, D = h.shape
        h = self.fc(h.reshape(B * N, D)).reshape(B, N, -1)
        h = self.dynamic_graph(h, mask=attn_mask)
        return h

    def _inst_eval_in(self, A, h, classifier):
        """
        A: (num_valid,)
        h: (num_valid, D)
        """
        num_valid = A.numel()
        if num_valid < 2:
            return None

        k = min(self.k_sample, num_valid // 2)
        if k <= 0:
            return None

        sorted_ids = torch.argsort(A, descending=True)
        top_p_ids = sorted_ids[:k]
        top_n_ids = sorted_ids[-k:]

        top_p = h[top_p_ids]
        top_n = h[top_n_ids]

        p_targets = self.create_positive_targets(k, h.device)
        n_targets = self.create_negative_targets(k, h.device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)

        logits = classifier(all_instances)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, all_targets)
        return loss, preds, all_targets

    def _inst_eval_out(self, A, h, classifier):
        """
        A: (num_valid,)
        h: (num_valid, D)
        """
        num_valid = A.numel()
        if num_valid < 1:
            return None

        k = min(self.k_sample, num_valid)
        if k <= 0:
            return None

        top_ids = torch.argsort(A, descending=True)[:k]
        top_instances = h[top_ids]
        targets = self.create_negative_targets(k, h.device)

        logits = classifier(top_instances)
        preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, targets)
        return loss, preds, targets

    def _instance_eval_sb(self, A_prob, h, label, attn_mask=None):
        if (not self.enable_instance_eval) or (label is None):
            return None

        B, N, _ = A_prob.shape
        label = label.view(B)

        total_inst_loss = h.new_tensor(0.0)
        total_terms = 0
        all_preds = []
        all_targets = []

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=h.device)
        else:
            attn_mask = attn_mask.bool()

        one_hot = F.one_hot(label, num_classes=self.n_classes)

        for b in range(B):
            valid_mask = attn_mask[b]
            if valid_mask.sum().item() == 0:
                continue

            A_b = A_prob[b, valid_mask, 0]
            h_b = h[b, valid_mask]
            inst_labels = one_hot[b]

            for c, classifier in enumerate(self.instance_classifiers):
                if inst_labels[c].item() == 1:
                    out = self._inst_eval_in(A_b, h_b, classifier)
                elif self.subtyping:
                    out = self._inst_eval_out(A_b, h_b, classifier)
                else:
                    out = None

                if out is None:
                    continue

                inst_loss, preds, targets = out
                total_inst_loss = total_inst_loss + inst_loss
                total_terms += 1
                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())

        if total_terms == 0:
            return None

        total_inst_loss = total_inst_loss / float(total_terms)
        return {
            "instance_loss": total_inst_loss,
            "inst_loss": total_inst_loss,
            "inst_labels": np.asarray(all_targets, dtype=np.int64),
            "inst_preds": np.asarray(all_preds, dtype=np.int64),
        }

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
        attn_mask=None,
    ):
        """
        Args:
            h: (B, N, D)
            attn_mask: (B, N), True for valid patches.
        """
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        h = self._encode_instances(h, attn_mask=attn_mask)
        B, N, D = h.shape

        A, h_flat = self.attention_net(h.reshape(B * N, D))
        A = A.reshape(B, N, -1)
        h = h_flat.reshape(B, N, D)

        if attention_only:
            return A

        A_raw = A.clone()
        A = self._masked_softmax(A, attn_mask=attn_mask, dim=1)

        M = torch.bmm(A.transpose(1, 2), h).squeeze(1)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if instance_eval:
            inst_results = self._instance_eval_sb(A, h, label, attn_mask=attn_mask)
            if inst_results is not None:
                results_dict.update(inst_results)

        if return_features:
            results_dict["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        num_neighbors=5,
    ):
        super().__init__(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            embed_dim=embed_dim,
            num_neighbors=num_neighbors,
        )

        size = self.size_dict[size_arg]
        if gate:
            self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            self.attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])

    def _instance_eval_mb(self, A_prob, h, label, attn_mask=None):
        if (not self.enable_instance_eval) or (label is None):
            return None

        B, N, C = A_prob.shape
        label = label.view(B)

        total_inst_loss = h.new_tensor(0.0)
        total_terms = 0
        all_preds = []
        all_targets = []

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=h.device)
        else:
            attn_mask = attn_mask.bool()

        one_hot = F.one_hot(label, num_classes=self.n_classes)

        for b in range(B):
            valid_mask = attn_mask[b]
            if valid_mask.sum().item() == 0:
                continue

            A_b = A_prob[b, valid_mask]
            h_b = h[b, valid_mask]
            inst_labels = one_hot[b]

            for c, classifier in enumerate(self.instance_classifiers):
                A_c = A_b[:, c]
                if inst_labels[c].item() == 1:
                    out = self._inst_eval_in(A_c, h_b, classifier)
                elif self.subtyping:
                    out = self._inst_eval_out(A_c, h_b, classifier)
                else:
                    out = None

                if out is None:
                    continue

                inst_loss, preds, targets = out
                total_inst_loss = total_inst_loss + inst_loss
                total_terms += 1
                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())

        if total_terms == 0:
            return None

        total_inst_loss = total_inst_loss / float(total_terms)
        return {
            "instance_loss": total_inst_loss,
            "inst_loss": total_inst_loss,
            "inst_labels": np.asarray(all_targets, dtype=np.int64),
            "inst_preds": np.asarray(all_preds, dtype=np.int64),
        }

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
        attn_mask=None,
    ):
        """
        Args:
            h: (B, N, D)
            attn_mask: (B, N), True for valid patches.
        """
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        h = self._encode_instances(h, attn_mask=attn_mask)
        B, N, D = h.shape

        A, h_flat = self.attention_net(h.reshape(B * N, D))
        A = A.reshape(B, N, -1)
        h = h_flat.reshape(B, N, D)

        if attention_only:
            return A

        A_raw = A.clone()
        A = self._masked_softmax(A, attn_mask=attn_mask, dim=1)

        M = torch.bmm(A.transpose(1, 2), h)
        logits = torch.cat([classifier(M[:, c, :]) for c, classifier in enumerate(self.classifiers)], dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if instance_eval:
            inst_results = self._instance_eval_mb(A, h, label, attn_mask=attn_mask)
            if inst_results is not None:
                results_dict.update(inst_results)

        if return_features:
            results_dict["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results_dict
