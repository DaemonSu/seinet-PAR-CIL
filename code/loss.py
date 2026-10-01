
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss_OpenSet(nn.Module):
    """
    SupConLoss for open-set scenario:
    - 闭集样本之间做标准 SupCon 正向对比
    - 开集样本仅作为负样本施加动态 margin
    """
    def __init__(self, temperature=0.13, base_margin=0.4, beta=0.2, eps=1e-8):
        super(SupConLoss_OpenSet, self).__init__()
        self.temperature = temperature
        self.base_margin = base_margin
        self.beta = beta
        self.eps = eps

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)

        # 分离闭集和开集索引
        known_mask = (labels != -1)
        unknown_mask = (labels == -1)

        feat_known = features[known_mask]
        labels_known = labels[known_mask]

        # ---------- 闭集 SupCon ----------
        if feat_known.size(0) > 1:
            sim_matrix_known = torch.matmul(feat_known, feat_known.T) / self.temperature
            labels_known = labels_known.view(-1, 1)
            mask_known = (labels_known == labels_known.T).float()
            logits_mask = 1 - torch.eye(feat_known.size(0), device=device)
            exp_sim = torch.exp(sim_matrix_known) * logits_mask
            log_prob = sim_matrix_known - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)
            mean_log_prob_pos = (mask_known * log_prob).sum(1) / (mask_known.sum(1) + self.eps)
            loss_known = -mean_log_prob_pos.mean()
        else:
            loss_known = torch.tensor(0.0, device=device)

        # ---------- 开集负样本正则 ----------
        if unknown_mask.any():
            feat_unknown = features[unknown_mask]
            # 计算已知闭集与开集相似度
            sim_known_unknown = torch.matmul(feat_known, feat_unknown.T) / self.temperature
            dynamic_margin = self.base_margin + self.beta * (1 - sim_known_unknown.detach())
            # 拉低闭集与开集的相似度
            loss_open = sim_known_unknown - dynamic_margin
            loss_open = F.relu(loss_open).mean()  # 超过 margin 才惩罚
        else:
            loss_open = torch.tensor(0.0, device=device)

        # ---------- 总 loss ----------
        loss = loss_known + loss_open
        return loss


# 普通对比 学习，主要用来做消融实验

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        # Normalize features
        features = F.normalize(features, dim=1)  # [B, D]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Mask: only consider valid positive pairs (same label, not self)
        labels = labels.contiguous().view(-1, 1)  # [B, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
        logits_mask = 1 - torch.eye(batch_size, device=device)  # mask out self-similarity

        # Compute exp similarities (excluding self-similarity)
        exp_sim = torch.exp(similarity_matrix) * logits_mask  # [B, B]

        # Log-probability for positive pairs
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)  # [B, 1]
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)  # [B]

        loss = -mean_log_prob_pos.mean()
        return loss

# 动态maring 对比学习
class SupConLoss_DynamicMargin(nn.Module):
    # def __init__(self, temperature=0.07, base_margin=0.3, beta=0.4):
    def __init__(self, temperature=0.08, base_margin=0.35, beta=0.25):
        super(SupConLoss_DynamicMargin, self).__init__()
        self.temperature = temperature
        self.base_margin = base_margin
        self.beta = beta
        self.eps = 1e-8

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        labels = labels.contiguous().view(-1, 1)  # [B, 1]
        valid_mask = (labels != -1).float()

        mask = (labels == labels.T).float() * (valid_mask @ valid_mask.T)
        logits_mask = 1 - torch.eye(batch_size, device=features.device)
        exp_sim = torch.exp(similarity_matrix) * logits_mask



        open_mask_row = (labels == -1).float()
        open_mask_col = (labels.T == -1).float()
        open_mask = open_mask_row @ torch.ones_like(open_mask_col) + torch.ones_like(open_mask_row) @ open_mask_col
        open_mask = torch.clamp(open_mask, 0, 1) * logits_mask

        # sim_before = similarity_matrix.clone()

        # Apply dynamic margin
        dynamic_margin = self.base_margin + self.beta * (1 - similarity_matrix.detach())
        similarity_matrix = similarity_matrix - open_mask * dynamic_margin
        exp_sim = torch.exp(similarity_matrix) * logits_mask

        # Final log_prob and loss
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        loss = -mean_log_prob_pos.mean()

        return loss



class ContrastiveLossWithCE(nn.Module):
    """
    Combining SupCon Loss and CrossEntropy Loss
    """
    def __init__(self, temperature=0.07, weight_ce=2):
        super(ContrastiveLossWithCE, self).__init__()
        self.supcon_loss = SupConLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce

    def forward(self, features, labels=None, logits=None):
        # 计算对比损失
        supcon_loss = self.supcon_loss(features, labels)

        # 如果提供了logits（分类输出），则计算CE损失
        if logits is not None:
            ce_loss = self.ce_loss(logits, labels)
            total_loss = supcon_loss + self.weight_ce * ce_loss
        else:
            total_loss = supcon_loss

        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupCon_OpenSet_Mixed(nn.Module):
    """
    Mixed SupCon loss for open-set:
    - known-known: standard supervised contrastive (same label -> positives)
    - unknown-unknown: pseudo-positives based on similarity threshold or topk
    - known-unknown: margin-based penalty for high sim pairs
    """
    def __init__(self,
                 temperature=0.06,
                 base_margin=0.4,
                 beta=0.2,
                 unknown_tau=0.7,
                 unknown_topk=4,
                 weight_known=1.5,
                 weight_unknown=0.4,
                 weight_known_unknown=1,
                 eps=1e-8,
                 device='cuda'):
        """
        unknown_tau: 如果设定，用余弦相似度 > tau 的 unknown-unknown 作为伪正对
        unknown_topk: 如果设定 (>0)，则对每个 unknown 取 top-k most similar unknown 做正对（优先于 tau）
        """
        super().__init__()
        self.temperature = temperature
        self.base_margin = base_margin
        self.beta = beta
        self.unknown_tau = unknown_tau
        self.unknown_topk = unknown_topk
        self.wk = weight_known
        self.wu = weight_unknown
        self.wku = weight_known_unknown
        self.eps = eps
        self.device = device

    def forward(self, features, labels):
        """
        features: [B, D] (float), expected NOT to be logit but embedding; we'll normalize
        labels: [B] (long) : known class labels >=0, unknown marked as -1
        """
        device = features.device
        B = features.size(0)
        feats = F.normalize(features, dim=1)

        # pairwise similarity (cosine) scaled by temperature
        sim = torch.matmul(feats, feats.T) / self.temperature  # [B, B]

        # masks
        labels = labels.view(-1)
        known_mask = labels != -1
        unknown_mask = labels == -1

        # ------- known-known SupCon -------
        loss_known = torch.tensor(0.0, device=device)
        if known_mask.sum() > 1:
            idx_known = torch.nonzero(known_mask).squeeze(1)
            feats_k = feats[idx_known]  # [Nk, D]
            sim_k = torch.matmul(feats_k, feats_k.T) / self.temperature  # [Nk, Nk]
            Nk = sim_k.size(0)

            labs_k = labels[idx_known].view(-1,1)
            pos_mask_k = (labs_k == labs_k.T).float()  # positives (including self)
            # remove self from denominator
            logits_mask = 1 - torch.eye(Nk, device=device)
            exp_sim_k = torch.exp(sim_k) * logits_mask
            log_prob_k = sim_k - torch.log(exp_sim_k.sum(1, keepdim=True) + self.eps)
            # mean log-prob for positives (exclude self if you want)
            pos_mask_k = pos_mask_k * logits_mask  # exclude self
            denom_pos = pos_mask_k.sum(1)
            denom_pos[denom_pos == 0] = 1  # avoid /0
            mean_log_prob_pos = (pos_mask_k * log_prob_k).sum(1) / (denom_pos + self.eps)
            loss_known = -mean_log_prob_pos.mean()

        # ------- unknown-unknown pseudo-positives -------
        loss_unknown = torch.tensor(0.0, device=device)
        if unknown_mask.sum() > 1:
            idx_u = torch.nonzero(unknown_mask).squeeze(1)
            feats_u = feats[idx_u]  # [Nu, D]
            Nu = feats_u.size(0)
            sim_u = torch.matmul(feats_u, feats_u.T)  # cosine sim (not divided by temp here, but can)
            # construct pseudo-positive mask for unknown-unknown
            if self.unknown_topk is not None and self.unknown_topk > 0:
                # for each row, pick topk (excluding self)
                topk = min(self.unknown_topk, Nu-1)
                _, topk_idx = torch.topk(sim_u - torch.eye(Nu, device=device)*10.0, k=topk, dim=1)
                pos_mask_u = torch.zeros((Nu, Nu), device=device)
                row_idx = torch.arange(Nu, device=device).unsqueeze(1).repeat(1, topk)
                pos_mask_u[row_idx, topk_idx] = 1.0
            else:
                # thresholding on cosine similarity
                pos_mask_u = (sim_u > self.unknown_tau).float()
                pos_mask_u = pos_mask_u - torch.eye(Nu, device=device)  # exclude self if tau > 1 maybe
                pos_mask_u = pos_mask_u.clamp(min=0.0)

            # now compute a supcon-style loss but only on unknown subset,
            # scale by temperature for consistency
            sim_u_scaled = sim_u / self.temperature
            logits_mask = 1 - torch.eye(Nu, device=device)
            exp_sim_u = torch.exp(sim_u_scaled) * logits_mask
            log_prob_u = sim_u_scaled - torch.log(exp_sim_u.sum(1, keepdim=True) + self.eps)
            denom_pos_u = pos_mask_u.sum(1)
            denom_pos_u[denom_pos_u == 0] = 1
            mean_log_prob_pos_u = (pos_mask_u * log_prob_u).sum(1) / (denom_pos_u + self.eps)
            # average only over rows that have at least one positive
            valid_rows = (pos_mask_u.sum(1) > 0).float()
            if valid_rows.sum() > 0:
                loss_unknown = -(mean_log_prob_pos_u * valid_rows).sum() / (valid_rows.sum() + self.eps)
            else:
                loss_unknown = torch.tensor(0.0, device=device)

        # ------- known-unknown dynamic margin penalty -------
        loss_known_unknown = torch.tensor(0.0, device=device)
        if known_mask.any() and unknown_mask.any():
            idx_k = torch.nonzero(known_mask).squeeze(1)
            idx_u = torch.nonzero(unknown_mask).squeeze(1)
            sim_ku = torch.matmul(feats[idx_k], feats[idx_u].T) / self.temperature  # [Nk, Nu]
            # dynamic margin: base + beta*(1 - sim_detach)
            dyn_margin = self.base_margin + self.beta * (1 - sim_ku.detach())
            # penalize only when sim_ku > dyn_margin (soft margin)
            pen = F.relu(sim_ku - dyn_margin)
            # mean over pairs
            loss_known_unknown = pen.mean()

        # combine
        loss = self.wk * loss_known + self.wu * loss_unknown + self.wku * loss_known_unknown
        return loss, {'loss_known': loss_known.item() if isinstance(loss_known, torch.Tensor) else loss_known,
                      'loss_unknown': loss_unknown.item() if isinstance(loss_unknown, torch.Tensor) else loss_unknown,
                      'loss_known_unknown': loss_known_unknown.item() if isinstance(loss_known_unknown, torch.Tensor) else loss_known_unknown}

