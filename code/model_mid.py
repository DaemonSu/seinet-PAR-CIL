import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLayer(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024, out_dim=1024, use_norm=True, alpha=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.use_norm = use_norm
        self.alpha = alpha  # 0~1之间，越大越保留原特征

    def forward(self, x):
        proj = self.mlp(x)
        out = self.alpha * x + (1 - self.alpha) * proj   # 残差混合
        if self.use_norm:
            out = F.normalize(out, p=2, dim=1)
        return out



class IncrementalContrastiveModel(nn.Module):
    """
    增量训练用的网络结构：
    encoder -> contrastive layer -> classifier
    """
    def __init__(self, encoder, classifier, in_dim=1024, hidden_dim=512, feat_dim=1024, use_norm=True):
        super(IncrementalContrastiveModel, self).__init__()
        self.encoder = encoder
        self.contrastive_layer = ContrastiveLayer(in_dim, hidden_dim, feat_dim, use_norm)
        self.classifier = classifier

    def forward(self, x, return_feat=False):
        """
        x: [B, C, H, W]
        return_feat: 是否返回对比层输出特征
        """
        feat = self.encoder(x)
        feat = self.contrastive_layer(feat)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        else:
            return logits
