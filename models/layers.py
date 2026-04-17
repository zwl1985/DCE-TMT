import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio, norm='ln'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_attn_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(ffn_dropout_ratio)
        )
        norm_class = nn.LayerNorm if norm == 'ln' else nn.BatchNorm1d
        self.norm1 = norm_class(hidden_dim)
        self.norm2 = norm_class(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout_ratio),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(ffn_dropout_ratio)
        )

    def forward(self, x, src_mask):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dim_split = self.hidden_dim // self.attention_heads
        q_heads = torch.cat(q.split(dim_split, 2), dim=0)
        k_heads = torch.cat(k.split(dim_split, 2), dim=0)
        v_heads = torch.cat(v.split(dim_split, 2), dim=0)
        attention_score = q_heads.bmm(k_heads.transpose(1, 2))
        attention_score = attention_score / math.sqrt(self.hidden_dim // self.attention_heads)
        inf_mask = (~src_mask).unsqueeze(1).to(dtype=torch.float) * -1e9
        inf_mask = torch.cat([inf_mask for _ in range(self.attention_heads)], 0)
        A = torch.softmax(attention_score + inf_mask, -1)
        A = self.attn_dropout(A)
        attn_out = torch.cat((A.bmm(v_heads)).split(q.size(0), 0), 2)
        attn_out = self.linear_attn_out(attn_out)
        attn_out = attn_out + x
        attn_out = self.norm1(attn_out)
        out = self.ffn(attn_out) + attn_out
        out = self.norm2(out)
        return out


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown contrast_mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask.repeat(anchor_count, contrast_count)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss