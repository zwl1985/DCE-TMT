import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(HypergraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, H):
        W = torch.ones(H.size(0), H.size(2), 1, device=H.device)
        Dv = torch.bmm(H, W).squeeze(-1)
        Dv_inv = torch.pow(Dv + 1e-9, -1)
        Dv_inv_mat = torch.diag_embed(Dv_inv)

        De = torch.sum(H, dim=1)
        De_inv = torch.pow(De + 1e-9, -1)
        De_inv_mat = torch.diag_embed(De_inv)

        x = torch.matmul(x, self.weight)
        HTX = torch.bmm(H.transpose(1, 2), x)
        DeHTX = torch.bmm(De_inv_mat, HTX)
        HDeHTX = torch.bmm(H, DeHTX)
        out = torch.bmm(Dv_inv_mat, HDeHTX)

        return self.dropout(F.relu(out))


class HypergraphConv_mod(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_ratio, edge_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.hgnn = HypergraphConv(hidden_dim, hidden_dim, dropout=dropout_ratio)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, weight_adj, adj_dep=None):
        B, L, d = x.size()
        H_sem = weight_adj.mean(dim=1)
        if adj_dep is not None:
            H_syn = adj_dep.float()
            H = torch.cat([H_sem, H_syn], dim=2)
        else:
            H = H_sem
        out = self.hgnn(x, H)
        out = self.W_h(out)
        return out


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


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, aspect, weight_m, bias_m, mask, dropout, short):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    batch = scores.size(0)
    p = weight_m.size(0)
    max_dim = weight_m.size(1)

    weight_m = weight_m.unsqueeze(0).expand(batch, p, max_dim, max_dim)

    aspect_scores = torch.tanh(
        torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m)
    )
    scores = scores + aspect_scores

    if short is not None:
        scores = scores + short

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        self.weight_m = nn.Parameter(torch.randn(self.h, self.d_k, self.d_k))
        self.bias_m = nn.Parameter(torch.ones(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, mask, aspect, short):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]

        batch, aspect_dim = aspect.size(0), aspect.size(1)
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)
        aspect = self.dense(aspect)
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size(2), self.d_k)

        attn = attention(
            query, key, aspect, self.weight_m, self.bias_m, mask, self.dropout, short
        )
        return attn


class DCE_TMT(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = bert

        self.deprel_emb = nn.Embedding(
            opt.deprel_size + 1, opt.deprel_dim, padding_idx=0
        ) if hasattr(opt, 'deprel_dim') and opt.deprel_dim > 0 else None

        self.linear_in = nn.Linear(opt.bert_dim, opt.hidden_dim)
        self.linear_out = nn.Linear(opt.hidden_dim + opt.bert_dim, opt.polarities_dim)

        self.sentic_linear = nn.Linear(1, opt.hidden_dim)

        self.ate_linear = nn.Linear(opt.hidden_dim, 1)

        self.bert_drop = nn.Dropout(opt.bert_dropout if hasattr(opt, 'bert_dropout') else 0.1)
        self.pooled_drop = nn.Dropout(opt.bert_dropout if hasattr(opt, 'bert_dropout') else 0.1)
        self.ffn_dropout = opt.ffn_dropout

        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()

        norm_class = nn.LayerNorm if opt.norm == 'ln' else nn.BatchNorm1d

        for _ in range(opt.num_layers):
            graph_conv = HypergraphConv_mod(
                hidden_dim=opt.hidden_dim,
                num_heads=opt.graph_conv_attention_heads,
                dropout_ratio=opt.ffn_dropout,
                edge_dim=opt.deprel_dim
            )
            self.graph_convs.append(graph_conv)
            self.norms.append(norm_class(opt.hidden_dim))
            self.transformer_layers.append(
                TransformerLayer(
                    opt.hidden_dim,
                    opt.attention_heads,
                    attn_dropout_ratio=opt.attn_dropout,
                    ffn_dropout_ratio=opt.ffn_dropout,
                    norm=opt.norm
                )
            )

        self.attn = MultiHeadAttention(opt.attention_heads, opt.hidden_dim)
        self.graph_conv_type = opt.graph_conv_type
        self.attention_heads = opt.attention_heads

    def forward(self, inputs):
        if len(inputs) == 6:
            text_bert_indices, bert_segments_ids, attention_mask, adj_dep, src_mask, aspect_mask = inputs
            sentic_scores = None
        elif len(inputs) == 7:
            text_bert_indices, bert_segments_ids, attention_mask, adj_dep, sentic_scores, src_mask, aspect_mask = inputs
        else:
            raise ValueError(f"Expected 6 or 7 inputs, got {len(inputs)}")

        token_type_vocab_size = self.bert.embeddings.token_type_embeddings.num_embeddings
        bert_segments_ids = torch.clamp(bert_segments_ids, 0, token_type_vocab_size - 1)

        device = next(self.bert.parameters()).device
        text_bert_indices = text_bert_indices.to(device)
        bert_segments_ids = bert_segments_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.bert(
            input_ids=text_bert_indices,
            attention_mask=attention_mask
        )

        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        h = self.linear_in(gcn_inputs)
        B, L, d = h.size()

        if hasattr(self.opt, 'use_knowledge') and self.opt.use_knowledge and sentic_scores is not None:
            h = h + self.sentic_linear(sentic_scores.unsqueeze(-1).float())

        aspect_outs = (h * aspect_mask.unsqueeze(-1)).sum(dim=1) / \
                      aspect_mask.sum(dim=1, keepdim=True)

        range_tensor = torch.arange(L, device=h.device).float()
        dist_mat = torch.abs(range_tensor.unsqueeze(0) - range_tensor.unsqueeze(1))
        short_bias = -dist_mat.unsqueeze(0).unsqueeze(0)
        short_bias = short_bias.expand(B, -1, -1, -1)

        weight_adj = self.attn(h, h, src_mask.unsqueeze(-2), aspect_outs, short_bias)

        e = self.deprel_emb(adj_dep) if self.deprel_emb is not None else None

        for i in range(self.opt.num_layers):
            h0 = h

            if self.graph_conv_type == 'hgnn':
                h = self.graph_convs[i](h, weight_adj, adj_dep)
            else:
                h = self.graph_convs[i](h, adj_dep, e)

            h = self.norms[i](h)
            h = h.relu()
            h = F.dropout(h, self.ffn_dropout, training=self.training)

            h = self.transformer_layers[i](h, src_mask)
            h = h + h0

        aspect_words_num = aspect_mask.sum(dim=1).unsqueeze(-1)
        graph_out = (h * aspect_mask.unsqueeze(-1)).sum(dim=1) / aspect_words_num

        out = torch.cat([graph_out, pooled_output], dim=-1)

        self.features = F.normalize(out, dim=-1)

        out = self.linear_out(out)

        ate_logits = self.ate_linear(h).squeeze(-1)

        return out, ate_logits