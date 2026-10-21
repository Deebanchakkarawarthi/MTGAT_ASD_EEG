import torch
from torch import nn

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.src = nn.Parameter(torch.empty(out_dim))
        self.dst = nn.Parameter(torch.empty(out_dim))
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.src, std=0.1)
        nn.init.normal_(self.dst, std=0.1)
    def forward(self, x, adjacency):
        h = self.proj(x)
        e = self.act((h*self.src).sum(-1).unsqueeze(2) +
                     (h*self.dst).sum(-1).unsqueeze(1))
        e = e.masked_fill(adjacency <= 0, torch.finfo(e.dtype).min)
        alpha = self.drop(torch.softmax(e, dim=-1))
        return torch.bmm(alpha, h), alpha

class FiLM(nn.Module):
    def __init__(self, context_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(context_dim, feature_dim)
        self.beta = nn.Linear(context_dim, feature_dim)
    def forward(self, x, context):
        return (1 + self.gamma(context).unsqueeze(1))*x + self.beta(context).unsqueeze(1)

class MTGAT(nn.Module):
    # node_features [B,T,K,N,F]; adjacency [B,T,K,N,N]
    def __init__(self, node_feature_dim=1, hidden_dim=64, num_classes=2,
                 num_paradigms=1, phenotype_classes=0, state_classes=0, dropout=0.3):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_feature_dim, hidden_dim, dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.film = FiLM(max(1, num_paradigms), hidden_dim*2)
        self.drop = nn.Dropout(dropout)
        self.asd_head = nn.Linear(hidden_dim*2, num_classes)
        self.phenotype_head = nn.Linear(hidden_dim*2, phenotype_classes) if phenotype_classes else None
        self.state_head = nn.Linear(hidden_dim*2, state_classes) if state_classes else None
    def forward(self, node_features, adjacency, paradigm_ids=None):
        x, a = node_features.mean(2), adjacency.mean(2)
        seq, attentions = [], []
        for t in range(x.shape[1]):
            h, attn1 = self.gat1(x[:,t], a[:,t])
            h, attn2 = self.gat2(torch.relu(h), a[:,t])
            seq.append(torch.relu(h).mean(1))
            attentions.append((attn1, attn2))
        temporal, _ = self.gru(torch.stack(seq, dim=1))
        if paradigm_ids is not None:
            ctx = torch.nn.functional.one_hot(paradigm_ids.long(),
                    num_classes=self.film.gamma.in_features).float()
            temporal = self.film(temporal, ctx)
        pooled = self.drop(temporal.mean(1))
        out = {"asd_logits": self.asd_head(pooled), "attention": attentions}
        if self.phenotype_head is not None: out["phenotype_logits"] = self.phenotype_head(pooled)
        if self.state_head is not None: out["state_logits"] = self.state_head(temporal)
        return out
