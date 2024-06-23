from torch import nn
import torch

class CEMLoss(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.c1 = nn.Linear(hidden_dim, 1)
        self.c2 = nn.Linear(hidden_dim//16, 1)

        self.c3 = nn.Linear(hidden_dim, hidden_dim//16)

    def forward(self, rec_feat, res_feat):
        B, n_ph, n_q, c = rec_feat.shape
        rec_feat = rec_feat.view(B, -1, c)
        res_feat = res_feat.reshape(B, c//16, -1).transpose(1, 2)

        es = nn.functional.softmax(self.c1(rec_feat), dim=-2)
        ec = nn.functional.softmax(self.c2(res_feat), dim=-2)

        rec_feat = nn.functional.normalize(self.c3(rec_feat), dim=-1)
        res_feat = nn.functional.normalize(res_feat, dim=-1).transpose(-1, -2)

        tsc = torch.bmm(rec_feat, res_feat)
        tsc = torch.clamp((tsc + 1.) / 2., 1e-6, 1.-1e-6)
        energy = torch.bmm(es.transpose(-1, -2), tsc)
        energy = torch.bmm(energy, ec)

        return -1.0 * torch.sum(torch.log(energy+1e-6)) * 1.0 / B