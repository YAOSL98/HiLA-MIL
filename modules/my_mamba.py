import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from .nystrom_attention import NystromAttention
import h5py  # 确保导入 h5py 库

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = head,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
            x = x+z
            return x,attn
        else:
            x = x + self.attn(self.norm(x))
            return x  

class MAMBA_block_MY(nn.Module):

    def __init__(self,mlp_dim=512):
        super(MAMBA_block_MY, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.norm = nn.LayerNorm(mlp_dim)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        self.model = Mamba(d_model=mlp_dim)
        self.model2 = Mamba(d_model=mlp_dim)

        self.apply(initialize_weights)

    def masking(self, x, ids_shuffle=None,len_keep=None):
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def norm(self, new_slide_features):
        new_slide_features = new_slide_features.transpose(0,1)
        attns_min = new_slide_features.min()
        attns_max = new_slide_features.max()
        new_slide_features = (new_slide_features - attns_min) / (attns_max - attns_min)  # 显式归一化
        return new_slide_features

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,no_norm=False,mask_enable=False):
        if mask_enable and mask_ids is not None:
            x, _,_ = self.masking(x,mask_ids,len_keep)
        features = self.model(x)
        features = self.model2(features)
        features = features.squeeze(0)
        A = x.clone().squeeze(0) 
        A = self.attention(A)
        A = torch.transpose(A, -1, -2)  # KxN

        A_ori = A.clone().unsqueeze(0)
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, features)  # KxL
        Y_prob = M
        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob