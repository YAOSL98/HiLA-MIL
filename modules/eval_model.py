import torch
import torch.nn as nn
from einops import repeat
from .nystrom_attention import NystromAttention
from mamba_ssm import Mamba
import torch.nn.functional as F
import numpy as np

class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x



class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head_fusion='mean'):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x  


class SAttention(nn.Module):

    def __init__(self,mlp_dim=512,pos_pos=0,pos='ppeg',peg_k=7,head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

        self.layer1 = TransLayer(dim=mlp_dim)
        self.layer2 = TransLayer(dim=mlp_dim)
        self.model = Mamba(d_model=mlp_dim)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        self.patch_to_emb = [nn.Linear(1024, 512)]
        self.patch_to_emb += [nn.ReLU()]
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)
        self.dp = nn.Dropout(0.25)
        self.predictor = nn.Linear(512,4)

        # self.L = 512 #512
        # self.D = 128 #128
        # self.K = 1

        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        # )
        

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
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

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,mask_enable=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        x = x.unsqueeze(0)  # 在第0维增加一个 batch 维度


        batch, num_patches, C = x.shape 
        
        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)

        # masking
        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x,mask_ids,len_keep)

        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        x = self.layer1(x)
        # attn0.append(attn0.clone())

        # add pos embedding
        if self.pos_pos == 0:
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])
        
        # translayer2
        x = self.layer2(x)
        # attn1.append(attn1.clone())

        #---->cls_token
        h = self.norm(x)


        logits = h[:,0,:]

        logits = self.predictor(logits)

       
        return logits
