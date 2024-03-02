import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PromptTunnerModel(nn.Module):
    def __init__(self, args=None):
        super(PromptTunnerModel, self).__init__()
        self.args = args
        if not self.args.tunner_for_visual:
            self.q_conv = nn.Linear(1024, 768)
            self.k_conv = nn.Linear(768, 768)
            self.v_conv = nn.Linear(768, 768)
            
            # self.t_transform = nn.Linear(77, 256)
            self.out_conv = nn.Sequential(nn.Linear(768, 768),
                                        nn.LayerNorm(768),
                                        nn.LeakyReLU(),
                                        nn.Linear(768, 768))
        else:
            self.proj_v = nn.Linear(768, 768)
            self.proj_t = nn.Linear(768, 768)
        
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    
    def forward(self, clip_t, plug_in_embeddings):
                        
        bsz = plug_in_embeddings.shape[0]
        # means that we should project both the text and visual feats
        plug_in_embeddings = plug_in_embeddings.permute(0, 2, 1) # [B, 768, 256]
        plug_in_embeddings = F.max_pool1d(plug_in_embeddings, kernel_size=plug_in_embeddings.size(2))# [B, 768, 1]
        plug_in_embeddings = plug_in_embeddings.permute(0, 2, 1) #[B, 768]
        
        vis = self.proj_v(plug_in_embeddings)
        norm_vis = vis / vis.norm(dim=-1, keepdim=True)

        text = self.proj_t(clip_t).squeeze(1) #[B, 768]
        text = text.unsqueeze(0).repeat(bsz, 1 ,1)
        norm_text = text / text.norm(dim=-1, keepdim=True)

        score = torch.bmm(norm_vis, norm_text.transpose(1,2)).squeeze(1)
        logit_scale = self.logit_scale.exp()
        score = logit_scale * score
        return score


