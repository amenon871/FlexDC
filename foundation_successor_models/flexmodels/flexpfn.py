"""Set-valued observation memory with independent query cross-attention."""
import torch
from torch import nn
from .episodes import encode
from .objective import behavior_vector,targets,decode_vector

class FlexPFN(nn.Module):
    def __init__(self,prior,config):
        super().__init__();self.prior=prior.requires_grad_(False).eval();self.config=dict(config);d=config['d_model'];c=prior.config
        self.global_observation=nn.Linear(c.linear_dim_hidden+2,d);self.job_observation=nn.Linear(c.linear_dim_hidden+c.st_dim_hidden+1,d)
        self.types=nn.Parameter(torch.randn(2,d)*.02)
        layer=nn.TransformerEncoderLayer(d,config['heads'],dim_feedforward=4*d,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.context_encoder=nn.TransformerEncoder(layer,config['layers'],enable_nested_tensor=False)
        self.global_query=nn.Linear(c.linear_dim_hidden,d);self.job_query=nn.Linear(c.linear_dim_hidden+c.st_dim_hidden,d)
        self.cross=nn.MultiheadAttention(d,config['heads'],dropout=0.,batch_first=True)
        self.norm=nn.LayerNorm(d);self.tracking_head=nn.Linear(d,4);self.qos_head=nn.Linear(d,2)
    def train(self,mode=True):super().train(mode);self.prior.eval();return self
    def forward(self,context,query):
        q=encode(self.prior,query);k=context['mask'].shape[1]
        if not k:return dict(q,variance=None,uncertainty_available=False)
        c=encode(self.prior,context);eps=self.config['qos_logit_epsilon'];residual=targets(context,eps)-behavior_vector(c,eps)
        ch=c['context_features'];ce=c['token_features'];qh=q['context_features'];qe=q['token_features'];b,queries,j=qe.shape[:3]
        globals_=self.global_observation(torch.cat([ch,residual[...,:2]],-1))+self.types[0]
        jobs=self.job_observation(torch.cat([ch.unsqueeze(-2).expand(*ce.shape[:-1],ch.shape[-1]),ce,residual[...,2:].unsqueeze(-1)],-1))+self.types[1]
        tokens=torch.cat([globals_.unsqueeze(-2),jobs],-2).flatten(1,2)
        valid=torch.cat([torch.ones(b,k,1,dtype=torch.bool,device=tokens.device),context['mask']],-1).flatten(1,2)
        memory=self.context_encoder(tokens,src_key_padding_mask=~valid)
        gq=self.global_query(qh)+self.types[0];jq=self.job_query(torch.cat([qh.unsqueeze(-2).expand(*qe.shape[:-1],qh.shape[-1]),qe],-1))+self.types[1]
        qt=torch.cat([gq.unsqueeze(-2),jq],-2);flat=qt.flatten(1,2)
        attended=self.cross(flat,memory,memory,key_padding_mask=~valid,need_weights=False)[0]
        z=self.norm(flat+attended).reshape(b,queries,j+1,-1)
        track=self.tracking_head(z[:,:,0]);qos=self.qos_head(z[:,:,1:])
        correction=torch.cat([track[...,:2],qos[...,0]],-1).float()
        logstd=torch.cat([track[...,2:],qos[...,1]],-1).float().clamp(-7,3)
        mu=behavior_vector(q,eps)+correction;out=decode_vector(mu,query['mask'])
        out.update(transformed_mean=mu,variance=torch.exp(2*logstd),uncertainty_available=True);return out
