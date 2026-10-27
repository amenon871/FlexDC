"""Random-initialized Foundation port of the proven masked V3 Set Transformer."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    dim_job_mix:int=13
    dim_dc_features:int=14
    st_dim_hidden:int=512
    st_num_heads:int=4
    st_num_outputs:int=1
    global_projection_dim:int=128
    linear_dim_hidden:int=512
    qos_projection_dim:int=256
    skip_connections:bool=True
    layer_norm:bool=True
    include_masked_mean_pool:bool=True
    def to_dict(self):return asdict(self)

class MaskedMAB(nn.Module):
    def __init__(self,dim_q,dim_k,dim_v,num_heads,layer_norm=True):
        super().__init__()
        if dim_v%num_heads:raise ValueError('Hidden width must divide attention heads')
        self.dim_v=dim_v;self.num_heads=num_heads;self.head_dim=dim_v//num_heads
        self.fc_q=nn.Linear(dim_q,dim_v);self.fc_k=nn.Linear(dim_k,dim_v);self.fc_v=nn.Linear(dim_k,dim_v);self.fc_o=nn.Linear(dim_v,dim_v)
        self.ln0=nn.LayerNorm(dim_v) if layer_norm else None;self.ln1=nn.LayerNorm(dim_v) if layer_norm else None
    def forward(self,q_input,k_input,key_mask=None,query_mask=None):
        def split(x):return x.reshape(x.shape[0],x.shape[1],self.num_heads,self.head_dim).transpose(1,2)
        q=split(self.fc_q(q_input));k=split(self.fc_k(k_input));v=split(self.fc_v(k_input))
        scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        if key_mask is not None:scores=scores.masked_fill(~key_mask[:,None,None,:],torch.finfo(scores.dtype).min)
        out=q+scores.softmax(-1)@v;out=out.transpose(1,2).contiguous().reshape(q_input.shape[0],q_input.shape[1],self.dim_v)
        if self.ln0 is not None:out=self.ln0(out)
        out=out+F.silu(self.fc_o(out))
        if self.ln1 is not None:out=self.ln1(out)
        if query_mask is not None:out=out*query_mask.unsqueeze(-1)
        return out

class MaskedSAB(nn.Module):
    def __init__(self,din,dout,heads,layer_norm=True):
        super().__init__();self.mab=MaskedMAB(din,din,dout,heads,layer_norm)
    def forward(self,x,mask):return self.mab(x,x,key_mask=mask,query_mask=mask)

class MaskedPMA(nn.Module):
    def __init__(self,dim,heads,seeds,layer_norm=True):
        super().__init__();self.seed=nn.Parameter(torch.empty(1,seeds,dim));nn.init.xavier_uniform_(self.seed);self.mab=MaskedMAB(dim,dim,dim,heads,layer_norm)
    def forward(self,x,mask):
        seeds=self.seed.expand(x.shape[0],-1,-1)
        return self.mab(seeds,x,key_mask=mask,query_mask=torch.ones(seeds.shape[:2],dtype=torch.bool,device=x.device))

class FoundationBaseline(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=ModelConfig(**config) if isinstance(config,dict) else config or ModelConfig();c=self.config
        self.sab1=MaskedSAB(c.dim_job_mix,c.st_dim_hidden,c.st_num_heads,c.layer_norm);self.sab2=MaskedSAB(c.st_dim_hidden,c.st_dim_hidden,c.st_num_heads,c.layer_norm)
        self.pma=MaskedPMA(c.st_dim_hidden,c.st_num_heads,c.st_num_outputs,c.layer_norm)
        self.global_encoder=nn.Sequential(nn.Linear(c.dim_dc_features,c.global_projection_dim),nn.SiLU(),nn.Linear(c.global_projection_dim,c.global_projection_dim),nn.SiLU())
        pooled=c.st_dim_hidden*(2 if c.include_masked_mean_pool else 1)+c.global_projection_dim
        self.linear1=nn.Linear(pooled,c.linear_dim_hidden);self.linear2=nn.Linear(c.linear_dim_hidden,c.linear_dim_hidden);self.linear3=nn.Linear(c.linear_dim_hidden,c.linear_dim_hidden);self.tracking_output=nn.Linear(c.linear_dim_hidden,2)
        self.qos_token_projection=nn.Linear(c.st_dim_hidden,c.qos_projection_dim);self.qos_context_projection=nn.Linear(c.linear_dim_hidden,c.qos_projection_dim)
        self.qos_head=nn.Sequential(nn.Linear(2*c.qos_projection_dim,c.qos_projection_dim),nn.SiLU(),nn.Linear(c.qos_projection_dim,1))
    def forward(self,globals_,jobs,mask,return_latents=False):
        if jobs.ndim!=3 or globals_.shape!=(jobs.shape[0],self.config.dim_dc_features) or jobs.shape[-1]!=self.config.dim_job_mix or mask.shape!=jobs.shape[:2]:raise ValueError('Baseline feature/mask dimensions disagree')
        if not mask.any(-1).all():raise ValueError('Every row needs real job tokens')
        # Mask inputs as well as keys/queries: padding has exactly zero gradient.
        jobs=jobs*mask.unsqueeze(-1);e=self.sab2(self.sab1(jobs,mask),mask)
        pools=[self.pma(e,mask).mean(1)]
        if self.config.include_masked_mean_pool:pools.append((e*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True))
        fused=torch.cat([*pools,self.global_encoder(globals_)],-1)
        x2=F.silu(self.linear1(fused));x3=F.silu(self.linear2(x2))
        h=F.silu(self.linear3(x3+x2))+x3 if self.config.skip_connections else F.silu(self.linear3(x3))
        q=self.qos_head(torch.cat([self.qos_token_projection(e),self.qos_context_projection(h)[:,None,:].expand(-1,e.shape[1],-1)],-1)).squeeze(-1)
        result=dict(tracking_logs=self.tracking_output(h),qos_probabilities=q.sigmoid()*mask,qos_logits=q,mask=mask)
        if return_latents:result.update(context_features=h,token_features=e)
        return result
