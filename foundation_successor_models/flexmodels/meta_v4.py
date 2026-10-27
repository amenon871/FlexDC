"""Frozen Foundation prior + compact, differentiable exact residual GPs."""
import torch
from torch import nn
from torch.nn import functional as F
from .episodes import encode
from .objective import behavior_vector,targets,decode_vector

class MetaV4(nn.Module):
    def __init__(self,prior,config):
        super().__init__();self.prior=prior.requires_grad_(False).eval();self.config=dict(config);d=config['latent_dim'];c=prior.config
        self.global_projection=nn.Sequential(nn.Linear(c.linear_dim_hidden,d),nn.SiLU(),nn.Linear(d,d))
        self.job_projection=nn.Sequential(nn.Linear(c.linear_dim_hidden+c.st_dim_hidden,d),nn.SiLU(),nn.Linear(d,d))
        self.raw_length=nn.Parameter(torch.zeros(3,d));self.raw_scale=nn.Parameter(torch.zeros(3));self.raw_noise=nn.Parameter(torch.full((3,),-5.))
        self.last_jitter=0.
    def train(self,mode=True):super().train(mode);self.prior.eval();return self
    def kernel(self,x,y,channels):
        ell=F.softplus(self.raw_length[channels])+1e-4;scale=F.softplus(self.raw_scale[channels])+1e-6
        delta=(x.unsqueeze(-2)-y.unsqueeze(-3))/ell[None,:,None,None,:]
        radius=delta.square().sum(-1).clamp_min(1e-12).sqrt();r=5**.5*radius
        return scale[None,:,None,None]*(1+r+r.square()/3)*torch.exp(-r)
    def posterior(self,cx,qx,residual,channels):
        covariance=self.kernel(cx,cx,channels);noise=F.softplus(self.raw_noise[channels])+1e-6
        eye=torch.eye(cx.shape[-2],device=cx.device,dtype=cx.dtype);jitter=float(self.config['jitter'])
        for _ in range(7):
            chol,info=torch.linalg.cholesky_ex(covariance+(noise[None,:,None,None]+jitter)*eye)
            if not info.any():break
            jitter*=10
        else:raise RuntimeError('GP Cholesky failed after bounded jitter escalation')
        self.last_jitter=jitter;cross=self.kernel(cx,qx,channels)
        alpha=torch.cholesky_solve(residual.unsqueeze(-1),chol)
        mean=(cross.transpose(-2,-1)@alpha).squeeze(-1)
        scale=F.softplus(self.raw_scale[channels])+1e-6
        latent=(scale[None,:,None]-(cross*torch.cholesky_solve(cross,chol)).sum(-2)).clamp_min(1e-8)
        return mean,latent+noise[None,:,None],latent
    def projected(self,output):
        h=output['context_features'].float();e=output['token_features'].float();g=self.global_projection(h)
        job=self.job_projection(torch.cat([h.unsqueeze(-2).expand(*e.shape[:-1],h.shape[-1]),e],-1))
        return torch.cat([g.unsqueeze(-2).expand(*g.shape[:-1],2,g.shape[-1]),job],-2).permute(0,2,1,3)
    def forward(self,context,query):
        q=encode(self.prior,query);k=context['mask'].shape[1]
        if not k:return dict(q,variance=None,uncertainty_available=False)
        c=encode(self.prior,context);epsilon=self.config['qos_logit_epsilon'];j=query['mask'].shape[-1]
        # Cholesky/kernel operations are ALWAYS float32, including bf16 training.
        with torch.autocast(device_type=query['jobs'].device.type,enabled=False):
            cx=self.projected(c);qx=self.projected(q);channels=torch.tensor([0,1]+[2]*j,device=cx.device)
            residual=(targets(context,epsilon)-behavior_vector(c,epsilon)).float().transpose(1,2)
            correction,var,latent=self.posterior(cx,qx,residual,channels)
            mu=behavior_vector(q,epsilon)+correction.transpose(1,2)
            out=decode_vector(mu,query['mask']);out.update(transformed_mean=mu,variance=var.transpose(1,2),latent_variance=latent.transpose(1,2),uncertainty_available=True)
            return out
