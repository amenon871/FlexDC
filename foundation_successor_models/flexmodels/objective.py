"""One canonical behavior transform, feasibility rule and FlexDC objective."""
import torch
import torch.nn.functional as F

MEAN_FLOOR = 1e-6
P90_FLOOR = 1e-3
TRACKING_LIMIT = 0.30
QOS_LIMIT = 0.10
P90_BOUNDARY = (.20,.40)
QOS_BOUNDARY = (.05,.15)
CONSTANTS = dict(psi1=1., psi2=10., beta=20., rho=2., tracking_error_constraint=.3,
                 qos_constraint=.1, program_type='RSR')


def physical(output):
    logs = output['tracking_logs'].float()
    return (logs[..., 0].exp()-MEAN_FLOOR).clamp_min(0), (logs[..., 1].exp()-P90_FLOOR).clamp_min(0), output['qos_probabilities'].float()


def behavior_vector(output, epsilon):
    return torch.cat([output['tracking_logs'].float(), torch.logit(output['qos_probabilities'].float().clamp(epsilon,1-epsilon))], -1)


def decode_vector(values, mask):
    return dict(tracking_logs=values[..., :2], qos_probabilities=values[..., 2:].sigmoid()*mask, mask=mask)


def targets(batch, epsilon=None):
    qos = batch['qos'] if epsilon is None else torch.logit(batch['qos'].clamp(epsilon,1-epsilon))
    return torch.cat([batch['tracking_logs'], qos], -1)


def reconstruct(output, batch):
    mean,p90,pj = physical(output); mask=batch['mask']
    track = batch['r_actual_watts']*mean*batch['price_coefficient']*batch['hour_seconds']
    rsr = batch['power_cost']+track
    ctrack = F.softplus(CONSTANTS['psi2']*(p90-TRACKING_LIMIT))*CONSTANTS['psi1']
    cqos = (CONSTANTS['beta']*F.softplus(CONSTANTS['rho']*(pj-QOS_LIMIT))*mask).sum(-1)
    tracking_pass = p90 <= TRACKING_LIMIT
    qos_pass = ((pj <= QOS_LIMIT) | ~mask).all(-1)
    return dict(mean_tracking=mean,p90=p90,pj=pj,m_power=batch['power_cost'],m_track=track,
                m_rsr=rsr,tracking_cost=ctrack,qos_cost=cqos,objective=rsr+ctrack+cqos,
                tracking_feasible=tracking_pass,qos_feasible=qos_pass,joint_feasible=tracking_pass&qos_pass)


def boundary_weights(batch, config):
    p = batch['p90'];q=batch['qos']
    wp=1+(config['tracking_boundary_multiplier']-1)*((p>=P90_BOUNDARY[0])&(p<=P90_BOUNDARY[1])).float()
    wq=batch['mask'].float()*(1+(config['qos_boundary_multiplier']-1)*((q>=QOS_BOUNDARY[0])&(q<=QOS_BOUNDARY[1])).float())
    return wp,wq


def behavior_loss(output, batch, normalization, config):
    pred=output['tracking_logs'].float();truth=batch['tracking_logs'];scale=pred.new_tensor(normalization['target_std'])
    loss=F.smooth_l1_loss(pred/scale,truth/scale,reduction='none')
    wp,wq=boundary_weights(batch,config)
    mean=loss[...,0].mean();p90=(loss[...,1]*wp).sum()/wp.sum().clamp_min(1)
    qos=((output['qos_probabilities'].float()-batch['qos']).square()*wq).sum()/wq.sum().clamp_min(1)
    total=mean+p90+config['qos_weight']*qos
    return total,dict(mean=mean,p90=p90,qos=qos)


def predictive_loss(output, batch, config, epsilon):
    y=targets(batch,epsilon);mu=output['transformed_mean'].float();v=output['variance'].float().clamp_min(1e-8)
    nll=.5*((y-mu).square()/v+v.log()+1.8378770664093453)
    wp,wq=boundary_weights(batch,config)
    mean=nll[...,0].mean();p90=(nll[...,1]*wp).sum()/wp.sum().clamp_min(1)
    qos=(nll[...,2:]*wq).sum()/wq.sum().clamp_min(1)
    return mean+p90+config['qos_weight']*qos,dict(mean=mean,p90=p90,qos=qos,nll=nll.mean())
