import torch
import torch.nn as nn
from typing import List, Iterable

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + eps)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(isinstance(p, torch.nn.Parameter) for p in params)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Helper to get group specific args, though usually they are consistent
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                if g.ndim > 2:
                    # Flatten conv kernels to 2D: (out_channels, in_channels * kH * kW)
                    g = g.view(g.size(0), -1)
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                
                # Update momentum buffer
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                # Orthogonalize update
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                # Scale update
                # update *= max(1, g.size(0) / g.size(1))**0.5 # Removed from original impl in favor of control
                # The original code had: update *= max(1, grad.size(-2) / grad.size(-1))**0.5
                # Let's keep it to be faithful to the reference implementation
                if g.size(0) < g.size(1):
                     update = update * (g.size(1) / g.size(0))**0.5

                # Apply update
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update.view_as(p.data), alpha=-lr)

        return loss
