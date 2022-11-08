import torch
import torch.nn as nn
from torch.optim import Optimizer


@torch.no_grad()
def eki_update(params, update):
    torch._foreach_add_(params, update)


class Ensemble(nn.modules):
    def __init__(self, num_particles, model_cls, config) -> None:
        super().__init__()
        self.models = nn.ModuleList([
            model_cls(config) for i in range(num_particles)
        ])
        pass
    
    def init_particles(self, num_particles):
        '''
        initialize [num_particles] particles
        Args:
            - num_particles: number of particles 
        '''
        particles = []
        return particles

    @torch.no_grad()
    def forward(self, x, forward_fn) -> None:
        '''
        get forward values
        '''
        pred_list = []
        for model in self.models:
            pred = forward_fn(model, x)
            pred_list.append(pred)
        self.pred_vec = torch.stack(pred_list, dim=0)    # num_models, batchsize
        self.pred_mean = torch.mean(self.pred_vec)
        
    def update(self, y) -> None:
        '''
        Update parameters for each particle
        Args:
            - y: array 
        '''
        for i, model in enumerate(self.models):
            err = self.pred_vec[i] - y
            params = list(model.parameters())
            update = 0.
            for j, model_j in enumerate(self.models):
                left = self.pred_vec[j] - self.pred_mean
                coeff = torch.dot(left, err)
                params_j = list(model_j.parameters())
                grad = torch._foreach_mul(params_j, coeff)
                update = torch._foreach_add(params, grad)

