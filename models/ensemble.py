import torch
import torch.nn as nn


class Ensemble(nn.modules):
    def __init__(self, num_particles) -> None:
        super().__init__()
        pass
    
    def init_particles(self, num_particles):
        '''
        initialize [num_particles] particles
        Args:
            - num_particles: number of particles 
        '''
        particles = []
        return particles

    def forward(self, x) -> None:
        '''
        get forward values
        '''
        pass

    def update(self) -> None:
        '''
        Update parameters for each particle
        '''
