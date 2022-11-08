import torch


def explicit_solve(closure, f, bc, dt=1.0, Nt=1000, L=1.0):
    """_summary_

    Args:
        closure (callable): closure is a function of q,t
        f (tensor): forcing term
        bc (list): boundary conditions
        dt (float): time step
        Nt (int): number of time steps
        L (float): 
    Return: 
        solution
    """
    yy = torch.linspace(bc[0], bc[1], f.shape[0])