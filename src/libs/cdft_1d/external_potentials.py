import torch

def LJ71(x, x_wall, Ew, sigmaw):
    Vext = (
        torch.pi*sigmaw**3 *Ew
        * (
            (sigmaw / (x - x_wall))**7 *(4/15/7) 
            - (sigmaw / (x - x_wall))**1 *(2) 
        ) #* torch.sigmoid(-(x - x_wall - 2.5) / (0.001))
    )[None, :]
    return Vext

def LJ93(x, x_wall, Ew, sigmaw):
    Vext = (
        4/3*torch.pi*sigmaw**3 *Ew
        * (
            (sigmaw / (x - x_wall))**9 /15 
            - (sigmaw / (x - x_wall))**3 /2 
        ) 
    )[None, :]
    return Vext

def LJ93(x, x_wall, Ew, sigmaw):
    Vext = (
        Ew
        * (
            (2/15)*(sigmaw / (x - x_wall))**9  
            - (sigmaw / (x - x_wall))**3 
        ) 
    )[None, :]
    return Vext

def LJ126(x, x_wall, Ew, sigmaw):
    Vext = (
        4*Ew
        * (
            (sigmaw / (x - x_wall))**12 
            - (sigmaw / (x - x_wall))**6 
        )
    )[None, :]
    return Vext

def HW(x, x_wall, Ew=None, sigmaw=None):
    Vext = ((x < x_wall).double()*1e6)  [None, :]
    return Vext