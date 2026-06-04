import torch
import matplotlib.pyplot as plt


def c_2int(f_list, dx1, dx2, string):
    integral = torch.einsum(string, f_list) * dx1 * dx2
    return integral

def c_1int(f_list, dx, string):
    integral = torch.einsum(string, f_list) * dx
    return integral
