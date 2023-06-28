import torch
import matplotlib.pyplot as plt
import numpy as np

""" Selection of imports and helper functions. """

def spread_plot(
        means: list,
        stds:list,
        maxima: list,
        minima: list,
        save_folder: str,
        file_name: str,
        epoch: int):
    """ Plot the curriculum scores as an errorplot with the std and max / min values to indicate data spread. """
    
    epochs = np.arange(epoch + 1)

    plt.errorbar(
        x=epochs,
        y=means,
        yerr=stds, 
        linestyle=':',
        marker='_',
        ecolor='black',
        markersize=10,
        markeredgecolor='blue',
        markeredgewidth=2)

    plt.scatter(x=epochs, y=maxima, color='green')
    plt.scatter(x=epochs, y=minima, color='red')

    plt.savefig(save_folder + file_name)
    plt.close()


def basic_plot(data, save_folder, file_name, epoch):
    plt.plot(np.arange(epoch + 1), data)
    plt.savefig(save_folder + file_name)
    plt.close()
    

def max_norm_normalize(x):
    return x / torch.amax(x.norm(dim=1).flatten(1, -1), dim=-1)[:, None, None, None]


def hyperbolic_norm(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    sqrt_c = torch.sqrt(c)
    return (2. / sqrt_c) * torch.arctanh(sqrt_c * torch.linalg.norm(x, dim=1))


def compute_alpha(iters, beta):
    return -beta * iters + 1.