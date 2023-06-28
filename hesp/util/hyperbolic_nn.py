import logging
import numpy as np
import torch
from torch import tensor
from torch.nn.functional import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)

# hyperparameters
PROJ_EPS = 1e-3
EPS = 1e-15
MAX_TANH_ARG = 15.0



def torch_dot(x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """ Dot product between two collections of vectors. 
    dim determines the dimension of the vectors, all other dimensions are considered batches. """
    return torch.sum(x * y, dim=dim, keepdim=True)


def torch_lambda_x(x: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return 2.0 / (1 - c * torch_dot(x, x, dim))


def torch_norm(x: torch.Tensor, dim: int = 1):
    """Compute the vector norm over a specific dimension and keeps that dimension. """
    return torch.linalg.norm(x, dim=dim, keepdim=True)


def torch_riemannian_gradient_c(x: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return ((1.0 - c * torch_dot(x, x, dim)) ** 2.) / 4.0


def torch_project_hyp_vecs(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

    clip_norm = (1.0 - PROJ_EPS) / torch.sqrt(c)

    norms = torch.linalg.norm(x, dim=1, keepdim=True)
    
    clipped = torch.where(
        condition = norms < clip_norm,
        input = x,
        other = normalize(x, dim=1) * clip_norm)
    
    return clipped


def np_project(x: np.array, c: torch.Tensor) -> torch.Tensor:
    max_norm = (1.0 - PROJ_EPS) / np.sqrt(c)
    old_norms = np.linalg.norm(x, dim=1)
    clip_idx = old_norms > max_norm
    x[clip_idx, :] /= (np.linalg.norm(x, dim=1, keepdims=True)
                       [clip_idx, :]) / max_norm
    return x


def exp_map_zero(inputs: np.array, c: torch.Tensor) -> torch.Tensor:
    inputs = inputs + EPS
    norm = np.linalg.norm(inputs, dim=1)

    gamma = np.tanh(
        np.minimum(
            np.maximum(
                np.sqrt(c) * norm, -MAX_TANH_ARG),
            MAX_TANH_ARG)
    ) / (np.sqrt(c) * norm)

    scaled_inputs = gamma[:, None] * inputs
    return np_project(scaled_inputs, c)


def torch_exp_map_zero(inputs: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    EPS = torch.tensor(EPS, device=inputs.device)
    
    sqrt_c = torch.sqrt(c)

    norm = torch.linalg.norm(
        torch.maximum(inputs, EPS), dim=1)  # protect divide by 0

    gamma = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)  # sh ncls

    scaled_inputs = gamma[:, None, :] * inputs

    return torch_project_hyp_vecs(scaled_inputs, c)


def atanh_new(inputs: torch.Tensor) -> torch.Tensor:
    x = torch.clip(inputs, min=-1 + EPS, max=1 - EPS)
    res = torch.log(1 + x) - torch.log(1 - x)
    return res * 0.5


def torch_log_map_zero_batch(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    diff = y
    norm_diff = torch.maximum(
        torch.linalg.norm(y, dim=1, keepdims=True), torch.full(
            y.shape, EPS, device=y.device))  # ( batch, height, width, 1 )
    return 1.0 / torch.sqrt(c) * atanh_new(torch.sqrt(c) * norm_diff) / norm_diff * diff


def torch_sqnorm(u: torch.Tensor, keepdims: bool = True, dim: int = 1) -> torch.Tensor:
    """ calculate vector norms over given axis """
    return torch.sum(u * u, dim=dim, keepdims=keepdims)


def torch_mob_add(u: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """ Adds two feature batches of shape (B, D, H, W) """
    torch_dot_u_v = 2.0 * c * torch_dot(u, v, dim)

    # return torch_dot_u_v
    torch_norm_u_sq = c * torch_dot(u, u, dim)
    torch_norm_v_sq = c * torch_dot(v, v, dim)
    denominator = 1.0 + torch_dot_u_v + torch_norm_v_sq * torch_norm_u_sq
    result = (1.0 + torch_dot_u_v + torch_norm_v_sq) / (denominator + EPS) * \
        u + (1.0 - torch_norm_u_sq) / (denominator * v + EPS)

    return torch_project_hyp_vecs(result, c)


def torch_exp_map_x(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:

    norm_v = torch_norm(v, dim)
    second_term = (torch.tanh(torch.sqrt(c) * torch_lambda_x(x, c, dim)
                   * norm_v / 2.) / ((torch.sqrt(c) * norm_v)) * v + EPS)

    return torch_mob_add(x, second_term, c, dim)


def torch_mob_add_batch(u: torch.Tensor, v: torch.Tensor, c: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """ adds two feature batches of shape [B, H, W, D]. Default dimension over which addition is done is 1. """

    # B,H,W,1 torch_dot(u, v)
    torch_dot_u_v = (2.0 * c * torch.sum(u * v, dim, keepdims=True))

    torch_norm_u_sq = c * torch.sum(u * u, dim, keepdims=True)

    torch_norm_v_sq = c * torch.sum(v * v, dim, keepdims=True)

    denominator = 1.0 + torch_dot_u_v + torch_norm_v_sq * torch_norm_u_sq

    result = (1.0 + torch_dot_u_v + torch_norm_v_sq) / (denominator + EPS) * \
        u + (1.0 - torch_norm_u_sq) / (denominator * v + EPS)

    return torch_project_hyp_vecs(result, c, dim)
