import torch
from .hyperbolic_nn import torch_sqnorm, PROJ_EPS, EPS
from torch.nn.functional import conv2d, normalize
from math import sqrt


def torch_cross_correlate(inputs: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    return conv2d(inputs, weight=filters, stride=(1, 1), padding='same')


def torch_euc_mlr(inputs: torch.Tensor, P_mlr: torch.Tensor, A_mlr: torch.Tensor):
    """ Perform euclidean MLR to calculate the class logits
    
    args:
        inputs: shape (b, ch, h, w) embeddding model outputs.
        P_mlr: shape (ncls, ch) class hyperplane offsets
        A_mlr: shape (ncls, ch) class hyperplane normals
        
    returns:
        logits of shape (batch, num_classes, height_width)"""

    A_kernel = A_mlr[:, :, None, None]
    xdota = torch_cross_correlate(inputs, filters=A_kernel)
    pdota = torch.sum(-P_mlr * A_mlr, dim=1)[None, :, None, None]
    
    return pdota + xdota


def torch_hyp_mlr(inputs: torch.Tensor, c: torch.Tensor, P_mlr: torch.Tensor, A_mlr: torch.Tensor, EPS=EPS) -> torch.Tensor:
    """ Perform hyperbolic MLR to calculate the class logits
    
    args:
        inputs: shape (b, ch, h, w) embeddding model outputs.
        P_mlr: shape (ncls, ch) class hyperplane offsets
        A_mlr: shape (ncls, ch) class hyperplane normals
        
    returns:
        logits of shape (batch, num_classes, height_width)"""
        
    EPS = torch.tensor(EPS, device=inputs.device)
    
    xx = torch.linalg.norm(inputs, dim=1, keepdim=True)**2
    
    pp = torch.linalg.norm(-P_mlr, dim=-1)**2
    
    P_kernel = -P_mlr[:, :, None, None]
    
    px = torch_cross_correlate(inputs, filters=P_kernel) 
    
    sqsq = torch.multiply(
        c * xx, c * pp[None, :, None, None]) 
    
    
    A_norm = torch.linalg.norm(A_mlr, dim=1)  
    
    normed_A = normalize(A_mlr, dim=1)
    
    A = 1 + torch.add(2 * c * px, c * xx) 
    
    B = 1. - c * pp  
    
    D = 1 + (2 * c * px +  sqsq)
    
    D = torch.maximum(D, EPS)
    
    alpha = A / D  
    
    beta = B[None, :, None, None] / D  
    
    mobaddnorm = ((alpha ** 2 * pp[None, :, None, None]) +
                  (beta ** 2 * xx) + (2 * alpha * beta * px))
    
    
    maxnorm = (1.0 - PROJ_EPS) / sqrt(c)
  
    project_normalized = torch.where(
        torch.sqrt(mobaddnorm) > maxnorm,
        maxnorm / torch.maximum(torch.sqrt(mobaddnorm), EPS),
        1.,
    )
    
    mobaddnormprojected = torch.where(
        torch.sqrt(mobaddnorm) < maxnorm,
        input=mobaddnorm,
        other=maxnorm**2
    )
    
    xdota = beta * torch_cross_correlate(
        inputs, filters=normed_A[:, :, None, None])
    
    pdota = alpha * (-P_mlr * normed_A).sum(dim=1)[None, :, None, None]
    
    mobdota = xdota + pdota  
    
    mobdota_out = mobdota * project_normalized  
    
    lamb_px = 2.0 / torch.maximum(1. - c * mobaddnormprojected, EPS)
    
    sineterm = sqrt(c) * mobdota_out * lamb_px
    
    return 2.0 / sqrt(c) * A_norm.view(1, -1, 1, 1) * torch.asinh(sineterm)

