import torch
import numpy as np
from layers_tf import tf_hyp_mlr
import tensorflow as tf
from hesp.util.layers import *
from hesp.util.hyperbolic_nn import *
from hyperbolic_nn_tf import tf_exp_map_zero, tf_project_hyp_vecs

torch.set_default_dtype = torch.float64

sess = tf.Session()

c = 1.0
c_torch = torch.tensor(c)
c_tf = tf.convert_to_tensor(c, dtype=tf.float64)

count = 0
for i in range(20):
    inputs = np.random.rand(5, 100, 100, 100) * 2.
    # P_mlr = np.random.rand(10, 10) *  0.3
    # A_mlr = np.random.rand(10, 10) *  0.3

    inputs_tf = tf.convert_to_tensor(np.moveaxis(inputs, 1, -1), dtype=tf.float64)
    # P_mlr_tf = tf.convert_to_tensor(P_mlr, dtype=tf.float64)
    # A_mlr_tf = tf.convert_to_tensor(A_mlr, dtype=tf.float64)

    inputs_torch = torch.tensor(inputs)
    # P_mlr_torch = torch.tensor(P_mlr)
    # A_mlr_torch = torch.tensor(A_mlr)
    
    outputs_torch = torch_exp_map_zero(inputs_torch, c_torch)
    
    with sess.as_default():
        outputs_tf = tf_exp_map_zero(inputs_tf, c_tf,).eval()
    
    output_torch = outputs_torch.numpy()
    
    output_torch = np.moveaxis(output_torch, 1, -1)
    
    if np.allclose(output_torch, outputs_tf, atol=1e-8):
        count += 1
        
print(count / 100)
    
    
    