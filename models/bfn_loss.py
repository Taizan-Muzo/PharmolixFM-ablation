"""
BFN (Bayesian Flow Networks) 损失函数 - 完整实现
基于论文: "Bayesian Flow Networks" (Graves et al., 2023)
https://arxiv.org/abs/2308.07037
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BFNLoss:
    """完整的 BFN 损失函数实现"""
    
    def __init__(self, config):
        self.config = config
        self.sigma_data_pos = getattr(config, 'sigma_data_pos', 10.0)
        self.sigma_data_node = getattr(config, 'sigma_data_node', 1.0)
        self.sigma_data_halfedge = getattr(config, 'sigma_data_halfedge', 1.0)
        self.sigma1 = getattr(config, 'sigma1', 0.001)
        self.beta1 = getattr(config, 'beta1', 2.0)
    
    def compute_sender_continuous(self, x, t, sigma_data):
        """连续变量 Sender Distribution: 输出 (mean, precision)"""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        
        # γ(t) = (1 - σ1^(2t)) / (1 - σ1^2)
        gamma_t = (1 - torch.pow(self.sigma1, 2 * t)) / (1 - self.sigma1 ** 2)
        
        # Sender precision: ρ(t)
        rho = 1.0 / (gamma_t * sigma_data ** 2 + 1e-8)
        
        # Sender mean: y = γ(t) * x
        y = gamma_t.view(-1, 1) * x
        
        return y, rho
    
    def compute_sender_discrete(self, x, t, K):
        """离散变量 Sender Distribution: 输出概率分布 θ"""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        
        x_onehot = F.one_hot(x, num_classes=K).float()
        
        # α(t) = (1 - (1 - β1/K)^t)
        alpha_t = 1 - torch.pow(1 - self.beta1 / K, t)
        
        uniform = torch.ones_like(x_onehot) / K
        theta = x_onehot * (1 - alpha_t.view(-1, 1)) + uniform * alpha_t.view(-1, 1)
        
        return theta
    
    def continuous_var_loss(self, pred, target, t, sigma_data):
        """连续变量 BFN 损失: weighted MSE"""
        y_sender, rho_sender = self.compute_sender_continuous(target, t, sigma_data)
        y_receiver = pred
        
        diff = y_receiver - y_sender
        loss_per_sample = rho_sender.view(-1, 1) * (diff ** 2).sum(dim=-1)
        
        return loss_per_sample.mean()
    
    def discrete_var_loss(self, pred_logits, target, t, K):
        """离散变量 BFN 损失: KL(sender || receiver)"""
        theta_sender = self.compute_sender_discrete(target, t, K)
        theta_receiver = F.softmax(pred_logits, dim=-1)
        
        epsilon = 1e-10
        kl_per_sample = (theta_sender * (
            torch.log(theta_sender + epsilon) - torch.log(theta_receiver + epsilon)
        )).sum(dim=-1)
        
        return kl_per_sample.mean()
    
    def compute_loss(self, outputs, targets, t):
        """计算总 BFN 损失"""
        pos_loss = self.continuous_var_loss(
            outputs['pos'], targets['pos'], t, self.sigma_data_pos)
        
        node_loss = self.discrete_var_loss(
            outputs['node_type'], targets['node_type'], t,
            outputs['node_type'].shape[-1])
        
        edge_loss = self.discrete_var_loss(
            outputs['halfedge_type'], targets['halfedge_type'], t,
            outputs['halfedge_type'].shape[-1])
        
        total_loss = pos_loss + node_loss + edge_loss
        
        return {
            'loss': total_loss,
            'pos_loss': pos_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
        }


def add_noise_to_continuous(x, t, sigma_data):
    """给连续变量添加噪声 (BFN 前向过程)"""
    noise = torch.randn_like(x) * sigma_data
    return x * (1 - t.view(-1, 1)) + noise * t.view(-1, 1)


def add_noise_to_discrete(x, t, K):
    """给离散变量添加噪声 (BFN 前向过程)"""
    N = x.shape[0]
    x_onehot = F.one_hot(x, num_classes=K).float()
    uniform = torch.ones(N, K, device=x.device) / K
    alpha = t.view(-1, 1)
    return x_onehot * (1 - alpha) + uniform * alpha
