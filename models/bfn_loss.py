"""
BFN (Bayesian Flow Network) 损失函数实现
基于论文: "Bayesian Flow Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BFNLoss:
    """
    BFN 损失函数
    
    包含:
    - 连续变量损失（位置坐标）
    - 离散变量损失（原子类型、键类型）
    """
    
    def __init__(self, config):
        self.config = config
        self.sigma_data_pos = getattr(config, 'sigma_data_pos', 10.0)
        self.sigma_data_node = getattr(config, 'sigma_data_node', 1.0)
        self.sigma_data_halfedge = getattr(config, 'sigma_data_halfedge', 1.0)
    
    def continuous_var_loss(self, pred, target, t, sigma_data):
        """
        连续变量损失（位置坐标）
        
        Args:
            pred: 预测值 (N, 3)
            target: 目标值 (N, 3)
            t: 时间步 (N,)
            sigma_data: 数据标准差
        
        Returns:
            损失值
        """
        # BFN 连续变量损失: ||pred - target||^2 / (2 * sigma^2)
        loss = F.mse_loss(pred, target, reduction='none').sum(dim=-1)
        
        # 根据时间步加权
        weight = 1.0 / (sigma_data ** 2)
        
        return (loss * weight).mean()
    
    def discrete_var_loss(self, pred_logits, target, t, K):
        """
        离散变量损失（原子类型、键类型）
        
        Args:
            pred_logits: 预测 logits (N, K)
            target: 目标类别 (N,)
            t: 时间步 (N,)
            K: 类别数
        
        Returns:
            损失值
        """
        # 交叉熵损失
        loss = F.cross_entropy(pred_logits, target, reduction='none')
        
        return loss.mean()
    
    def compute_loss(self, outputs, targets, t):
        """
        计算总损失
        
        Args:
            outputs: 模型输出字典
                - pos: 预测位置 (N, 3)
                - node_type: 预测节点类型 logits (N, num_node_types)
                - halfedge_type: 预测边类型 logits (E, num_edge_types)
            targets: 目标字典
                - pos: 目标位置 (N, 3)
                - node_type: 目标节点类型 (N,)
                - halfedge_type: 目标边类型 (E,)
            t: 时间步
        
        Returns:
            损失字典
        """
        # 位置损失
        pos_loss = self.continuous_var_loss(
            outputs['pos'], 
            targets['pos'], 
            t, 
            self.sigma_data_pos
        )
        
        # 节点类型损失
        node_loss = self.discrete_var_loss(
            outputs['node_type'],
            targets['node_type'],
            t,
            outputs['node_type'].shape[-1]
        )
        
        # 边类型损失
        edge_loss = self.discrete_var_loss(
            outputs['halfedge_type'],
            targets['halfedge_type'],
            t,
            outputs['halfedge_type'].shape[-1]
        )
        
        # 总损失
        total_loss = pos_loss + node_loss + edge_loss
        
        return {
            'loss': total_loss,
            'pos_loss': pos_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
        }


def create_noise_schedule(config, num_steps=1000):
    """
    创建噪声调度（用于训练时采样时间步）
    
    Args:
        config: 配置对象
        num_steps: 时间步数量
    
    Returns:
        时间步张量
    """
    # 均匀采样
    t = torch.rand(num_steps)
    return t


def add_noise_to_continuous(x, t, sigma_data):
    """
    给连续变量添加噪声
    
    Args:
        x: 原始数据
        t: 时间步 (0-1)
        sigma_data: 数据标准差
    
    Returns:
        加噪后的数据
    """
    noise = torch.randn_like(x) * sigma_data
    return x * (1 - t.view(-1, 1)) + noise * t.view(-1, 1)


def add_noise_to_discrete(x, t, K):
    """
    给离散变量添加噪声（通过随机替换）
    
    Args:
        x: 原始类别 (N,)
        t: 时间步 (0-1)
        K: 类别数
    
    Returns:
        加噪后的类别分布 (N, K)
    """
    N = x.shape[0]
    
    # 原始类别的 one-hot
    x_onehot = F.one_hot(x, num_classes=K).float()
    
    # 随机噪声（均匀分布）
    noise = torch.ones(N, K) / K
    
    # 插值
    alpha = t.view(-1, 1)
    x_noisy = x_onehot * (1 - alpha) + noise * alpha
    
    return x_noisy
