import torch
import torch.nn as nn
import torch.nn.functional as F

class PharMolixBFNLoss(nn.Module):
    """
    基于 PharMolixFM 论文 Eq. 12-15 实现的 BFN 损失
    [cite: 184, 197-202]
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # beta_base 用于控制离散变量在连续空间中的信噪比
        self.beta_base = getattr(config, 'beta_base', 2.0) 
        self.sigma_data_pos = getattr(config, 'sigma_data_pos', 10.0)
        self.sigma1 = getattr(config, 'sigma1', 0.001)
        self.beta1 = getattr(config, 'beta1', 2.0)

    def compute_sender_continuous(self, x, t, sigma_data=10.0):
        """兼容旧版 BFNLoss 的接口 - 连续变量 Sender Distribution"""
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
        """兼容旧版 BFNLoss 的接口 - 离散变量 Sender Distribution"""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        x_onehot = F.one_hot(x, num_classes=K).float()
        # α(t) = (1 - (1 - β1/K)^t)
        alpha_t = 1 - torch.pow(1 - self.beta1 / K, t)
        uniform = torch.ones_like(x_onehot) / K
        theta = x_onehot * (1 - alpha_t.view(-1, 1)) + uniform * alpha_t.view(-1, 1)
        return theta
    
    def continuous_var_loss(self, pred, target, t, sigma_data):
        """兼容旧版 BFNLoss 的接口 - 连续变量加权 MSE"""
        y_sender, rho_sender = self.compute_sender_continuous(target, t, sigma_data)
        diff = pred - y_sender
        loss_per_sample = rho_sender.view(-1, 1) * (diff ** 2).sum(dim=-1)
        return loss_per_sample.mean()
    
    def discrete_var_loss(self, pred_logits, target, t, K):
        """兼容旧版 BFNLoss 的接口 - 离散变量 KL 散度"""
        theta_sender = self.compute_sender_discrete(target, t, K)
        theta_receiver = F.softmax(pred_logits, dim=-1)
        epsilon = 1e-10
        kl_per_sample = (theta_sender * (
            torch.log(theta_sender + epsilon) - torch.log(theta_receiver + epsilon)
        )).sum(dim=-1)
        return kl_per_sample.mean()

    def compute_loss(self, outputs, targets, t):
        """
        计算总 BFN 损失
        outputs: 模型预测的 clean data (去噪后的结果)
        targets: 真实数据 (ground truth)
        """
        # 1. 连续变量损失 (Atom Coordinates) - Eq. 14 [cite: 197]
        # 论文中使用 L2 范数 (MSE)
        pos_loss = F.mse_loss(outputs['pos'], targets['pos'], reduction='mean')
        
        # 2. 离散变量损失 (Atom Types) - Eq. 15 [cite: 200]
        # BFN 的 Receiver Distribution 是 Categorical(Softmax(output))
        # 最大化似然等价于最小化 Cross Entropy
        node_loss = F.cross_entropy(
            outputs['node_type'], 
            targets['node_type'], 
            reduction='mean'
        )
        
        # 3. 离散变量损失 (Bond Types)
        edge_loss = F.cross_entropy(
            outputs['halfedge_type'], 
            targets['halfedge_type'], 
            reduction='mean'
        )
        
        total_loss = pos_loss + node_loss + edge_loss
        
        return {
            'loss': total_loss,
            'pos_loss': pos_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
        }

# 别名，用于兼容旧代码导入
BFNLoss = PharMolixBFNLoss

# --- BFN 前向加噪过程 (Sender Distribution) ---

def get_sigma_schedule(t):
    """简单线性噪声调度，对应论文描述"""
    return t 

def make_broadcastable(tensor, target_tensor):
    """
    【重要】自动扩展维度，解决 3D batch vs 2D graph 的报错问题
    """
    while tensor.dim() < target_tensor.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor

def bfn_noise_continuous(x_0, t, sigma_data=None):
    """
    连续变量加噪 (Coordinates) - 对应 Eq. 12 第一行 
    p_F(X_tilde | X_0) = N(gamma * X_0, gamma * (1 - gamma) * I)
    """
    sigma = get_sigma_schedule(t)
    # 自动适配维度
    sigma = make_broadcastable(sigma, x_0)
    
    # Gamma 计算: γ = 1 - σ^2
    gamma = 1 - torch.pow(sigma, 2)
    gamma = torch.clamp(gamma, min=1e-6, max=1.0)
    
    # 均值 = γ * x_0
    mean = gamma * x_0
    # 标准差 = sqrt(γ * (1 - γ))
    std = torch.sqrt(gamma * (1 - gamma))
    
    noise = torch.randn_like(x_0)
    return mean + std * noise

def bfn_noise_discrete(k_indices, t, num_classes, beta_base=2.0):
    """
    离散变量加噪 (Atom/Bond Types) - 对应 Eq. 12 第二/三行 
    p_F(A_tilde | A_0) = N(beta * (K * A_0 - 1), beta * K * I)
    
    注意：这里将离散的 index 转换为了连续的 Embedding 输入给模型
    """
    # 1. 转 One-hot
    x_onehot = F.one_hot(k_indices, num_classes=num_classes).float()
    
    # 2. 计算 Beta: β = β_base * (1 - σ)^2
    # 参考公式 (183) 的推导逻辑 (Beta 随时间衰减)
    sigma = get_sigma_schedule(t)
    sigma = make_broadcastable(sigma, x_onehot)
    
    beta = beta_base * torch.pow(1 - sigma, 2)
    
    # 3. 计算均值: μ = β * (K * x_onehot - 1)
    K = num_classes
    mean = beta * (K * x_onehot - 1)
    
    # 4. 计算标准差: σ = sqrt(β * K)
    std = torch.sqrt(beta * K)
    
    noise = torch.randn_like(mean)
    # 返回的是连续的张量 (Continuous Tensor)，而非离散索引
    return mean + std * noise

# 兼容旧版导入
add_noise_to_continuous = bfn_noise_continuous
add_noise_to_discrete = bfn_noise_discrete
