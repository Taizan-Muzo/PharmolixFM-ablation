"""
PharmolixFM 配置类
"""

from typing import Dict, Any


class Config:
    """简化版配置类"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        # 模型配置
        self.num_node_types = 12
        self.num_edge_types = 6
        self.node_dim = 128
        self.edge_dim = 64
        self.hidden_dim = 256
        self.num_layers = 6
        self.num_heads = 8
        self.dropout = 0.1
        
        # BFN 配置
        self.sigma_data_pos = 10.0
        self.sigma_data_node = 1.0
        self.sigma_data_halfedge = 1.0
        
        # BFN 噪声调度参数
        self.sigma1 = 0.001  # 最小噪声
        self.beta1 = 2.0     # 离散变量参数
        
        # 采样配置
        self.num_sample_steps = 100
        self.pos_noise_scale = 0.1
        
        # 模型维度配置
        self.pocket_dim = 128
        self.pocket_in_dim = 25  # 4 (elements) + 20 (aa_types) + 1 (is_backbone)
        self.pocket_knn = 32
        
        # 额外特征
        self.addition_node_features = []
        self.add_output = []
        
        # 编码器配置（简化为字典）
        self.pocket = {
            'num_layers': 3,
            'num_heads': 4,
            'hidden_dim': 128,
            'gate_dim': 64,
        }
        
        self.denoiser = {
            'num_layers': 6,
            'num_heads': 8,
            'hidden_dim': 256,
            'gate_dim': 128,
        }
        
        # 训练配置
        self.learning_rate = 1e-4
        self.batch_size = 8
        
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    def __getattr__(self, name):
        """默认返回 None 避免 AttributeError"""
        return None
    
    def todict(self):
        """转换为字典（用于模型配置）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
