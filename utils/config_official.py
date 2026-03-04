"""
PharmolixFM 官方模型配置 (从 OpenBioMed 获取)
"""

from typing import Dict, Any


class OfficialConfig:
    """官方模型配置 - 匹配 HuggingFace 上的预训练权重"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        # 模型配置 (来自 OpenBioMed/configs/model/pharmolix_fm.yaml)
        self.num_node_types = 12
        self.num_edge_types = 6
        self.node_dim = 320  # 官方: 320
        self.edge_dim = 96   # 官方: 96
        self.hidden_dim = 320  # 官方: 320
        self.num_layers = 6
        self.num_heads = 8
        self.dropout = 0.1
        
        # BFN 配置
        self.sigma_data_pos = 10.0
        self.sigma_data_node = 1.0
        self.sigma_data_halfedge = 1.0
        
        # BFN 噪声调度参数 (官方)
        self.sigma1 = 0.03  # 官方: 0.03
        self.beta1 = 1.5    # 官方: 1.5
        
        # 采样配置 (官方)
        self.num_sample_steps = 100
        self.pos_noise_scale = 0.1
        self.pos_norm = 6.0  # 官方: 6
        
        # 模型维度配置
        self.pocket_dim = 128
        self.pocket_in_dim = 25  # 4 (elements) + 20 (aa_types) + 1 (is_backbone)
        self.pocket_knn = 32
        
        # 额外特征 (官方)
        self.addition_node_features = ['is_peptide']
        self.add_output = ['confidence']  # 官方有置信度输出
        
        # 距离扩展配置 (官方)
        self.dist_cfg = {
            'start': 0.0,
            'stop': 15.0,  # 官方: 15
            'num_gaussians': 64,  # 官方: 64
        }
        
        # 编码器配置（官方）
        self.pocket = {
            'node_dim': 128,
            'edge_dim': 32,
            'hidden_dim': 128,
            'num_blocks': 4,  # 官方: 4
            'dist_cfg': {
                'start': 0.0,
                'stop': 15.0,
                'num_gaussians': 32,  # 官方: 32
            },
            'gate_dim': 2,
        }
        
        self.denoiser = {
            'node_dim': 320,  # 官方: 320
            'edge_dim': 96,   # 官方: 96
            'hidden_dim': 320,  # 官方: 320
            'num_blocks': 6,
            'dist_cfg': {
                'start': 0.0,
                'stop': 15.0,
                'num_gaussians': 64,  # 官方: 64
            },
            'gate_dim': 2,
            'knn': 32,
            'context_cfg': {
                'edge_dim': 128,  # 官方: 128
                'knn': 32,
                'dist_cfg': {
                    'stop': 20.0,  # 官方: 20
                    'num_gaussians': 64,  # 官方: 64
                    'type_': 'linear',  # 官方: linear
                },
            },
        }
        
        # 训练配置
        self.learning_rate = 1e-4
        self.batch_size = 32  # 官方: 32
        
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    def __getattr__(self, name):
        """默认返回 None 避免 AttributeError"""
        return None
    
    def todict(self):
        """转换为字典（用于模型配置）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
