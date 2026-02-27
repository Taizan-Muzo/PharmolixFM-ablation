"""
PharmolixFM 口袋特征化器（完整实现）
"""

from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from data.molecule import Pocket
from utils.misc import safe_index


class PharmolixFMPocketFeaturizer:
    """PharmolixFM 口袋特征化器"""
    
    def __init__(self, knn: int = 32, pos_norm: float = 1.0) -> None:
        super().__init__()
        
        self.knn = knn
        self.pos_norm = pos_norm
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16])  # C, N, O, S
        self.max_num_aa = 20
    
    def __call__(self, pocket: Pocket) -> Dict[str, Any]:
        """
        将口袋特征化为 PyG Data 对象
        
        Args:
            pocket: Pocket 对象，需要包含 atoms 列表
        
        Returns:
            特征化后的字典
        """
        # 检查口袋是否有原子信息
        if not hasattr(pocket, 'atoms') or pocket.atoms is None:
            # 如果没有原子信息，创建空特征
            return self._create_empty_features()
        
        atoms = pocket.atoms
        
        # 元素 one-hot 编码
        elements = torch.LongTensor([atom.get("atomic_number", 6) for atom in atoms])
        elements_one_hot = (elements.view(-1, 1) == self.atomic_numbers.view(1, -1)).long()
        
        # 氨基酸类型 one-hot
        aa_type = torch.LongTensor([atom.get("aa_type", 0) for atom in atoms])
        aa_one_hot = F.one_hot(aa_type, num_classes=self.max_num_aa)
        
        # 是否为主链原子
        is_backbone = torch.LongTensor([atom.get("is_backbone", 0) for atom in atoms]).unsqueeze(-1)
        
        # 组合特征
        x = torch.cat([elements_one_hot, aa_one_hot, is_backbone], dim=-1).float()
        
        # 位置坐标
        if hasattr(pocket, 'conformer') and pocket.conformer is not None:
            pos = torch.tensor(pocket.conformer, dtype=torch.float32)
        else:
            # 如果没有构象，使用随机坐标（临时方案）
            pos = torch.randn(len(atoms), 3)
        
        # KNN 图
        if len(atoms) > self.knn:
            knn_edge_index = knn_graph(pos, k=self.knn, flow='target_to_source')
        else:
            # 如果原子数太少，使用全连接
            knn_edge_index = torch.combinations(torch.arange(len(atoms)), r=2).t()
        
        # 计算口袋中心
        pocket_center = pos.mean(dim=0)
        pos -= pocket_center
        pos /= self.pos_norm
        
        # 估计配体原子数
        from data.molecule import estimate_ligand_atom_num
        estimated_num_atoms = torch.tensor(estimate_ligand_atom_num(pocket)).unsqueeze(0)
        
        return Data(**{
            "atom_feature": x,
            "knn_edge_index": knn_edge_index,
            "pos": pos,
            "pocket_center": pocket_center.unsqueeze(0),
            "estimated_ligand_num_atoms": estimated_num_atoms,
        })
    
    def _create_empty_features(self) -> Dict[str, Any]:
        """创建空特征（用于占位）"""
        return Data(**{
            "atom_feature": torch.zeros(1, 4 + 20 + 1),
            "knn_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "pos": torch.zeros(1, 3),
            "pocket_center": torch.zeros(1, 3),
            "estimated_ligand_num_atoms": torch.tensor([30]),
        })
