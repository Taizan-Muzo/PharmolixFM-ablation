"""
PharmolixFM 数据集类
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.molecule import Molecule, Protein, Pocket
from utils.featurizer import PharmolixFMMoleculeFeaturizer
from utils.pocket_featurizer import PharmolixFMPocketFeaturizer


class PocketMoleculeDataset(Dataset):
    """
    口袋-分子对接数据集
    
    数据格式：
    - 蛋白质 PDB 文件
    - 配体 SDF 文件
    - 对接姿态（可选）
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        featurizer_config: Dict[str, Any] = None,
    ):
        """
        Args:
            data_dir: 数据目录
            split: train/val/test
            featurizer_config: 特征化器配置
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 创建特征化器
        self.mol_featurizer = PharmolixFMMoleculeFeaturizer(
            pos_norm=featurizer_config.get("pos_norm", 1.0) if featurizer_config else 1.0
        )
        self.pocket_featurizer = PharmolixFMPocketFeaturizer(
            knn=featurizer_config.get("knn", 32) if featurizer_config else 32
        )
        
        # 加载数据索引
        self.data_list = self._load_data_list()
    
    def _load_data_list(self) -> List[Dict[str, str]]:
        """加载数据索引"""
        data_list = []
        
        # 查找所有复合物
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            return data_list
        
        # 假设每个子目录是一个复合物
        for complex_dir in split_dir.iterdir():
            if complex_dir.is_dir():
                pdb_file = complex_dir / "protein.pdb"
                sdf_file = complex_dir / "ligand.sdf"
                
                if pdb_file.exists() and sdf_file.exists():
                    data_list.append({
                        "name": complex_dir.name,
                        "pdb": str(pdb_file),
                        "sdf": str(sdf_file),
                    })
        
        print(f"Loaded {len(data_list)} complexes for {self.split}")
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict]:
        """
        获取一个样本
        
        Returns:
            (molecule_features, pocket_features, metadata)
        """
        item = self.data_list[idx]
        
        try:
            # 加载配体
            molecule = Molecule.from_sdf_file(item["sdf"])
            mol_features = self.mol_featurizer(molecule)
            
            # 加载蛋白质并定义口袋
            protein = Protein.from_pdb_file(item["pdb"])
            pocket = Pocket.from_protein_ref_ligand(protein, molecule)
            
            # TODO: 从 PDB 解析口袋原子信息
            # 暂时使用空的原子列表
            pocket.atoms = []
            
            pocket_features = self.pocket_featurizer(pocket)
            
            metadata = {
                "name": item["name"],
                "num_atoms": molecule.get_num_atoms(),
            }
            
            return mol_features, pocket_features, metadata
            
        except Exception as e:
            print(f"Error loading {item['name']}: {e}")
            # 返回空样本
            return self._get_empty_sample()
    
    def _get_empty_sample(self) -> Tuple[Dict, Dict, Dict]:
        """返回空样本（用于错误处理）"""
        empty_mol = {
            "node_type": torch.zeros(1, 12),
            "pos": torch.zeros(1, 3),
            "halfedge_index": torch.zeros(2, 0, dtype=torch.long),
            "halfedge_type": torch.zeros(0, 6),
        }
        empty_pocket = {
            "atom_feature": torch.zeros(1, 25),
            "pos": torch.zeros(1, 3),
            "knn_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "pocket_center": torch.zeros(1, 3),
            "estimated_ligand_num_atoms": torch.tensor([30]),
        }
        return empty_mol, empty_pocket, {"name": "empty", "num_atoms": 0}


class DummyDataset(Dataset):
    """
    虚拟数据集（用于测试）
    """
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.mol_featurizer = PharmolixFMMoleculeFeaturizer()
        self.pocket_featurizer = PharmolixFMPocketFeaturizer()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # 创建虚拟分子
        from data.molecule import Molecule
        molecule = Molecule.from_smiles("CCO")  # 乙醇
        
        # 创建虚拟口袋
        from data.molecule import Pocket
        pocket = Pocket()
        pocket.atoms = [
            {"atomic_number": 6, "aa_type": 0, "is_backbone": 1},
            {"atomic_number": 7, "aa_type": 1, "is_backbone": 0},
        ]
        pocket.conformer = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        
        mol_features = self.mol_featurizer(molecule)
        pocket_features = self.pocket_featurizer(pocket)
        
        metadata = {
            "name": f"dummy_{idx}",
            "num_atoms": molecule.get_num_atoms(),
        }
        
        return mol_features, pocket_features, metadata
