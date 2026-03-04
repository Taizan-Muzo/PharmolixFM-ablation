"""
CrossDocked 数据集加载器
兼容官方 OpenBioMed 格式
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

from data.molecule import Molecule, Protein, Pocket


class CrossDockedDataset(Dataset):
    """
    CrossDocked 数据集
    用于基于结构的药物设计任务
    """
    
    def __init__(self, data_dir: str, split: str = "train", 
                 featurizer_config: dict = None, debug: bool = False):
        """
        Args:
            data_dir: 数据目录 (包含 crossdocked_pocket10_with_protein/)
            split: train/val/test
            featurizer_config: 特征化器配置
            debug: 是否使用调试模式（少量数据）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.debug = debug
        
        # 加载数据
        self.data_list = self._load_data()
        
        if debug:
            self.data_list = self.data_list[:100]
        
        print(f"Loaded {len(self.data_list)} samples for {split} split")
        
        # 创建特征化器
        from utils.featurizer import PharmolixFMMoleculeFeaturizer
        from utils.pocket_featurizer import PharmolixFMPocketFeaturizer
        self.mol_featurizer = PharmolixFMMoleculeFeaturizer(
            pos_norm=featurizer_config.get("pos_norm", 1.0) if featurizer_config else 1.0
        )
        self.pocket_featurizer = PharmolixFMPocketFeaturizer(
            knn=featurizer_config.get("knn", 32) if featurizer_config else 32
        )
    
    def _load_data(self):
        """加载数据索引"""
        data_list = []
        
        # 尝试加载 split 文件
        split_file = self.data_dir / "split_by_name.pt"
        if split_file.exists():
            split_index = torch.load(split_file)
            
            # 映射 split 名称
            split_map = {"train": "train", "val": "train", "test": "test"}
            target_split = split_map.get(self.split, self.split)
            
            # 如果是 val，取 train 的前 100 个作为验证集
            if self.split == "val":
                all_train = split_index.get("train", [])
                split_data = all_train[:100] if len(all_train) > 100 else all_train
            else:
                split_data = split_index.get(target_split, [])
            
            for protein_file, ligand_file in split_data:
                data_list.append({
                    "protein": self.data_dir / protein_file,
                    "ligand": self.data_dir / ligand_file,
                })
        else:
            # 如果没有 split 文件，扫描目录
            if self.split == "test":
                base_dir = self.data_dir / "test_set"
            else:
                base_dir = self.data_dir
            
            if base_dir.exists():
                for subdir in base_dir.iterdir():
                    if subdir.is_dir():
                        # 查找配体文件 (*_uff.sdf, *_lig.sdf, 或 *.sdf.gz)
                        ligand_files = list(subdir.glob("*_uff.sdf")) + list(subdir.glob("*_lig.sdf")) + list(subdir.glob("*.sdf.gz"))
                        # 查找蛋白质文件 (*_rec.pdb)
                        protein_files = list(subdir.glob("*_rec.pdb"))
                        
                        if ligand_files and protein_files:
                            data_list.append({
                                "protein": protein_files[0],
                                "ligand": ligand_files[0],
                            })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    # models/dataset.py (或者你定义 CrossDockedDataset 的地方)

    def __getitem__(self, idx):
        # 尝试加载当前索引的数据
        success = False
        max_retries = 20
        current_idx = idx
        
        for _ in range(max_retries):
            item = self.data_list[current_idx]
            try:
                # 1. 尝试解析分子
                molecule = Molecule.from_sdf_file(str(item["ligand"]))
                if molecule is None or molecule.get_num_atoms() == 0:
                    raise ValueError("Empty or invalid molecule")
                    
                # 2. 尝试解析口袋/蛋白质
                protein = Protein.from_pdb_file(str(item["protein"]))
                pocket = Pocket.from_protein_ref_ligand(protein, molecule, pocket_size=10.0)
                
                # 3. 特征化
                mol_features = self.mol_featurizer(molecule)
                pocket_features = self.pocket_featurizer(pocket)
                
                metadata = {
                    "name": item["ligand"].stem,
                    "num_atoms": molecule.get_num_atoms(),
                }
                
                return mol_features, pocket_features, metadata
                
            except Exception as e:
                # 如果失败，随机换一个索引继续尝试
                import random
                current_idx = random.randint(0, len(self.data_list) - 1)
                continue 
                
        # 如果实在倒霉，连试 20 次都失败（基本不可能），最后才返回一个默认空值
        return self._get_empty_sample()
    
    def _get_empty_sample(self):
        """返回空样本"""
        import torch
        from torch_geometric.data import Data
        empty_mol = Data(
            node_type=torch.zeros(1, 12),
            pos=torch.zeros(1, 3),
            halfedge_index=torch.zeros(2, 0, dtype=torch.long),
            halfedge_type=torch.zeros(0, 6),
        )
        empty_pocket = Data(
            atom_feature=torch.zeros(1, 25),
            pos=torch.zeros(1, 3),
            knn_edge_index=torch.zeros(2, 0, dtype=torch.long),
            pocket_center=torch.zeros(1, 3),
            estimated_ligand_num_atoms=torch.tensor([30]),
        )
        return empty_mol, empty_pocket, {"name": "empty", "num_atoms": 0}


class PDBBindDataset(Dataset):
    """
    PDBBind 数据集
    """
    
    def __init__(self, data_dir: str, split: str = "train", debug: bool = False):
        """
        Args:
            data_dir: 数据目录
            split: train/val/test
            debug: 是否使用调试模式
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.debug = debug
        
        # 加载数据
        self.data_list = self._load_data()
        
        if debug:
            self.data_list = self.data_list[:100]
        
        print(f"Loaded {len(self.data_list)} samples for {split} split")
    
    def _load_data(self):
        """加载数据索引"""
        data_list = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            return data_list
        
        for complex_dir in split_dir.iterdir():
            if complex_dir.is_dir():
                # 查找配体和蛋白质文件
                ligand_files = list(complex_dir.glob("*.sdf")) + list(complex_dir.glob("*.mol2"))
                protein_files = list(complex_dir.glob("*.pdb"))
                
                if ligand_files and protein_files:
                    data_list.append({
                        "name": complex_dir.name,
                        "protein": protein_files[0],
                        "ligand": ligand_files[0],
                    })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        item = self.data_list[idx]
        
        try:
            # 加载配体
            molecule = Molecule.from_sdf_file(str(item["ligand"]))
            
            # 加载蛋白质并创建口袋
            protein = Protein.from_pdb_file(str(item["protein"]))
            pocket = Pocket.from_protein_ref_ligand(protein, molecule)
            
            metadata = {
                "name": item["name"],
                "num_atoms": molecule.get_num_atoms(),
            }
            
            return molecule, pocket, metadata
            
        except Exception as e:
            print(f"Error loading {item.get('name', 'unknown')}: {e}")
            return self._get_empty_sample()
    
    def _get_empty_sample(self):
        """返回空样本"""
        empty_mol = Molecule()
        empty_pocket = Pocket()
        return empty_mol, empty_pocket, {"name": "empty", "num_atoms": 0}
