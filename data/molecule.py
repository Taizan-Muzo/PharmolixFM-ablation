"""
PharmolixFM 精简版数据模块
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import gzip
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem


class Molecule:
    """分子数据类"""
    
    def __init__(self) -> None:
        self.name = None
        self.smiles = None
        self.rdmol = None
        self.conformer = None
        self.graph = None
    
    @classmethod
    def from_smiles(cls, smiles: str):
        molecule = cls()
        molecule.smiles = smiles
        molecule._add_rdmol()
        return molecule
    
    @classmethod
    def from_sdf_file(cls, sdf_file: str):
        # 支持 gzip 压缩的 SDF 文件
        if sdf_file.endswith('.gz'):
            with gzip.open(sdf_file, 'rt') as f:
                mol = Chem.MolFromMolBlock(f.read())
                if mol is not None:
                    return cls.from_rdmol(mol)
        else:
            loader = Chem.SDMolSupplier(sdf_file)
            for mol in loader:
                if mol is not None:
                    molecule = cls.from_rdmol(mol)
                    return molecule
        raise ValueError(f"Failed to load molecule from {sdf_file}")
    
    @classmethod
    def from_rdmol(cls, rdmol: Chem.RWMol):
        molecule = cls()
        molecule.rdmol = rdmol
        molecule.smiles = Chem.MolToSmiles(rdmol)
        conformer = rdmol.GetConformer()
        if conformer is not None:
            molecule.conformer = np.array(conformer.GetPositions())
        return molecule
    
    def _add_rdmol(self):
        """从 SMILES 创建 RDKit 分子对象"""
        if self.smiles is not None and self.rdmol is None:
            self.rdmol = Chem.MolFromSmiles(self.smiles)
            if self.rdmol is None:
                raise ValueError(f"Invalid SMILES: {self.smiles}")
            self.rdmol = Chem.AddHs(self.rdmol)
    
    def get_num_atoms(self) -> int:
        """获取原子数量"""
        if self.rdmol is not None:
            return self.rdmol.GetNumAtoms()
        return 0


class Protein:
    """蛋白质数据类"""
    
    def __init__(self) -> None:
        self.name = None
        self.sequence = None
        self.pdb_file = None
        self.structure = None
    
    @classmethod
    def from_pdb_file(cls, pdb_file: str):
        protein = cls()
        protein.pdb_file = pdb_file
        protein.name = pdb_file.split("/")[-1].strip(".pdb")
        return protein
    
    @classmethod
    def from_fasta(cls, sequence: str):
        protein = cls()
        protein.sequence = sequence
        return protein


class Pocket:
    """口袋数据类"""
    
    def __init__(self) -> None:
        self.protein = None
        self.center = None
        self.size = None
        self.residues = None
        self.estimated_num_atoms = None
        self.atoms = []  # 口袋原子列表
        self.conformer = None  # 原子坐标 [N, 3]
    
    @classmethod
    def from_protein_ref_ligand(cls, protein: Protein, ligand: Molecule, pocket_size: float = 10.0):
        """从蛋白质和参考配体定义口袋
        
        Args:
            protein: 蛋白质对象
            ligand: 参考配体（用于定义口袋中心）
            pocket_size: 口袋半径（Å）
        """
        from Bio.PDB import PDBParser
        
        pocket = cls()
        pocket.protein = protein
        
        # 使用配体中心定义口袋中心
        if ligand.conformer is not None:
            pocket.center = np.mean(ligand.conformer, axis=0)
        else:
            pocket.center = np.array([0.0, 0.0, 0.0])
        
        # 解析 PDB 文件提取口袋原子
        if protein.pdb_file and os.path.exists(protein.pdb_file):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', protein.pdb_file)
            
            # 氨基酸映射
            aa_to_idx = {
                'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
                'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
                'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
                'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
            }
            
            # 原子序数映射
            element_to_z = {
                'C': 6, 'N': 7, 'O': 8, 'S': 16, 'H': 1,
                'P': 15, 'F': 9, 'Cl': 17, 'CL': 17, 'Br': 35, 'BR': 35, 'I': 53,
            }
            
            # 骨架原子
            backbone_atoms = {'N', 'CA', 'C', 'O'}
            
            pocket.atoms = []
            coordinates = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # 获取残基名
                        res_name = residue.resname
                        aa_type = aa_to_idx.get(res_name, 0)
                        
                        for atom in residue:
                            # 获取原子坐标
                            coord = atom.coord
                            
                            # 计算到口袋中心的距离
                            distance = np.linalg.norm(coord - pocket.center)
                            
                            # 只保留口袋范围内的原子
                            if distance <= pocket_size:
                                atom_name = atom.name
                                element = atom.element if atom.element else atom_name[0]
                                atomic_number = element_to_z.get(element, 6)
                                is_backbone = 1 if atom_name in backbone_atoms else 0
                                
                                pocket.atoms.append({
                                    'atomic_number': atomic_number,
                                    'aa_type': aa_type,
                                    'is_backbone': is_backbone,
                                    'name': atom_name,
                                    'residue': res_name,
                                })
                                coordinates.append(coord)
            
            if coordinates:
                pocket.conformer = np.array(coordinates)
        
        return pocket


def estimate_ligand_atom_num(pocket: Pocket) -> int:
    """估计口袋中的配体原子数"""
    if pocket.estimated_num_atoms is not None:
        return pocket.estimated_num_atoms
    # 默认估计值
    return 30
