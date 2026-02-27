"""
PharmolixFM 精简版数据模块
"""

from typing import Any, Dict, List, Optional, Tuple
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
    
    @classmethod
    def from_protein_ref_ligand(cls, protein: Protein, ligand: Molecule):
        """从蛋白质和参考配体定义口袋"""
        pocket = cls()
        pocket.protein = protein
        
        # 使用配体中心定义口袋中心
        if ligand.conformer is not None:
            pocket.center = np.mean(ligand.conformer, axis=0)
        
        return pocket


def estimate_ligand_atom_num(pocket: Pocket) -> int:
    """估计口袋中的配体原子数"""
    if pocket.estimated_num_atoms is not None:
        return pocket.estimated_num_atoms
    # 默认估计值
    return 30
