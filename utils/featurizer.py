"""
PharmolixFM 特征化器
"""

from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from data.molecule import Molecule, Pocket
from utils.misc import safe_index


class MoleculeFeaturizer:
    """分子特征化器基类"""
    
    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        raise NotImplementedError


class PocketFeaturizer:
    """口袋特征化器基类"""
    
    def __call__(self, pocket: Pocket) -> Dict[str, Any]:
        raise NotImplementedError


class PharmolixFMMoleculeFeaturizer(MoleculeFeaturizer):
    """PharmolixFM 分子特征化器"""
    
    def __init__(self, pos_norm=1.0, num_node_types=12, num_edge_types=6) -> None:
        super().__init__()
        self.atomic_numbers = [6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34]
        from rdkit import Chem
        self.mol_bond_types = [
            'empty',
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        self.pos_norm = pos_norm
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        molecule._add_rdmol()
        rdmol = molecule.rdmol
        
        node_type_list = []
        for atom in rdmol.GetAtoms():
            node_type_list.append(safe_index(self.atomic_numbers, atom.GetAtomicNum()))
        node_type = F.one_hot(torch.LongTensor(node_type_list), num_classes=self.num_node_types).float()
        num_nodes = node_type.shape[0]

        if molecule.conformer is not None:
            pos = torch.tensor(molecule.conformer).float()
        else:
            pos = torch.zeros(num_nodes, 3)
        
        # Move to center
        pos -= pos.mean(0)
        pos /= self.pos_norm

        # Build halfedge
        if len(rdmol.GetBonds()) <= 0:
            halfedge_index = torch.empty((2, 0), dtype=torch.long)
            halfedge_type = torch.empty(0, dtype=torch.long)
        else:
            halfedge_matrix = torch.zeros([num_nodes, num_nodes], dtype=torch.long)
            for bond in rdmol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = safe_index(self.mol_bond_types, bond.GetBondType())
                halfedge_matrix[i, j] = bond_type
                halfedge_matrix[j, i] = bond_type
            halfedge_index = torch.triu_indices(num_nodes, num_nodes, offset=1)
            halfedge_type = F.one_hot(halfedge_matrix[halfedge_index[0], halfedge_index[1]], num_classes=self.num_edge_types).float()
        
        return {
            'node_type': node_type,
            'pos': pos,
            'halfedge_index': halfedge_index,
            'halfedge_type': halfedge_type,
            'num_nodes': num_nodes,
        }
