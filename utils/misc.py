"""
PharmolixFM 工具函数
"""

from typing import Any, List
import torch
from torch_geometric.data import Batch


def safe_index(lst: List[Any], item: Any) -> int:
    """安全索引，如果不在列表中返回最后一个索引"""
    try:
        return lst.index(item)
    except ValueError:
        return len(lst) - 1


class PygCollator:
    """PyG 数据批次整理器"""
    
    def __call__(self, batch):
        return Batch.from_data_list(batch)
