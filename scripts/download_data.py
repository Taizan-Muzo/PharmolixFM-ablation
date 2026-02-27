"""
数据下载脚本
"""

import os
import urllib.request
from pathlib import Path


def download_file(url: str, output_path: str):
    """下载文件"""
    print(f"Downloading {url}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading PharmolixFM test data...")
    
    # 示例 PDB 文件（可从 RCSB PDB 下载）
    # download_file(
    #     "https://files.rcsb.org/download/4XLI.pdb",
    #     str(data_dir / "4xli.pdb")
    # )
    
    print("Data download completed!")
    print(f"Data directory: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
