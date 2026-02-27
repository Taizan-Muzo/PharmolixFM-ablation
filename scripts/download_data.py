"""
数据下载脚本（实现版）
下载 PDBbind 和 CrossDocked 数据集
"""

import argparse
import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """下载文件并显示进度"""
    print(f"Downloading {url}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    print(f"Saved to {output_path}")
    return output_path


def extract_archive(archive_path: str, output_dir: str):
    """解压压缩包"""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path}...")
    
    if archive_path.suffix == '.gz' or '.tar' in archive_path.suffixes:
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(output_dir)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    print(f"Extracted to {output_dir}")


def download_pdbbind(data_dir: str, version: str = "2020"):
    """
    下载 PDBbind 数据集
    注意：PDBbind 需要注册后下载，这里提供链接
    """
    print(f"PDBbind {version} 需要手动下载:")
    print("1. 访问 http://www.pdbbind.org.cn/")
    print("2. 注册账号并下载")
    print(f"3. 解压到 {data_dir}/pdbbind/{version}/")
    
    # 创建目录结构
    pdbbind_dir = Path(data_dir) / "pdbbind" / version
    pdbbind_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {pdbbind_dir}")


def download_crossdocked(data_dir: str):
    """
    下载 CrossDocked 数据集
    """
    print("CrossDocked 数据集:")
    print("1. 访问 https://github.com/gnina/models/tree/master/data/CrossDocked2020")
    print("2. 下载 crossdocked_pocket10.tar.gz")
    print(f"3. 解压到 {data_dir}/crossdocked/")
    
    crossdocked_dir = Path(data_dir) / "crossdocked"
    crossdocked_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {crossdocked_dir}")


def download_test_examples(data_dir: str):
    """下载测试示例（从 RCSB PDB）"""
    test_dir = Path(data_dir) / "test_examples"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 示例 PDB ID
    test_pdbs = ["4XLI", "3POZ", "1AZM"]
    
    print("Downloading test examples from RCSB PDB...")
    for pdb_id in test_pdbs:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_path = test_dir / f"{pdb_id}.pdb"
        try:
            download_file(url, str(output_path))
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download PharmolixFM datasets")
    parser.add_argument("--data_dir", type=str, default="data/", help="数据目录")
    parser.add_argument("--dataset", type=str, choices=["pdbbind", "crossdocked", "test", "all"], 
                        default="test", help="要下载的数据集")
    parser.add_argument("--pdbbind_version", type=str, default="2020", help="PDBbind 版本")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir.absolute()}")
    print()
    
    if args.dataset in ["pdbbind", "all"]:
        download_pdbbind(str(data_dir), args.pdbbind_version)
        print()
    
    if args.dataset in ["crossdocked", "all"]:
        download_crossdocked(str(data_dir))
        print()
    
    if args.dataset in ["test", "all"]:
        download_test_examples(str(data_dir))
        print()
    
    print("=" * 60)
    print("Download instructions completed!")
    print()
    print("Note: Most datasets require manual download due to licensing.")
    print("Please visit the respective websites and follow their terms.")
    print()
    print("Next steps:")
    print("1. Download datasets manually")
    print("2. Extract to the appropriate directories")
    print("3. Run preprocessing scripts (if needed)")
    print("4. Start training with: python scripts/train.py")


if __name__ == "__main__":
    main()
