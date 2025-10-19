"""
随机种子设置工具
确保实验的可复现性
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    设置所有随机种子以确保可复现性
    
    参数:
        seed: 随机种子值
    """
    # Python内置random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 随机种子已设置: {seed}")


def worker_init_fn(worker_id):
    """
    DataLoader worker初始化函数
    确保每个worker有不同但可复现的随机种子
    
    参数:
        worker_id: worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

