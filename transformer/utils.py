

import copy
import torch.nn as nn

def get_clones(module, N):
    # deepcopy 保证每一层都是独立的，不共享参数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])









