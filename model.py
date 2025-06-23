import torch
from torch.nn import Module
from submodules.DCU import depthCompletionNew_blockN
from submodules.data_rectification import rectify_depth

class DenseLiDAR(Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

    def forward(self, image, sparse, pseudo_depth_map, device):
        # mask: where sparse points exist
        mask = (sparse > 0).float()

        # DCU 예측
        rectified_depth = self.rectification(sparse, pseudo_depth_map)
        residual, concat2 = self.DCU(image, pseudo_depth_map, rectified_depth)  # normal2: [B, 1, H, W]

        return residual