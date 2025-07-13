import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualRefineBlock(nn.Module):
    """
    Simple Conv-based Residual Refinement Block.
    Input:  residual map (B, 1, H, W)
    Output: refined residual (B, 1, H, W)
    """
    def __init__(self, in_channels=1, mid_channels=32):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels,    mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels,   mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels,   mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 프로젝트 채널 수를 1로 줄여서 refined residual 생성
            nn.Conv2d(mid_channels,   1,            kernel_size=3, padding=1, bias=True),
        )

    def forward(self, residual):
        # residual: (B,1,H,W)
        refined = self.refine(residual)
        # Skip connection: 초기 residual 더해주면 잔차 보존하면서 부드럽게 보정
        return residual + refined


# submodules/refine.py (또는 별도 파일)
import torch.nn as nn

class DilatedRefineBlock(nn.Module):
    def __init__(self, in_ch=1, mid_ch=32):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch2 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch3 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=4, dilation=4, bias=False)
        self.bn = nn.BatchNorm2d(mid_ch * 3)
        self.relu = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(mid_ch * 3, 1, kernel_size=1, bias=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        cat = torch.cat([b1, b2, b3], dim=1)
        out = self.relu(self.bn(cat))
        res = self.fuse(out)
        return x + res
