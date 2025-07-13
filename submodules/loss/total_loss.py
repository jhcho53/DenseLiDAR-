import torch
from submodules.loss.L1_Structural_loss import l_structural
from submodules.loss.L2_depth_loss import L2_depth_loss

def edge_aware_smoothness(depth, guide, beta=10.0):
    # depth: [B,1,H,W], guide: [B,1,H,W] (pseudo-depth)
    dzdx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    dzdy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    # guide gradient
    gdx = torch.abs(guide[:, :, :, :-1] - guide[:, :, :, 1:])
    gdy = torch.abs(guide[:, :, :-1, :] - guide[:, :, 1:, :])
    wx = torch.exp(-beta * gdx)
    wy = torch.exp(-beta * gdy)
    return (dzdx * wx).mean() + (dzdy * wy).mean()

def total_loss(pseudo_gt_map, gt_lidar, dense_depth, gamma=0.1):
    structural_loss = l_structural(pseudo_gt_map, dense_depth)
    depth_loss = L2_depth_loss(gt_lidar, dense_depth)
    sm_loss = edge_aware_smoothness(dense_depth, pseudo_gt_map) * gamma

    loss = 1.5 * structural_loss + depth_loss + sm_loss

    return loss, structural_loss, depth_loss, sm_loss

