import torch
import torch.nn.functional as F
from dataloader import dataLoader as lsn
from dataloader import trainLoader as DA
from model import DenseLiDAR
from tqdm import tqdm
import argparse
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Evaluate DenseLiDAR on validation set')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
parser.add_argument('--crop_eval', action='store_true', default=False, help='If set, crop images during evaluation (default: use full resolution)')
parser.add_argument('--output_dir', type=str, default='eval_outputs', help='Directory to save predicted depth maps')
args = parser.parse_args()

def remove_module_prefix(state_dict):
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

def save_depth_map_colored(depth_tensor, save_path, cmap='jet'):
    """
    depth_tensor: torch.Tensor, shape [1, H, W] or [H, W]
    save_path: output file path (.png)
    cmap: colormap to use, e.g., 'jet', 'magma', 'viridis'
    """
    depth = depth_tensor.squeeze().cpu().numpy()

    # 정규화 (0~1)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_norm_uint8 = (depth_norm * 255).astype(np.uint8)

    # OpenCV 컬러맵 적용
    if cmap == 'jet':
        colored = cv2.applyColorMap(depth_norm_uint8, cv2.COLORMAP_JET)
    elif cmap == 'magma':
        colored = cv2.applyColorMap(depth_norm_uint8, cv2.COLORMAP_MAGMA)
    elif cmap == 'viridis':
        colored = cv2.applyColorMap(depth_norm_uint8, cv2.COLORMAP_VIRIDIS)
    else:
        raise ValueError(f"Unsupported colormap: {cmap}")

    cv2.imwrite(save_path, colored)

def evaluate_validation_dataset(model_path, data_path, batch_size=1, device='cuda', crop_eval=False, output_dir='eval_outputs'):
    os.makedirs(output_dir, exist_ok=True)

    val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map = lsn.dataloader(data_path, mode='val')
    val_dataset = DA.myImageFloder(val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map, training=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseLiDAR(bs=batch_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(remove_module_prefix(checkpoint['model_state_dict']))
    model.eval()

    total_l1_loss = 0.0
    total_sq_loss = 0.0
    total_imae = 0.0
    total_irmse = 0.0
    total_points = 0

    with torch.no_grad():
        for idx, (image, gt, sparse, pseudo_depth_map, pseudo_gt_map) in enumerate(tqdm(val_loader, desc='Validation Eval')):
            image = image.to(device)
            gt = gt.to(device)
            sparse = sparse.to(device)
            pseudo_depth_map = pseudo_depth_map.to(device)

            mask = (gt > 0).float()
            valid_pixels = mask.sum().item()

            residual = model(image, sparse, pseudo_depth_map, device)
            pred_depth = pseudo_depth_map + residual

            # 저장
            for b in range(pred_depth.size(0)):
                save_name = os.path.join(output_dir, f"depth_{idx * batch_size + b:05d}.png")
                save_depth_map_colored(pred_depth[b], save_name, cmap='jet')


            # MAE / RMSE
            l1_loss = torch.abs(pred_depth - gt) * mask
            total_l1_loss += l1_loss.sum().item()

            sq_loss = ((pred_depth - gt) ** 2) * mask
            total_sq_loss += sq_loss.sum().item()

            # iMAE / iRMSE [1/km]
            gt_km = gt / 1000.0
            pred_km = pred_depth / 1000.0
            gt_inv = torch.where(gt_km > 1e-6, 1.0 / gt_km, torch.zeros_like(gt_km))
            pred_inv = torch.where(pred_km > 1e-6, 1.0 / pred_km, torch.zeros_like(pred_km))
            inv_diff = (pred_inv - gt_inv) * mask

            total_imae += inv_diff.abs().sum().item()
            total_irmse += (inv_diff ** 2).sum().item()
            total_points += valid_pixels

    mae = total_l1_loss / total_points
    rmse = (total_sq_loss / total_points) ** 0.5
    imaekm = total_imae / total_points
    irmsekmsq = (total_irmse / total_points) ** 0.5

    print(f'\n[Evaluation Result]')
    print(f'  ▸ MAE     = {mae * 1000:.1f} mm')
    print(f'  ▸ RMSE    = {rmse * 1000:.1f} mm')
    print(f'  ▸ iMAE    = {imaekm:.6f} [1/km]')
    print(f'  ▸ iRMSE   = {irmsekmsq:.6f} [1/km]')

    return mae, rmse, imaekm, irmsekmsq

if __name__ == '__main__':
    evaluate_validation_dataset(
        model_path=args.model_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        device='cuda',
        crop_eval=args.crop_eval,
        output_dir=args.output_dir
    )
