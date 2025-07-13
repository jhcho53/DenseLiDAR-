import os, glob, cv2, numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 깊이 맵 로드 함수 ---
def load_mm_depth(path: str) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"File not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d.astype(np.float32) / 256.0

# --- 초기 추정 로드 (0-1 정규화) ---
def load_mm_depth2(path: str) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"File not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d.astype(np.float32) / 255.0

# --- 5점 스텐실 라플라시안 생성 ---
def build_Laplacian(H: int, W: int) -> sparse.csr_matrix:
    N = H * W
    main_diag = np.full(N, 4, dtype=np.float32)
    off_diag  = np.full(N, -1, dtype=np.float32)
    diags     = [main_diag, off_diag, off_diag, off_diag, off_diag]
    offs      = [0, -1, +1, -W, +W]
    A = sparse.diags(diags, offs, shape=(N, N), format='csr')
    for i in range(H):  # 경계 wrap 방지
        idx = i * W
        if idx-1 >= 0:
            A[idx, idx-1] = 0
            A[idx-1, idx] = 0
    return A

# --- 포아송 보간 함수 ---
def poisson_complete(sparse_d, est_d, A_base):
    H, W = est_d.shape
    N = H*W
    # Dirichlet mask (GT + 테두리)
    mask_sparse = (sparse_d > 0)
    mask_border = np.zeros_like(mask_sparse)
    mask_border[[0,-1],:] = True
    mask_border[:,[0,-1]] = True
    mask_known = (mask_sparse | mask_border).flatten()
    # 초기 추정 그래디언트 및 발산
    gx = cv2.Sobel(est_d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(est_d, cv2.CV_32F, 0, 1, ksize=3)
    div = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    b_full = div.flatten()
    # 알려진 값 벡터
    d_flat = sparse_d.flatten(); e_flat = est_d.flatten()
    v_known = np.where(mask_sparse.flatten(), d_flat, e_flat)
    idx_k = np.where(mask_known)[0]; idx_u = np.where(~mask_known)[0]
    # 축소 시스템
    A_uu = A_base[idx_u][:, idx_u]
    A_uk = A_base[idx_u][:, idx_k]
    b_u = b_full[idx_u] - A_uk.dot(v_known[idx_k])
    x_u = spsolve(A_uu, b_u).astype(np.float32)
    # 재조립
    x_full = np.empty(N, np.float32)
    x_full[idx_k] = v_known[idx_k]
    x_full[idx_u] = x_u
    pseudo = x_full.reshape(H, W)
    return np.clip(pseudo, 0.0, float(sparse_d.max())), mask_sparse, div

# --- 예시 이미지 중간 시각화 ---
if __name__ == "__main__":
    root = "/home/vip/Desktop/DC/DenseLiDAR/datasets"
    mode = 'train'
    gt_paths = sorted(glob.glob(os.path.join(root, 'data_depth_annotated', mode,
                                          '**/proj_depth/groundtruth/image_02/*.png'),
                                  recursive=True))
    est_paths = sorted(glob.glob(os.path.join(root, 'kitti_raw', mode,
                                           '**/proj_depth/image_02/*.png'),
                                  recursive=True))
    gt = load_mm_depth(gt_paths[0])
    est = load_mm_depth2(est_paths[0])
    H, W = gt.shape
    A_base = build_Laplacian(H, W)
    pseudo, mask_sparse, div = poisson_complete(gt, est, A_base)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(gt, cmap='jet'); axes[0,0].set_title('Sparse GT')
    axes[0,1].imshow(est, cmap='jet'); axes[0,1].set_title('Initial Estimate')
    axes[0,2].imshow(mask_sparse, cmap='jet'); axes[0,2].set_title('Known Mask')
    axes[1,0].imshow(div, cmap='jet'); axes[1,0].set_title('Divergence')
    axes[1,1].imshow(pseudo, cmap='jet'); axes[1,1].set_title('Poisson Output')
    diff = np.abs(pseudo - est)
    axes[1,2].imshow(diff, cmap='hot'); axes[1,2].set_title('Diff (Output - Est)')
    for ax in axes.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()