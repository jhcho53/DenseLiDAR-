import os
import glob
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

def load_mm_depth(path: str) -> np.ndarray:
    """
    - uint16(mm) PNG → float32 meters
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"File not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d.astype(np.float32) / 256.0

def load_mm_depth2(path: str) -> np.ndarray:
    """
    - uint16(mm) PNG → float32 meters
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"File not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d.astype(np.float32) / 255.0

def build_Laplacian(H: int, W: int) -> sparse.csr_matrix:
    """
    5-point Laplacian stencil (no BC), shape (HW, HW)
    """
    N = H * W
    main_diag = np.full(N, 4, dtype=np.float32)
    off_diag  = np.full(N, -1, dtype=np.float32)
    diags     = [main_diag, off_diag, off_diag, off_diag, off_diag]
    offs      = [0, -1, +1, -W, +W]
    A = sparse.diags(diags, offs, shape=(N, N), format='csr')
    # 좌우 wrap-around 방지
    for i in range(H):
        idx = i * W
        if idx - 1 >= 0:
            A[idx, idx - 1] = 0
            A[idx - 1, idx] = 0
    return A

def get_A_base(h: int, w: int, cache_dir: str) -> sparse.csr_matrix:
    """
    h×w 해상도에 대응하는 Laplacian 행렬을,
    없으면 build_Laplacian으로 생성 후 저장하고,
    있으면 로드해서 반환.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"A_base_{h}x{w}.npz")
    if os.path.exists(fn):
        return sparse.load_npz(fn)
    A = build_Laplacian(h, w)
    sparse.save_npz(fn, A)
    return A

def poisson_complete(sparse_d: np.ndarray,
                     est_d:   np.ndarray,
                     A_base:  sparse.csr_matrix) -> np.ndarray:
    """
    - sparse_d: GT depth [m], 0인 곳이 unknown
    - est_d:    initial estimate [m]
    - A_base:   no-BC Laplacian from build_Laplacian
    """
    H, W = est_d.shape
    N    = H * W

    # 1) Dirichlet mask: GT 포인트 + 이미지 테두리
    mask_sparse = (sparse_d > 0)
    mask_border = np.zeros_like(mask_sparse)
    mask_border[0, :] = mask_border[-1, :] = True
    mask_border[:, 0] = mask_border[:, -1] = True
    mask_known = (mask_sparse | mask_border).flatten()

    # 2) Poisson RHS: est_d 그라디언트 발산
    gx    = cv2.Sobel(est_d, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(est_d, cv2.CV_32F, 0, 1, ksize=3)
    div_x = cv2.Sobel(gx,    cv2.CV_32F, 1, 0, ksize=3)
    div_y = cv2.Sobel(gy,    cv2.CV_32F, 0, 1, ksize=3)
    b_full = (div_x + div_y).flatten()

    # 3) Known values 벡터 (GT or border est)
    d_flat = sparse_d.flatten()
    e_flat = est_d.flatten()
    v_known = np.where(mask_sparse.flatten(), d_flat, e_flat)

    # 4) Unknown-only 시스템 추출
    idx_k = np.nonzero(mask_known)[0]
    idx_u = np.nonzero(~mask_known)[0]

    A_uu = A_base[idx_u][:, idx_u]
    A_uk = A_base[idx_u][:, idx_k]
    b_u  = b_full[idx_u] - A_uk.dot(v_known[idx_k])

    # 5) Solve
    x_u = spsolve(A_uu, b_u).astype(np.float32)

    # 6) Reconstruct
    x_full = np.empty(N, dtype=np.float32)
    x_full[idx_k] = v_known[idx_k]
    x_full[idx_u] = x_u
    pseudo = x_full.reshape(H, W)

    # 7) Clip to [0, max_gt]
    max_gt = float(sparse_d.max())
    return np.clip(pseudo, 0.0, max_gt)

def build_dataset(root: str, out_root: str, mode: str):
    ann02 = glob.glob(os.path.join(root, 'data_depth_annotated', mode,
                                   '**/proj_depth/groundtruth/image_02/*.png'),
                      recursive=True)
    ann03 = glob.glob(os.path.join(root, 'data_depth_annotated', mode,
                                   '**/proj_depth/groundtruth/image_03/*.png'),
                      recursive=True)
    annotated = sorted(ann02) + sorted(ann03)
    raw02 = glob.glob(os.path.join(root, 'kitti_raw', mode,
                                   '**/proj_depth/image_02/*.png'),
                      recursive=True)
    raw03 = glob.glob(os.path.join(root, 'kitti_raw', mode,
                                   '**/proj_depth/image_03/*.png'),
                      recursive=True)
    raws = sorted(raw02) + sorted(raw03)
    assert len(annotated) == len(raws), "개수 불일치"

    save_dir = os.path.join(out_root, mode)
    os.makedirs(save_dir, exist_ok=True)

    for gt_p, est_p in tqdm(zip(annotated, raws), total=len(annotated), desc=f"Poisson {mode}"):
        # 상대 경로 유지
        rel      = os.path.relpath(gt_p, os.path.join(root, 'data_depth_annotated', mode))
        out_path = os.path.join(save_dir, rel)
        if os.path.exists(out_path):
            continue

        sparse = load_mm_depth(gt_p)
        est    = load_mm_depth2(est_p)
        h, w   = sparse.shape

        # 해상도별로 A_base 얻기
        A_base = get_A_base(h, w, cache_dir=out_root)

        pseudo = poisson_complete(sparse, est, A_base)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pseudo_uint16 = (pseudo * 256.0).astype(np.uint16)
        cv2.imwrite(out_path, pseudo_uint16)

    print(f"[{mode}] saved in {save_dir}")

if __name__ == "__main__":
    ROOT = "/home/vip/Desktop/DC/DenseLiDAR/datasets"
    OUT  = os.path.join(ROOT, "poisson_depth_map")
    for m in ("train", "val"):
        build_dataset(ROOT, OUT, m)
    print("=== 완료 ===")