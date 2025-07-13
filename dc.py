# build_poisson_dataset.py

import os, glob, cv2, numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def load_mm_depth(path: str) -> np.ndarray:
    """
    - uint16(mm) PNG → float32 meters
    """
    # print(f"[load_mm_depth] Loading GT from: {path}")
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"File not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # print(f"[GT] min: {d.min()}, max: {d.max()} (unit: mm)")
    return d.astype(np.float32) / 256.0



def load_mm_depth2(path: str) -> np.ndarray:
    """
    - uint16(mm) PNG → float32 meters
    """
    # print(f"[load_mm_depth2] Loading Estimation from: {path}")
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"File not found: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # print(f"[Estimation] min: {d.min()}, max: {d.max()} (unit: mm)")
    return d.astype(np.float32) / 255.0

def poisson_complete(sparse_d: np.ndarray, est_d: np.ndarray) -> np.ndarray:
    H, W = est_d.shape
    N    = H * W

    mask_sparse = (sparse_d > 0)
    # --- 2) Poisson RHS: Estimation gradient divergence -------------
    gx    = cv2.Sobel(est_d, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(est_d, cv2.CV_32F, 0, 1, ksize=3)
    div_x = cv2.Sobel(gx,        cv2.CV_32F, 1, 0, ksize=3)
    div_y = cv2.Sobel(gy,        cv2.CV_32F, 0, 1, ksize=3)
    b     = (div_x + div_y).flatten()

    # --- 3) Poisson LHS: 5-point Laplacian --------------------------
    main_diag = np.ones(N, dtype=np.float32) * 4
    off_diag  = np.ones(N, dtype=np.float32) * -1
    diags     = [main_diag, off_diag, off_diag, off_diag, off_diag]
    offs      = [0, -1, +1, -W, +W]
    A         = sparse.diags(diags, offs, shape=(N, N), format='csr')

    # --- 4) Dirichlet BC: sparse locations fixed -------------------
    d_flat    = sparse_d.flatten()
    idx_valid = np.nonzero(mask_sparse.flatten())[0]
    for idx in idx_valid:
        A.data[A.indptr[idx]:A.indptr[idx+1]] = 0
        A[idx, idx] = 1
        b[idx]      = d_flat[idx]

    # --- 5) Solve Ax = b → pseudo-dense depth [m] ----------------
    x      = spsolve(A, b)
    pseudo = x.reshape(H, W).astype(np.float32)
    max_depth_m = float(sparse_d.max())
    pseudo = np.clip(pseudo, 0.0, max_depth_m)
    return pseudo

def build_dataset(root, out_root, mode):
    # 1) GT 포인트 (Dirichlet BC)
    ann02 = glob.glob(os.path.join(root, 'data_depth_annotated', mode,
                                   '**/proj_depth/groundtruth/image_02/*.png'),
                      recursive=True)
    ann03 = glob.glob(os.path.join(root, 'data_depth_annotated', mode,
                                   '**/proj_depth/groundtruth/image_03/*.png'),
                      recursive=True)
    annotated = sorted(ann02) + sorted(ann03)

    # 2) 초기 estimation (gradient source)
    raw02 = glob.glob(os.path.join(root, 'kitti_raw', mode,
                                   '**/proj_depth/image_02/*.png'),
                      recursive=True)
    raw03 = glob.glob(os.path.join(root, 'kitti_raw', mode,
                                   '**/proj_depth/image_03/*.png'),
                      recursive=True)
    raws = sorted(raw02) + sorted(raw03)

    assert len(annotated)==len(raws), \
        f"개수 불일치: annotated={len(annotated)}, raw={len(raws)}"

    save_dir = os.path.join(out_root, mode)
    for gt_p, est_p in tqdm(zip(annotated, raws),
                             total=len(annotated),
                             desc=f"Poisson {mode}"):
        sparse = load_mm_depth(gt_p)
        est    = load_mm_depth2(est_p)
        pseudo = poisson_complete(sparse, est)

        rel = os.path.relpath(
            gt_p,
            os.path.join(root, 'data_depth_annotated', mode)
        )
        out_path = os.path.join(save_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 2) 보간된 깊이(pseudo)는 meters 단위 float32 → 
        #    다시 KITTI 포맷(uint16(depth[m]*256)) 으로 변환하여 저장
        pseudo_uint16 = (pseudo * 256.0).astype(np.uint16)
        cv2.imwrite(out_path, pseudo_uint16)

    print(f"[{mode}] saved {len(annotated)} files → {save_dir}")

if __name__ == "__main__":
    ROOT = "/home/vip/Desktop/DC/DenseLiDAR/datasets"
    OUT  = os.path.join(ROOT, "poisson_depth_map")
    for m in ("train","val"):
        build_dataset(ROOT, OUT, m)
    print("=== 완료 ===")