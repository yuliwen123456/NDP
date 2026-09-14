import os
import numpy as np
from scipy.spatial import cKDTree
import random
from sklearn.cluster import DBSCAN
from collections import Counter
import shutil


SEED = 2025
np.random.seed(SEED)
random.seed(SEED)
rng = np.random.default_rng(SEED)
### === Configuration === ###
INPUT_SEQ_DIR = r"data/original/206/velodyne"
INPUT_LABEL_DIR = r"data/original/206/labels"
OUTPUT_SEQ_DIR = r"data/train/206/velodyne"
OUTPUT_LABEL_DIR = r"data/train/206/labels"

RAISE_HEIGHT_RANGE = (0.0, 0.5)
CLUSTER_RADIUS = 0.5
RAISED_CLASS = 2
ROAD_CLASS = 40

os.makedirs(OUTPUT_SEQ_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

shutil.copy("data/original/206/poses.txt", "data/train/206/poses.txt")
shutil.copy("data/original/206/calib.txt", "data/train/206/calib.txt")
#############################

def load_pointcloud(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def load_labels(label_path):
    return np.fromfile(label_path, dtype=np.uint32)

def save_pointcloud(bin_path, points):
    points.astype(np.float32).tofile(bin_path)

def save_labels(label_path, labels):
    labels.astype(np.uint32).tofile(label_path)

def raise_cluster_with_pull(points, labels, class_id=40, cluster_radius=1.0, height_range=(1.2, 2.5), pull_factor = 2):
    road_mask = labels == class_id
    road_points = points[road_mask]
    road_indices = np.where(road_mask)[0]

    if len(road_points) == 0:
        print("No road points found.")
        return points, []

    # Choose center
    center_idx = np.random.choice(len(road_points))
    center_point = road_points[center_idx, :3]

    # Get nearby points
    tree = cKDTree(road_points[:, :3])
    cluster_local_idx = tree.query_ball_point(center_point, r=cluster_radius)
    cluster_global_idx = road_indices[cluster_local_idx]

    cluster_xyz = points[cluster_global_idx, :3]
    distances = np.linalg.norm(cluster_xyz, axis=1, keepdims=True)
    dist_min = np.min(distances)
    dist_max = np.max(distances)

    #dist_min = dist_max * e^-a(dist_max-dist_min)
    #dist_min/dist_max = e^-a(dist_max-dist_min)
    a = -np.log(dist_min / dist_max) / (dist_max - dist_min)
    a/=pull_factor
    distances = distances - np.min(distances)
    shift = np.exp(-distances*a)
    points[cluster_global_idx, :3] *= shift
    # Different height for each point
    random_heights = np.random.uniform(*height_range, size=len(cluster_global_idx))
    points[cluster_global_idx, 2] += random_heights
    labels[cluster_global_idx] = RAISED_CLASS

    return points, labels

# ---------------- Perlin grid (fBm) ----------------
def _fade(t):  # quintic smoothstep
    return t*t*t*(t*(t*6 - 15) + 10)

def _perlin_grid(H, W, res_h, res_w, rng):
    """
    Tileable 2-D Perlin sampled on an HxW grid; res_h/res_w periods across the grid.
    Output in ~[-1,1] before later min-max normalization.
    """
    theta = rng.random((res_h+1, res_w+1)) * 2*np.pi
    g = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (Rh+1, Rw+1, 2)

    ys = np.linspace(0, res_h, H, endpoint=False)
    xs = np.linspace(0, res_w, W, endpoint=False)
    xi = xs.astype(int)
    yi = ys.astype(int)
    xf = xs - xi
    yf = ys - yi

    u = _fade(xf)[None, :]      # (1,W)
    v = _fade(yf)[:, None]      # (H,1)

    def gxy(ix, iy):            # wrap
        return g[iy % (res_h+1), ix % (res_w+1)]

    def dot(ix, iy, dx, dy):
        gr = gxy(ix, iy)
        return gr[...,0]*dx + gr[...,1]*dy

    Xf = xf[None, :]
    Yf = yf[:, None]
    n00 = dot(xi[None,:],   yi[:,None],   Xf,     Yf    )
    n10 = dot(xi[None,:]+1, yi[:,None],   Xf-1.0, Yf    )
    n01 = dot(xi[None,:],   yi[:,None]+1, Xf,     Yf-1.0)
    n11 = dot(xi[None,:]+1, yi[:,None]+1, Xf-1.0, Yf-1.0)

    nx0 = n00*(1-u) + n10*u
    nx1 = n01*(1-u) + n11*u
    n = nx0*(1-v) + nx1*v
    return n.astype(np.float32)

def fbm_perlin_grid(H, W, base_res=(3,3), octaves=3, persistence=0.55, lacunarity=2.0, seed=0):
    total = np.zeros((H, W), dtype=np.float32)
    amp = 1.0; amp_sum = 0.0
    rh, rw = float(base_res[0]), float(base_res[1])
    for _ in range(octaves):
        Rh = max(1, int(np.ceil(rh)))
        Rw = max(1, int(np.ceil(rw)))
        total += amp * _perlin_grid(H, W, Rh, Rw, rng)
        amp_sum += amp
        amp *= persistence
        rh *= lacunarity; rw *= lacunarity
    total /= (amp_sum + 1e-8)
    # Normalize to [0,1]
    tmin, tmax = total.min(), total.max()
    if tmax > tmin:
        total = (total - tmin) / (tmax - tmin)
    return total

# ---------------- Bulge-only augmentation (no normals) ----------------
def perlin_raise(points,
                                     labels,
                                     class_id=40,
                                     target_ratio=0.30,     # fraction of patch points to raise
                                     strength=0.40,         # meters: peak uplift after local norm
                                     patch_radius=1.2,      # meters
                                     grid_res=192,          # Perlin grid resolution
                                     base_res=(3,3),
                                     octaves=3,
                                     persistence=0.55,
                                     lacunarity=2.0,
                                     seed=123,
                                     raise_threshold = 0.01,
                                     RAISED_CLASS=2,
                                     debug=True):
    """
    Steps:
      1) Pick a road-centered patch (radius in meters).
      2) Build a Perlin-fBm grid over the patch bounding box in XY (global).
      3) Map each patch point to the grid cell; take its Perlin value n in [0,1].
      4) Keep the top-quantile points (bulge-only set). Locally min–max normalize n there to [0,1].
      5) Add dz = strength * gain to GLOBAL Z ONLY (no normals).

    Returns (points_out, modified_indices).
    """

    # 1) select road patch
    road_mask = (labels == class_id)
    if not np.any(road_mask):
        if debug: print("[BULGE0] No points with the given class_id.")
        return points, labels

    road_pts = points[road_mask]
    road_idx = np.where(road_mask)[0]

    center = road_pts[rng.integers(len(road_pts)), :3]
    tree = cKDTree(road_pts[:, :3])
    loc = tree.query_ball_point(center, r=patch_radius)
    if len(loc) < 5:
        if debug: print(f"[BULGE0] Sparse patch ({len(loc)} pts). Increase patch_radius.")
        return points, labels
    gi = road_idx[np.asarray(loc, int)]
    X = points[gi, :3].copy()

    # 2) grid over XY bbox (with padding)
    x0, x1 = X[:,0].min(), X[:,0].max()
    y0, y1 = X[:,1].min(), X[:,1].max()
    pad_x = 0.05 * max(1e-3, x1 - x0)
    pad_y = 0.05 * max(1e-3, y1 - y0)
    x0 -= pad_x; x1 += pad_x; y0 -= pad_y; y1 += pad_y

    perlin = fbm_perlin_grid(grid_res, grid_res,
                             base_res=base_res, octaves=octaves,
                             persistence=persistence, lacunarity=lacunarity, seed=seed)

    # 3) sample per-point Perlin via nearest grid cell (fast and robust); could switch to bilinear if desired
    tx = (X[:,0] - x0) / (x1 - x0 + 1e-12) * (grid_res - 1)
    ty = (X[:,1] - y0) / (y1 - y0 + 1e-12) * (grid_res - 1)
    ix = np.clip(np.round(tx).astype(int), 0, grid_res-1)
    iy = np.clip(np.round(ty).astype(int), 0, grid_res-1)
    nval = perlin[iy, ix]  # in [0,1]

    # 4) bulge-only: threshold by high quantile; local min–max in mask; dz >= 0
    if target_ratio <= 0.0:
        mask = np.zeros_like(nval, dtype=bool)
    elif target_ratio >= 1.0:
        mask = np.ones_like(nval, dtype=bool)
    else:
        th = np.quantile(nval, 1.0 - target_ratio)
        mask = nval >= th
        # ensure non-empty
        if mask.sum() == 0:
            # relax slightly
            th = np.quantile(nval, 1.0 - 0.9*target_ratio)
            mask = nval >= th

    dz = np.zeros_like(nval, dtype=np.float32)
    if mask.any():
        sel = nval[mask]
        smin, smax = sel.min(), sel.max()
        if smax > smin + 1e-8:
            gain = (nval - smin) / (smax - smin + 1e-8)
        else:
            gain = np.zeros_like(nval)
        gain = np.clip(gain, 0.0, 1.0)
        dz[mask] = gain[mask] * strength  # strictly upward

    big = (dz >= raise_threshold)
    dz[~big]=0
    # 5) apply global-Z displacement only
    X[:,2] += dz
    
    out = points.copy()
    out[gi[big], :3] = X[big]

    clusters = (
    DBSCAN(eps=0.1, min_samples=1, n_jobs=-1)
    .fit(out[gi[big], :3])
    .labels_
    )
    #print(clusters)
    most_freq = Counter(clusters).most_common(1)[0][0]
    if most_freq>-1:
        cluster_filter = (clusters==most_freq)
        out = points.copy()
        out[gi[big][cluster_filter], :3] = X[big][cluster_filter]

    if debug:
        frac = mask.mean() if len(mask) else 0.0
        print(f"[BULGE0] patch_pts={len(gi)} affected={mask.sum()} ({frac*100:.1f}%) "
              f"dz_max={dz.max():.3f} m, strength={strength}")

    labels[gi[big][cluster_filter]] = RAISED_CLASS
    return out, labels


def augment_sequence():
    frame_ids = sorted([f[:-4] for f in os.listdir(INPUT_SEQ_DIR) if f.endswith(".bin")])

    for fid in frame_ids:
        bin_path = os.path.join(INPUT_SEQ_DIR, f"{fid}.bin")
        label_path = os.path.join(INPUT_LABEL_DIR, f"{fid}.label")

        points = load_pointcloud(bin_path)
        labels = load_labels(label_path)
        sem_labels = labels & 0xFFFF  # extract semantic part
        num = np.random.uniform(-0.5, 0.25)
        num2 = np.random.uniform(-0.25, 0.5)
        # Augment
        '''
        points_aug, sem_labels_aug = raise_cluster_with_pull(
            points.copy(), sem_labels.copy(),
            class_id=ROAD_CLASS,
            cluster_radius=CLUSTER_RADIUS+num,
            height_range=(0,0.5+num2)
        )
        '''
        points_aug, sem_labels_aug = perlin_raise(
            points.copy(), sem_labels.copy(),
            class_id=40,
            target_ratio=0.30,     # footprint size
            strength=0.5 + num2,         # peak height in meters
            patch_radius=1.25 + num,
            grid_res=192,
            base_res=(3,3),
            octaves=3,
            seed=0,
            debug=True
        )

        # Re-encode labels (preserve instance ID if needed)
        final_labels = (labels & 0xFFFF0000) | (sem_labels_aug & 0xFFFF)

        # Save
        save_pointcloud(os.path.join(OUTPUT_SEQ_DIR, f"{fid}.bin"), points_aug)
        save_labels(os.path.join(OUTPUT_LABEL_DIR, f"{fid}.label"), final_labels)

        print(f"Augmented frame {fid}")

augment_sequence()
