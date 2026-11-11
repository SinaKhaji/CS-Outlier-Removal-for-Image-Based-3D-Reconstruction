import numpy as np
import open3d as o3d

def view(samples):
    windows = []
    for name, m in samples.items():
        w = o3d.visualization.Visualizer()
        w.create_window(window_name=name)
        w.add_geometry(m)
        windows.append(w)
    while all(w.poll_events() for w in windows):
        for w in windows:
            w.update_renderer()

def load_bunny(dir):
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh(data_root=dir).path)
    return mesh

def texture(P):
    return (127.5 + 127.5 * np.sin(P * [40, 55, 70] + [0, 2, 4])).astype(np.uint8)

def render_views(mesh, n_views=200, dist=1.0, W=640, H=480, f=525.0, outliers=0.05, seed=0):
    rng = np.random.default_rng(seed)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    K = np.array([f, f, (W - 1) / 2, (H - 1) / 2], np.float32)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays_cam = np.stack([(u - K[2]) / f, (v - K[3]) / f, np.ones((H, W))], -1).reshape(-1, 3)
    views = []
    for k in range(n_views):
        az, el = 2 * np.pi * k / (n_views / 4), np.deg2rad([-20, 0, 20, 40][k % 4])
        C = dist * np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
        z = -C / np.linalg.norm(C)                                   # look at the origin, y is up
        x = np.cross(z, [0, 1, 0]); x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])                         # world -> camera
        rays = np.hstack([np.tile(C, (W * H, 1)), rays_cam @ R]).astype(np.float32)
        D = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)   # z-depth (ray z = 1)
        D[~np.isfinite(D)] = 0
        m = D > 0
        col = np.zeros((H, W, 3), np.uint8)
        col[m] = texture(C + D[m, None] * (rays_cam @ R).reshape(H, W, 3)[m])     # true surface colour
        D[m] += rng.normal(0, 0.0012 + 0.0019 * (D[m] - 0.4) ** 2)   # Kinect noise (Nguyen et al. 2012)
        out = m & (rng.random((H, W)) < outliers)
        D[out] = rng.uniform(D[m].min(), D[m].max(), out.sum())      # outliers
        views.append(dict(name=f"view_{k:03d}", R=R.astype(np.float32), t=(-R @ C).astype(np.float32),
                          axis=R[2], K=K, H=H, W=W, D=D.astype(np.float32), col=col))
    return views

def load_views(scene, n_views=200):
    K = np.array([517.3, 516.5, 318.6, 255.3], np.float32)          # fr1 colour intrinsics (TUM)
    depth = np.loadtxt(scene + "/depth.txt", dtype=str)             # timestamp, file   ('#' lines skipped)
    rgb   = np.loadtxt(scene + "/rgb.txt", dtype=str)               # timestamp, file
    gt    = np.loadtxt(scene + "/groundtruth.txt")                  # timestamp tx ty tz qx qy qz qw
    t_d, t_c = depth[:, 0].astype(float), rgb[:, 0].astype(float)
    views = []
    for i in np.linspace(0, len(depth) - 1, n_views).astype(int):   # n_views evenly spread frames
        c = np.abs(t_c - t_d[i]).argmin()                           # closest colour image
        g = np.abs(gt[:, 0] - t_d[i]).argmin()                      # closest pose
        tx, ty, tz, qx, qy, qz, qw = gt[g, 1:]
        R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz]).T   # world -> camera
        t = -R @ np.array([tx, ty, tz])
        D = np.asarray(o3d.io.read_image(scene + "/" + depth[i, 1]), np.float32) / 5000  # metres
        col = np.asarray(o3d.io.read_image(scene + "/" + rgb[c, 1]))[..., :3]
        views.append(dict(name=depth[i, 1], R=R.astype(np.float32), t=t.astype(np.float32), axis=R[2],
                          K=K, H=D.shape[0], W=D.shape[1], D=D, col=col))
    return views

def save_all_points(V, path):
    pts, nrm, cols = [], [], []
    for Vi in V:
        prepare_view(Vi, 7, 1.0)
        m = Vi["valid"].ravel()
        pts.append((Vi["X"].reshape(-1, 3)[m] - Vi["t"]) @ Vi["R"])
        nrm.append(Vi["N"].reshape(-1, 3)[m] @ Vi["R"])
        cols.append(Vi["col"].reshape(-1, 3)[m])
    save_cloud(path, np.concatenate(pts), np.concatenate(nrm), np.concatenate(cols))

def poisson(src, dst, depth):
    pcd = o3d.io.read_point_cloud(src)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, n_threads=1)
    o3d.io.write_triangle_mesh(dst, mesh)
    return mesh

def save_cloud(path, points, normals, colors):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    pcd.colors  = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(path, pcd)

# --------------------------------------------------------------------------- #
# Filter: Wolff et al., 3DV 2016 (supplement, Algorithm 1)
# Per-depth-map preprocessing
# --------------------------------------------------------------------------- #
def box_sum(a, k):
    """Sum over a k x k window (zero padded), a: (H,W,...)."""
    r = k // 2
    p = np.pad(a, [(r + 1, r), (r + 1, r)] + [(0, 0)] * (a.ndim - 2))
    c = p.cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def pca_normals(X, valid, k):
    """Normals by PCA of the k x k image neighbourhood, oriented towards the camera."""
    m = valid[..., None].astype(np.float64)
    Xm = np.where(valid[..., None], X, 0).astype(np.float64)
    n = np.maximum(box_sum(m, k), 1)
    mu = box_sum(Xm, k) / n
    outer = (Xm[..., :, None] * Xm[..., None, :]).reshape(*X.shape[:2], 9)
    C = box_sum(outer, k).reshape(*X.shape[:2], 3, 3) / n[..., None] - mu[..., :, None] * mu[..., None, :]
    _, vec = np.linalg.eigh(C)
    N = vec[..., :, 0]                                       # smallest eigenvalue
    N *= np.where((N * X).sum(-1, keepdims=True) > 0, -1, 1) # point towards camera (origin)
    return N.astype(np.float32)


def triangle_ok(X, valid, min_deg):
    """Validity of the two triangles of every pixel quad: upper-left A, lower-right B."""
    def min_angle_ok(a, b, c):
        cos_max = np.cos(np.deg2rad(min_deg))
        ok = np.ones(a.shape[:2], bool)
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            u, v = q - p, r - p
            cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)
            ok &= cos < cos_max
        return ok
    v00, v01, v10, v11 = X[:-1, :-1], X[:-1, 1:], X[1:, :-1], X[1:, 1:]
    m00, m01, m10, m11 = valid[:-1, :-1], valid[:-1, 1:], valid[1:, :-1], valid[1:, 1:]
    A = m00 & m01 & m10 & min_angle_ok(v00, v01, v10)
    B = m11 & m10 & m01 & min_angle_ok(v11, v10, v01)
    return A, B


def prepare_view(Vi, pca_patch, min_angle):
    """Back-project the depth map and add normals, weights (Eq. 3) and valid triangles."""
    D, (fx, fy, cx, cy) = Vi["D"], Vi["K"]
    valid = D > 0
    rr, cc = np.mgrid[0:Vi["H"], 0:Vi["W"]].astype(np.float32)
    X = np.stack([(cc - cx) / fx * D, (rr - cy) / fy * D, D], -1)              # camera coords
    N = pca_normals(X, valid, pca_patch)
    wgt = np.clip(-(N * X).sum(-1) / np.maximum(np.linalg.norm(X, axis=-1), 1e-12), 0, 1)  # Eq. 3
    A, B = triangle_ok(X, valid, min_angle)
    Vi.update(valid=valid, X=X, N=N, wgt=wgt.astype(np.float32), A=A, B=B)


# --------------------------------------------------------------------------- #
# Algorithm 1: test every point of depth map i against all other depth maps
# --------------------------------------------------------------------------- #
def filter_view(i, V, sigma, td, tp, tv):
    Vi = V[i]
    m = Vi["valid"].ravel()
    Xc = Vi["X"].reshape(-1, 3)[m]
    P = (Xc - Vi["t"]) @ Vi["R"]                             # world coords  R^T (X - t)
    Nw = Vi["N"].reshape(-1, 3)[m] @ Vi["R"]
    c_own = Vi["col"].reshape(-1, 3)[m].astype(np.float32)
    M = len(P)
    # p itself: distance 0 to its own range surface, own weight and colour
    sw  = Vi["wgt"].ravel()[m].astype(np.float64)
    swd = np.zeros(M)
    vis = np.ones(M, np.int32)
    s   = c_own.astype(np.float64)
    s2  = (c_own.astype(np.float64) ** 2).sum(1)
    for j, Vj in enumerate(V):
        if j == i or Vi["axis"] @ Vj["axis"] <= 0:           # only views within 90 deg
            continue
        Xj = P @ Vj["R"].T + Vj["t"]
        z = Xj[:, 2]
        fx, fy, cx, cy = Vj["K"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * Xj[:, 0] / z + cx
            v = fy * Xj[:, 1] / z + cy
        inside = (z > 0) & (u >= 0) & (v >= 0) & (u < Vj["W"] - 1) & (v < Vj["H"] - 1)
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            continue
        u, v, z = u[idx], v[idx], z[idx]
        c0, r0 = u.astype(np.int32), v.astype(np.int32)
        fu, fv = u - c0, v - r0
        upper = fu + fv <= 1                                 # upper-left triangle A, else lower-right B
        keep = np.where(upper, Vj["A"][r0, c0], Vj["B"][r0, c0])   # skip removed / missing triangles
        idx, r0, c0, fu, fv, upper, z = idx[keep], r0[keep], c0[keep], fu[keep], fv[keep], upper[keep], z[keep]
        # triangle corners + barycentric weights
        ra = np.where(upper, r0, r0 + 1);     ca = np.where(upper, c0, c0 + 1)
        rb, cb = r0, c0 + 1
        rc, ccn = r0 + 1, c0
        la = np.where(upper, 1 - fu - fv, fu + fv - 1)
        lb = np.where(upper, fu, 1 - fv)
        lc = np.where(upper, fv, 1 - fu)

        def interp(F, sel=slice(None)):
            a, b, c = la[sel], lb[sel], lc[sel]
            if F.ndim == 3:
                a, b, c = a[:, None], b[:, None], c[:, None]
            return a * F[ra[sel], ca[sel]] + b * F[rb[sel], cb[sel]] + c * F[rc[sel], ccn[sel]]

        d = interp(Vj["D"]) - z                              # Eq. 1
        w = interp(Vj["wgt"])
        g = d > -sigma                                       # Eq. 2: not too far behind
        dg = np.minimum(d, sigma)                            # truncate in front
        sw[idx]  += np.where(g, w, 0)
        swd[idx] += np.where(g, w * dg, 0)
        ph = g & (d < sigma)                                 # Eq. 5: close to the surface
        ip = idx[ph]
        col = interp(Vj["col"], ph)
        vis[ip] += 1
        s[ip]   += col
        s2[ip]  += (col.astype(np.float64) ** 2).sum(1)
    dp = swd / np.maximum(sw, 1e-12)                         # Eq. 4
    var = np.maximum(s2 / vis - (s ** 2).sum(1) / vis ** 2, 0)
    pp = np.sqrt(var) * 2 / (255 * np.sqrt(3))               # Eq. 7 scaled to [0,1]
    keep = (dp > -td) & (dp < 0) & (pp < tp) & (vis > tv)    # Eq. 8
    return P[keep], Nw[keep], c_own[keep].astype(np.uint8), keep


def denoise(V, cloud_path,
            sigma=None, sigma_ratio=0.01, td_ratio=0.1, tp=0.2, tv_ratio=0.075,
            min_angle=1.0, pca_patch=7):
    """Run the filter on all views (defaults = the paper's parameters, Sec. 4).

    Saves the filtered cloud (points + normals + colours) to cloud_path.
    """
    for Vi in V:
        prepare_view(Vi, pca_patch, min_angle)
    all_d = np.concatenate([x["D"][x["valid"]][::50] for x in V])
    d_lo, d_hi = np.percentile(all_d, [1, 99])               # robust depth range
    sigma = sigma if sigma is not None else sigma_ratio * (d_hi - d_lo)
    td, tv = td_ratio * sigma, tv_ratio * len(V)
    print(f"depth range [{d_lo:.3f}, {d_hi:.3f}] -> sigma={sigma:.4f}, t_d={td:.4f}, t_p={tp}, t_v={tv:.2f}")

    pts, nrm, cols, n_in = [], [], [], 0
    for i in range(len(V)):
        P_i, N_i, C_i, keep = filter_view(i, V, sigma, td, tp, tv)
        pts.append(P_i); nrm.append(N_i); cols.append(C_i); n_in += len(keep)
        print(f"[{i+1:3d}/{len(V)}] {V[i]['name']}: kept {keep.mean()*100:5.1f}%", flush=True)
    pts, nrm, cols = np.concatenate(pts), np.concatenate(nrm), np.concatenate(cols)
    print(f"input {n_in:,} points -> kept {len(pts):,} ({len(pts)/n_in*100:.1f}%)")

    save_cloud(cloud_path, pts, nrm, cols)