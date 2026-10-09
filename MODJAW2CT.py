"""
MODJAW2CT: Anatomical layer+Kinematic layer+Patient-facing demonstration
MODJAW2CT.py
======================
Animate the mandible moving relative to the skull, driven by a ModJaw
(.jawmotion) recording, and render an MP4 with multiple views.

INPUTS
  --mandible   mandible bone STL (segmented from the CT, in CT/DICOM coords)
  --skull      skull/maxilla bone STL (same CT, same coordinate frame)
  --jawmotion  ModJaw .jawmotion XML file
  --scan       (optional but recommended) intra-oral UPPER-arch scan STL,
               in the SAME coordinate frame as the .jawmotion recording.
               Used to rigidly align the ModJaw frame to the CT frame.
               If omitted, the script assumes the CT bones are already in
               the ModJaw/upper_arch frame (W = identity).

OUTPUT
  --out        MP4 file (default: jaw_motion.mp4)

RUN MODES
  * GUI  : run with NO arguments (or --gui) -> a file-picker window opens
             python3 MODJAW2CT.py
  * CLI  : pass --mandible/--skull/--jawmotion (+ optional --scan/--level/...)
             python3 MODJAW2CT.py --mandible M.stl --skull S.stl \
                     --jawmotion J.jawmotion --scan scan.stl --out out.mp4 --level
  (GUI needs tkinter: ships with Python on Win/macOS; on Linux: apt install python3-tk)

WHAT IT DOES
  1. Parses the .jawmotion -> per-frame positions of the 3 mandible marks.
  2. Procrustes (Kabsch) per frame -> rigid mandible motion M_t (ModJaw frame).
  3. Aligns ModJaw frame -> CT frame (W) by ICP of the intra-oral scan onto
     the CT maxilla (identity-initialised so laterality can NOT flip).
     CT-frame motion per frame  =  W . M_t . W^-1
  4. Auto-detects landmarks on the CT mandible: condyles (L/R), incisal point.
  5. Optional roll-levelling from facial mirror-symmetry of the skull.
  6. Depth-shaded "relief" rendering of each bone (front / side / top views),
     z-composited, mandible drawn translucent so upper teeth show through.
  7. Draws growing traces (incisal, both condyles) + the inter-condylar
     (hinge) axis line, reference axes through the dental midline / orbits.
  8. Concatenates all phases and writes an MP4.

LATERALITY: DICOM LPS convention is assumed -> +X = patient LEFT, -X = RIGHT.
Always sanity-check sides against a known landmark (here: the higher canine).

DEPENDENCIES
  numpy, scipy, trimesh, open3d, imageio, imageio-ffmpeg
      pip install numpy scipy trimesh open3d imageio imageio-ffmpeg
"""




import argparse, time, xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage as ndi
import trimesh, imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
 
 
 
 
# ---------- colours (deliberately avoid skull-grey / mandible-blue) ----------
C_INC, C_R, C_L, C_AX = "#e8000b", "#ff8c00", "#e0119d", "#8e2de2"
SKULL_RGB, MAND_RGB   = (0.82, 0.80, 0.75), (0.20, 0.50, 0.88)
PXMM = 4.0                      # pixels per mm – SAME on both axes => isotropic
 
# =============================================================================
# 1. ModJaw parsing + rigid motion
# =============================================================================
def parse_jawmotion(path):
    """Return {movement_name: {'mark_1':(N,3),'mark_2':(N,3),'mark_3':(N,3)}}."""
    root = ET.parse(path).getroot()
    out = {}
    for mv in root.find("movements").findall("movement"):
        name = mv.find("type").text
        tracks = {}
        for tk in mv.findall(".//track"):
            tid = tk.get("id") or (tk.find("id").text if tk.find("id") is not None else None)
            q = tk.find("quants")
            pts = np.array([[float(c.find("x").text), float(c.find("y").text),
                             float(c.find("z").text)] for c in q.findall("quant")])
            tracks[tid] = pts
        out[name] = tracks
    return out
 
 
 
 
def kabsch(P, Q):
    """Rigid transform mapping P onto Q:  Q ~= R @ P.T + t."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cQ - R @ cP
 
 
 
 
def motion_transforms(tracks, W):
    """Per-frame 4x4 CT-frame transforms, relative to frame 0.
       CT motion = W @ (ModJaw motion) @ W^-1."""
    M = np.stack([tracks["mark_1"], tracks["mark_2"], tracks["mark_3"]], 1)  # (N,3,3)
    ref = M[0]
    Winv = np.linalg.inv(W)
    out = []
    for t in range(len(M)):
        R, tr = kabsch(ref, M[t])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tr
        out.append(W @ T @ Winv)
    return np.array(out)
 
 
 
# =============================================================================
# 2. Frame alignment  (ModJaw/scan frame  ->  CT frame)
# =============================================================================
def align_scan_to_ct(scan_stl, skull_stl, log=print):
    """ICP of the intra-oral upper scan onto the CT maxilla dental region.
       Identity-initialised: axes of both frames are anatomical & ~parallel,
       so the rotation stays small and laterality can never flip."""
    import open3d as o3d
    scan = np.asarray(trimesh.load(scan_stl, process=False).vertices)
    Vx = np.asarray(trimesh.load(skull_stl, process=False).vertices)
    zmin = Vx[:, 2].min()
    ct = Vx[(Vx[:, 1] < Vx[:, 1].min() + 63) & (Vx[:, 2] < zmin + 22)]  # dental arch block
    pc = lambda P: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    src, tgt = pc(scan).voxel_down_sample(0.6), pc(ct).voxel_down_sample(0.6)
    for p in (src, tgt):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=30))
    T0 = np.eye(4)
    T0[:3, 3] = np.asarray(tgt.points).mean(0) - np.asarray(src.points).mean(0)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, 3.0, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, 1.2, reg.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    W = np.asarray(reg.transformation)
    ang = np.degrees(np.arccos(np.clip((np.trace(W[:3, :3]) - 1) / 2, -1, 1)))
    log(f"[align] scan->CT  fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.3f} "
        f"rot={ang:.1f}deg  (small = OK, no flip)")
    return W
 
 
 
# =============================================================================
# 3. Landmarks + levelling  (auto from the geometry)
# =============================================================================
def detect_condyles(Vm):
    """Return (right, left) condyle centres. DICOM: -X = patient RIGHT."""
    up = Vm[Vm[:, 2] > Vm[:, 2].min() + (np.ptp(Vm[:, 2]) * 0.72)]  # upper rami
    def head(side):
        P = up[up[:, 0] < -8] if side == "R" else up[up[:, 0] > 8]
        post = P[P[:, 1] > np.median(P[:, 1])]          # condyle = posterior peak
        return post[post[:, 2] > post[:, 2].max() - 5].mean(0)
    return head("R"), head("L")
 
 
def detect_incisal(Vm, midx):
    a = Vm[np.abs(Vm[:, 0] - midx) < 10]
    f = a[a[:, 1] < np.percentile(a[:, 1], 4)]
    return f[f[:, 2] > np.percentile(f[:, 2], 65)].mean(0)
 
 
def detect_dental_midline(pts):
    """X of the UPPER central-incisor contact = median X of the most ANTERIOR
       points (the incisal edge of the two centrals straddles the midline).
       Most accurate on a clean intra-oral scan; usable on the maxilla bone too."""
    front = pts[pts[:, 1] < np.percentile(pts[:, 1], 6)]   # anterior-most = central incisors
    return float(np.median(front[:, 0]))
 
 
 
def facial_roll(maxP):
    """Roll (deg about Y) that maximises left-right mirror symmetry of the face.
       ~0 means the CT is already level."""
    z = maxP[:, 2]
    face = maxP[(z > np.percentile(z, 30)) & (z < np.percentile(z, 92))]
    def score(th):
        c, s = np.cos(th), np.sin(th)
        X = face[:, 0] * c + face[:, 2] * s
        Z = -face[:, 0] * s + face[:, 2] * c
        X = X - np.median(X)
        gx = np.round(X * 2).astype(int); gz = np.round((Z - Z.min()) * 2).astype(int)
        W = np.ptp(gx) + 1; H = gz.max() + 1
        img = np.zeros((H, W), bool); img[gz, gx - gx.min()] = True
        cc = -gx.min(); half = min(cc, W - 1 - cc)
        if half < 10: return 0
        L, Rr = img[:, cc - half:cc], img[:, cc + 1:cc + 1 + half][:, ::-1]
        return (L & Rr).sum() / max((L | Rr).sum(), 1)
    ths = np.deg2rad(np.arange(-10, 10.1, 0.3))
    best = ths[int(np.argmax([score(t) for t in ths]))]
    return float(np.degrees(best))
 
 
 
# =============================================================================
# 4. Depth-shaded relief renderer
# =============================================================================
def relief(P, h0, h1, dax, closer_small, bounds, H, W, base, smooth=1.5, clean=True):
    """Project points, z-buffer the nearest surface, Lambert-shade the relief."""
    xmn, xmx, ymn, ymx = bounds
    gx = ((P[:, h0] - xmn) * PXMM).astype(int)
    gy = ((ymx - P[:, h1]) * PXMM).astype(int)
    dep = P[:, dax].copy(); dep = dep if closer_small else -dep
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    gx, gy, dep = gx[ok], gy[ok], dep[ok]
    D = np.full((H, W), np.inf); np.minimum.at(D, (gy, gx), dep)
    mask = np.isfinite(D)
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2))
    if clean:                                   # kill thin sampling "drips"
        mask = ndi.binary_opening(mask, iterations=1)
        lb, n = ndi.label(mask)
        if n > 1:
            sz = np.bincount(lb.ravel()); sz[0] = 0; mask = lb == sz.argmax()
    idx = ndi.distance_transform_edt(~np.isfinite(D), return_indices=True,
                                     return_distances=False)
    Df = ndi.gaussian_filter(D[tuple(idx)], smooth)
    gyv, gxv = np.gradient(-Df)                  # height field = -depth
    nrm = np.stack([-gxv, -gyv, np.ones_like(Df) * 6.0], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L = np.array([-0.45, -0.55, 0.72]); L /= np.linalg.norm(L)
    val = 0.30 + 0.80 * np.clip(nrm @ L, 0, 1)
    rgb = np.clip(np.stack([base[c] * val for c in range(3)], -1), 0, 1)
    return rgb, mask, np.where(mask, Df, np.inf)
 
 
# view = (h0, h1, depth_axis, closer_is_smaller, xlabel, draw_skull)
VIEWS = {
    "front": (0, 2, 1, True,  "patient RIGHT <- X -> LEFT", True),
    "side":  (1, 2, 0, True,  "anterior <- Y -> posterior", True),
    "top":   (0, 1, 2, False, "patient RIGHT <- X -> LEFT", False),
}
 
 
ARCH_RGB = (0.94, 0.90, 0.78)      # ivory for dental-arch scans/designs
 
 
UPPER_TOOTH_RGB = (0.95, 0.93, 0.84)      # ivory  (maxillary teeth)
LOWER_TOOTH_RGB = (0.93, 0.80, 0.52)      # warm gold (mandibular teeth)
 
 
def render_phase(title, Ts, views, geom, Rdisp=np.eye(3), nframes=44, occl=None,
                 upperT=None, lowerT=None, log=None):
    """occl = (skull_kdtree_CT, threshold_mm) or None -> RED contact/interference dots.
       upperT = maxillary-teeth points (CT, static);  lowerT = mandibular-teeth points
       (CT rest, ride with the mandible).  When given, teeth are drawn as bright
       coloured layers over faint bone so the two rows and their closure are clear."""
    maxP0, manP, incisal, condR, condL, facex, dentx, orbz = geom
    teeth = (upperT is not None) or (lowerT is not None)
    centerx = dentx                               # centre the view on the dental midline
    disp = lambda P: np.atleast_2d(P) @ Rdisp.T
    maxP = disp(maxP0)
    upS = disp(upperT) if upperT is not None else None
    px = lambda x, y, b: ((np.asarray(x) - b[0]) * PXMM, (b[3] - np.asarray(y)) * PXMM)
    ap_ct = lambda T, P: (T[:3, :3] @ np.atleast_2d(P).T).T + T[:3, 3]   # CT frame
    ap = lambda T, P: disp(ap_ct(T, P))                                 # display frame
    idxs = np.linspace(0, len(Ts) - 1, nframes).astype(int)
    posed = [ap(Ts[i], manP[::3]) for i in idxs]
    bnd, dims = {}, {}
    for v in views:
        h0, h1, _, _, _, sk = VIEWS[v]
        base = np.vstack(([maxP[::3]] if sk else []) + posed)[:, [h0, h1]]
        xmn, xmx = base[:, 0].min() - 9, base[:, 0].max() + 9
        ymn, ymx = base[:, 1].min() - 11, base[:, 1].max() + 11
        if h0 == 0:                               # centre horizontally on dental midline
            hw = max(centerx - xmn, xmx - centerx); xmn, xmx = centerx - hw, centerx + hw
        bnd[v] = (xmn, xmx, ymn, ymx)
        dims[v] = (int((ymx - ymn) * PXMM) + 1, int((xmx - xmn) * PXMM) + 1)
    bone_a = (0.35, 0.18) if teeth else (0.62, 0.33)         # faint mandible when teeth shown
    skull_rgb = tuple(0.55 + 0.45 * c for c in SKULL_RGB) if teeth else SKULL_RGB
    skull = {v: relief(maxP, *VIEWS[v][:4], bnd[v], *dims[v], skull_rgb, 1.7)
             for v in views if VIEWS[v][5]}
    # static maxillary arch: shaded layer in the skull views, FIXED REFERENCE OUTLINE
    # in the top view (there is no skull there, so without it the gothic-arch view has
    # nothing static to judge the excursion against and the mandible looks motionless)
    upper = {v: relief(upS, *VIEWS[v][:4], bnd[v], *dims[v], UPPER_TOOTH_RGB, 1.0)
             for v in views} if upS is not None else {}
    uedge = {}
    for v in upper:
        if not VIEWS[v][5]:
            mU = upper[v][1]
            uedge[v] = mU ^ ndi.binary_erosion(mU, iterations=2)
    # occlusal-interference query points: the mandibular crowns (what can actually clash)
    src = lowerT if lowerT is not None else manP[manP[:, 1] < np.percentile(manP[:, 1], 60)]
    dental_pts = np.asarray(src)
    if len(dental_pts) > 20000:                            # subsample for speed
        dental_pts = dental_pts[np.random.RandomState(0).choice(len(dental_pts), 20000, False)]
    tr = {v: {"inc": [], "cR": [], "cL": []} for v in views}
    frames = []
    t_start = time.time()
    for n, i in enumerate(idxs):
        Mn_ct = ap_ct(Ts[i], manP); Mn = disp(Mn_ct)
        Lo = ap(Ts[i], lowerT) if lowerT is not None else None
        contact = None
        if occl is not None:                                # occlusal interference points
            tree, thr = occl
            q = ap_ct(Ts[i], dental_pts)
            try:                       # bounded + threaded query: ~10x faster than k=1
                d, _ = tree.query(q, k=1, distance_upper_bound=thr, workers=-1)
            except TypeError:                                  # older scipy
                d, _ = tree.query(q, k=1, distance_upper_bound=thr)
            hit = q[np.isfinite(d)]
            if len(hit) > 1500:
                hit = hit[np.random.RandomState(1).choice(len(hit), 1500, False)]
            contact = disp(hit) if len(hit) else None
        it, cr, cl = ap(Ts[i], incisal)[0], ap(Ts[i], condR)[0], ap(Ts[i], condL)[0]
        fig, axs = plt.subplots(1, len(views), figsize=(13, 6.6), dpi=80)
        axs = np.atleast_1d(axs)
        for a, v in zip(axs, views):
            h0, h1, dax, cs, xl, sk = VIEWS[v]; H, W = dims[v]
            rgbM, mM, DM = relief(Mn, h0, h1, dax, cs, bnd[v], H, W, MAND_RGB, 1.3)
            out = np.ones((H, W, 3))
            if sk:
                rgbS, mS, DS = skull[v]; out[mS] = rgbS[mS]
                aM = np.where(DM <= DS, bone_a[0], bone_a[1]) * mM    # translucent mandible
            else:                       # top view: fixed upper arch as the tracing plate
                mS = np.zeros_like(mM)
                if v in upper:
                    rgbU, mU, _ = upper[v]
                    out[mU] = 1 - 0.40 * (1 - rgbU[mU])       # pale, stays underneath
                    mS = mU.copy()
                aM = mM * 0.95
            for c in range(3):
                out[:, :, c] = out[:, :, c] * (1 - aM) + rgbM[:, :, c] * aM
            # --- bright tooth layers on top (nearest-depth between upper & lower) ---
            if teeth:
                tD = np.full((H, W), np.inf)
                def paint_tooth(layer):
                    rgb, m, d = layer
                    n = m & (d < tD)
                    out[n] = rgb[n]; tD[n] = d[n]; mS2[n] = True
                mS2 = np.zeros((H, W), bool)
                if sk and v in upper: paint_tooth(upper[v])
                if Lo is not None:
                    paint_tooth(relief(Lo, h0, h1, dax, cs, bnd[v], H, W, LOWER_TOOTH_RGB, 1.0))
                mS = mS | mS2
            if v in uedge:              # fixed upper-arch outline always on top (top view)
                e = uedge[v]; out[e] = (0.52, 0.42, 0.12); mS = mS | e
            a.imshow(np.dstack([out, (mS | mM).astype(float)]),
                     aspect="equal", interpolation="bilinear")
            a.set_title(v.upper() + (" (gothic arch)" if v == "top" else ""),
                        fontsize=12, weight="bold")
            a.set_xlabel(xl, fontsize=9); a.set_xticks([]); a.set_yticks([])
            if contact is not None:                           # occlusion interference overlay
                cxp, cyp = px(contact[:, h0], contact[:, h1], bnd[v])
                a.scatter(cxp, cyp, s=7, c="#ff1111", alpha=0.45,
                          edgecolors="none", zorder=5)
            if h0 == 0:            # X is horizontal here -> fixed dental-midline reference
                a.axvline(px(dentx, 0, bnd[v])[0], color="#15a015", lw=1.4, ls="-.", alpha=.85)
            if v == "front":
                a.axhline(px(0, orbz, bnd[v])[1], color="#15a015", lw=1.0, ls=":", alpha=.55)
            for lm, k in [(it, "inc"), (cr, "cR"), (cl, "cL")]:
                tr[v][k].append(px(lm[h0], lm[h1], bnd[v]))
            a.plot(*zip(*tr[v]["inc"]), "-", color=C_INC, lw=2.4)
            a.plot(*tr[v]["inc"][-1], "o", color=C_INC, ms=6)
            if v in ("side", "top"):
                a.plot(*zip(*tr[v]["cR"]), "-", color=C_R, lw=2.2)
                a.plot(*zip(*tr[v]["cL"]), "-", color=C_L, lw=2.2)
            pr, pl = tr[v]["cR"][-1], tr[v]["cL"][-1]             # hinge axis line
            a.plot([pr[0], pl[0]], [pr[1], pl[1]], "-", color=C_AX, lw=2.0, alpha=.9)
            a.plot(*pr, "o", color=C_R, ms=6); a.plot(*pl, "o", color=C_L, ms=6)
            a.set_xlim(0, W); a.set_ylim(H, 0)
        legend = (title + "   ivory = UPPER teeth | gold = LOWER teeth" if teeth
                  else title + "   red=incisal | orange=R condyle | magenta=L condyle")
        legend += ("   | RED dots = contact/interference" if occl is not None else "")
        if uedge: legend += "   | dark outline = FIXED upper arch"
        fig.suptitle(legend, fontsize=11, weight="bold")
        fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.07, wspace=.06)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frames.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
                      .reshape(h, w, 4)[:, :, :3].copy())
        plt.close(fig)
        if log and (n + 1) % 8 == 0:                       # progress + ETA, not a hang
            el = time.time() - t_start
            log(f"      {title[:22]:<22} {n + 1:3d}/{nframes} frames "
                f"({el / (n + 1):.2f} s/frame, ~{el / (n + 1) * (nframes - n - 1):.0f}s left)")
    return frames
 
 
# =============================================================================
# 5. Pipeline (callable from CLI or GUI)
# =============================================================================
def run_pipeline(mandible, skull, jawmotion, scan=None, out="jaw_motion.mp4",
                 level=False, dentx=None, fps=13, occlusion=False, occ_thresh=0.6,
                 color_teeth=True, nframes=44, log=print):
    """Build the MP4. `log` is any function taking a string (print, or a GUI sink).
       The skull and mandible STLs are assumed to CARRY THEIR TEETH (current or new
       restorations modelled in).  occlusion=True highlights, per frame, where the
       mandibular dentition comes within occ_thresh mm of the maxilla = contact /
       interference on the recorded path (a 'where would it clash' map, NOT a
       re-simulation of the new functional movement)."""
    log("[1/5] frame alignment (ModJaw -> CT)")
    W = align_scan_to_ct(scan, skull, log=log) if scan else np.eye(4)
    if not scan:
        log("[warn] no scan: assuming CT bones already in the ModJaw frame (W=I)")
    to_ct = lambda P: (W[:3, :3] @ np.asarray(P).T).T + W[:3, 3]
 
    log("[2/5] sampling surfaces")
    man_mesh = trimesh.load(mandible, process=False)
    sku_mesh = trimesh.load(skull, process=False)
    manP = np.asarray(man_mesh.sample(140000))
    Vx = np.asarray(sku_mesh.vertices)
    keep = (Vx[:, 2] < 95) & (Vx[:, 1] < Vx[:, 1].min() + 111) & (np.abs(Vx[:, 0]) < 80)
    sku_face = sku_mesh.submesh([keep[sku_mesh.faces].all(1)], append=True)
    maxP = np.asarray(sku_face.sample(380000))
    low = sku_face.submesh([sku_face.vertices[sku_face.faces].mean(1)[:, 2] < 8],
                           append=True)
    if len(low.vertices):
        maxP = np.vstack([maxP, np.asarray(low.sample(120000))])   # dense teeth
 
    Rdisp = np.eye(3)
    if level:
        roll = np.deg2rad(facial_roll(maxP))
        c, s = np.cos(roll), np.sin(roll)
        Rdisp = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        log(f"[level] facial-symmetry roll = {np.degrees(roll):.2f} deg")
 
    log("[3/5] landmarks")
    Vm = np.asarray(man_mesh.vertices)
    condR, condL = detect_condyles(Vm)            # DICOM: -X = patient RIGHT
    maxP_lev = maxP @ Rdisp.T
    facex = float(np.median(maxP_lev[:, 0]))
    incisal = detect_incisal(Vm, float(np.median(maxP[:, 0])))
    # dental midline: best from the clean intra-oral scan if available, else the bone
    if dentx is not None:
        dx = float(dentx)
    elif scan:
        S = to_ct(trimesh.load(scan, process=False, force="mesh").vertices) @ Rdisp.T
        dx = detect_dental_midline(S)
    else:
        dx = detect_dental_midline(maxP_lev)
    log(f"[landmarks] dental midline X = {dx:.2f}  (facial midline X = {facex:.2f})")
    orbz = float(np.percentile(maxP_lev[:, 2], 80))
    geom = (maxP, manP, incisal, condR, condL, facex, dx, orbz)
 
    # auto-segment the two tooth rows from the bones (colour display + occlusion test)
    occ_up = np.percentile(maxP[maxP[:, 1] < 58, 2], 2)           # upper occlusal level
    upperT = maxP[(maxP[:, 1] < 58) & (maxP[:, 2] < occ_up + 12)]  # maxillary crowns
    ybody = np.percentile(manP[:, 1], 60)
    topz = np.percentile(manP[manP[:, 1] < ybody, 2], 88)         # mandibular ridge
    lowerT = manP[(manP[:, 1] < ybody) & (manP[:, 2] > topz - 12)]  # mandibular crowns
    log(f"[teeth] tooth rows: upper {len(upperT)} pts, lower {len(lowerT)} pts"
        + ("" if color_teeth else "  (colour display off)"))
 
    occl = None
    if occlusion:
        from scipy.spatial import cKDTree
        # tree on the MAXILLARY CROWNS only: interference is tooth-to-tooth, and a
        # 120k-point tree queries several times faster than the whole 500k skull
        occl = (cKDTree(upperT), float(occ_thresh))
        log(f"[occlusion] interference highlight ON (threshold {occ_thresh} mm)")
    if not color_teeth:
        upperT = lowerT = None
 
    log("[4/5] parsing motions")
    jm = parse_jawmotion(jawmotion)
    def match(anykeys):
        for name in jm:
            if any(k in name.lower() for k in anykeys):
                return jm[name]
        return None
    OPEN  = ["open", "відкрив"]
    PROTR = ["protr", "протру"]
    LAT_R = ["right latero", "правор"]
    LAT_L = ["left latero", "лівор"]
    pose = lambda tr: motion_transforms(tr, W)
    phases = []
    if match(OPEN)  is not None: phases.append(("OPENING",    pose(match(OPEN)),  ["front"]))
    if match(PROTR) is not None: phases.append(("PROTRUSION", pose(match(PROTR)), ["side", "front"]))
    r, l = match(LAT_R), match(LAT_L)
    if r is not None and l is not None:
        phases.append(("LATEROTRUSION (right -> left)",
                       np.concatenate([pose(r), pose(l)], 0), ["front", "top"]))
    if not phases:
        raise RuntimeError("No recognised movements in the .jawmotion file.")
 
    log(f"[5/5] rendering {len(phases) * nframes} frames"
        + ("  (+occlusion)" if occl else "") + " - progress printed every 8 frames")
    allframes = []
    for label, T, vs in phases:
        fr = render_phase(label, T, vs, geom, Rdisp=Rdisp, occl=occl, nframes=nframes,
                          upperT=upperT, lowerT=lowerT, log=log)
        allframes += fr + [fr[-1]] * 6
        log(f"   rendered {label}: {len(fr)} frames")
    H = max(f.shape[0] for f in allframes); Wd = max(f.shape[1] for f in allframes)
    pad = []
    for f in allframes:
        c = np.ones((H, Wd, 3), np.uint8) * 255
        y, x = (H - f.shape[0]) // 2, (Wd - f.shape[1]) // 2
        c[y:y + f.shape[0], x:x + f.shape[1]] = f; pad.append(c)
    out = write_video(pad, out, fps, log)
    log(f"[done] wrote {out}")
    return out
 
 
 
def write_video(frames, out, fps, log=print):
    """Robust MP4 writing. Forces the FFMPEG backend; if imageio-ffmpeg is
       missing it falls back to an animated GIF with a clear message."""
    import os
    if os.path.splitext(out)[1] == "":          # no extension typed
        out += ".mp4"
    ext = os.path.splitext(out)[1].lower()
    if ext in (".mp4", ".mov", ".m4v", ".avi", ".webm"):
        try:
            w = imageio.get_writer(out, format="FFMPEG", fps=int(fps),
                                   codec="libx264", quality=8, macro_block_size=None)
            for f in frames:
                w.append_data(f)
            w.close()
            return out
        except Exception as e:
            log(f"[warn] ffmpeg writer unavailable ({e}).")
            log("[warn] install it with:  pip install imageio-ffmpeg")
            log("[warn] falling back to GIF for now.")
            out = os.path.splitext(out)[0] + ".gif"
    imageio.mimsave(out, frames, duration=1.0 / float(fps), loop=0)   # GIF / other
    return out
 
 
 
# =============================================================================
# 6. Console entry point
# =============================================================================
def cli():
    ap = argparse.ArgumentParser(description="Animate mandible vs skull from ModJaw.")
    ap.add_argument("--mandible", required=True)
    ap.add_argument("--skull", required=True)
    ap.add_argument("--jawmotion", required=True)
    ap.add_argument("--scan", default=None, help="intra-oral upper scan (frame align)")
    ap.add_argument("--out", default="jaw_motion.mp4")
    ap.add_argument("--dentx", type=float, default=None,
                    help="dental-midline X in CT mm (default: auto-detect)")
    ap.add_argument("--fps", type=int, default=13)
    ap.add_argument("--level", action="store_true",
                    help="auto-level skull roll from facial symmetry")
    ap.add_argument("--occlusion", action="store_true",
                    help="highlight tooth contact/interference along the path")
    ap.add_argument("--occ-thresh", type=float, default=1.5,
                    help="contact distance threshold in mm (default 1.5)")
    ap.add_argument("--no-teeth", dest="teeth", action="store_false",
                    help="do not colour the tooth rows (plain bone)")
    ap.add_argument("--nframes", type=int, default=44,
                    help="frames rendered per movement (default 44; 24 = quick preview)")
    ap.add_argument("--fast", action="store_true",
                    help="quick preview: 24 frames per movement")
    a = ap.parse_args()
    run_pipeline(a.mandible, a.skull, a.jawmotion, a.scan, a.out,
                 a.level, a.dentx, a.fps, a.occlusion, a.occ_thresh, a.teeth,
                 24 if a.fast else a.nframes)
 
 
# =============================================================================
# 7. GUI entry point  (tkinter – standard library)
# =============================================================================
def gui():
    import threading, queue, traceback
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
 
    root = tk.Tk()
    root.title("MODJAW2CT: Jaw motion -> MP4")
    root.geometry("720x560")
    pad = dict(padx=6, pady=4)
    frm = ttk.Frame(root); frm.pack(fill="x", **pad)
 
    vars_ = {}
    def row(r, label, key, kind="open", types=(("STL", "*.stl"),)):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
        v = tk.StringVar(); vars_[key] = v
        ttk.Entry(frm, textvariable=v, width=64).grid(row=r, column=1, **pad)
        def browse():
            if kind == "save":
                p = filedialog.asksaveasfilename(defaultextension=".mp4",
                        filetypes=[("MP4", "*.mp4")])
            else:
                p = filedialog.askopenfilename(filetypes=list(types) + [("All", "*.*")])
            if p: v.set(p)
        ttk.Button(frm, text="...", width=3, command=browse).grid(row=r, column=2, **pad)
 
    row(0, "Mandible STL *  (with lower teeth)", "mandible")
    row(1, "Skull / maxilla STL *  (with upper teeth)", "skull")
    row(2, "ModJaw .jawmotion *",   "jawmotion", types=(("jawmotion/XML", "*.jawmotion *.xml"),))
    row(3, "UPPER-arch scan STL (optionally)",   "scan")
    row(4, "Output MP4",            "out", kind="save")
    vars_["out"].set("jaw_motion.mp4")
 
    opt = ttk.Frame(root); opt.pack(fill="x", **pad)
    level_v = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt, text="Auto-level", variable=level_v).grid(row=0, column=0, sticky="w", **pad)
    teeth_v = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt, text="Colour teeth", variable=teeth_v).grid(row=0, column=1, sticky="w", **pad)
    occ_v = tk.BooleanVar(value=False)
    ttk.Checkbutton(opt, text="Occlusal interferences",
                    variable=occ_v).grid(row=0, column=2, sticky="w", **pad)
    ttk.Label(opt, text="fps").grid(row=0, column=3, sticky="e", **pad)
    fps_v = tk.StringVar(value="13")
    ttk.Entry(opt, textvariable=fps_v, width=4).grid(row=0, column=4, **pad)
    ttk.Label(opt, text="dentX").grid(row=0, column=5, sticky="e", **pad)
    dentx_v = tk.StringVar(value="")
    ttk.Entry(opt, textvariable=dentx_v, width=6).grid(row=0, column=6, **pad)
    ttk.Label(opt, text="frames/movement").grid(row=0, column=7, sticky="e", **pad)
    nf_v = tk.StringVar(value="44")
    ttk.Entry(opt, textvariable=nf_v, width=4).grid(row=0, column=8, **pad)
 
    logbox = tk.Text(root, height=16, wrap="word"); logbox.pack(fill="both", expand=True, **pad)
    q = queue.Queue()
    def sink(msg): q.put(str(msg))
    def drain():
        while not q.empty():
            logbox.insert("end", q.get() + "\n"); logbox.see("end")
        root.after(120, drain)
    drain()
 
    btn = ttk.Button(root, text="Generate MP4")
    btn.pack(**pad)
 
    def worker():
        try:
            dx = float(dentx_v.get()) if dentx_v.get().strip() else None
            out = run_pipeline(vars_["mandible"].get(), vars_["skull"].get(),
                               vars_["jawmotion"].get(),
                               vars_["scan"].get() or None, vars_["out"].get(),
                               level_v.get(), dx, int(fps_v.get()),
                               occ_v.get(), 1.5, teeth_v.get(),
                               int(nf_v.get() or 44), log=sink)
            q.put("✓ FINISHED: " + out)
            root.after(0, lambda: messagebox.showinfo("Done", "Wrote:\n" + out))
        except Exception as e:
            q.put("ERROR: " + str(e)); q.put(traceback.format_exc())
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            root.after(0, lambda: btn.config(state="normal"))
 
    def start():
        for k in ("mandible", "skull", "jawmotion"):
            if not vars_[k].get():
                messagebox.showwarning("Missing", "Mandible, skull and jawmotion are required.")
                return
        btn.config(state="disabled")
        logbox.delete("1.0", "end")
        threading.Thread(target=worker, daemon=True).start()
    btn.config(command=start)
    root.mainloop()
 
 
 
 
if __name__ == "__main__":
    import sys
    # no console args  ->  launch the file-picker GUI;  otherwise run CLI
    if len(sys.argv) == 1 or "--gui" in sys.argv:
        try:
            gui()
        except ImportError:
            print("tkinter is not installed. Install it (e.g. 'sudo apt install "
                  "python3-tk'), or use the command line:\n"
                  "  python3 MODJAW2CT.py --mandible M.stl --skull S.stl "
                  "--jawmotion J.jawmotion --scan scan.stl --out out.mp4")
    else:
        cli()


