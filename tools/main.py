#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, math, glob, shutil
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from plyfile import PlyData
except Exception:
    PlyData = None

# ---------- FIXED PATHS ----------
INPUT_ROOT  = "/nas/data/araake/scenefun3d/original"
OUTPUT_ROOT = "/nas/data/araake/scenefun3d/openscene/data"

SCANNET_2D = os.path.join(OUTPUT_ROOT, "scannet_2d")
SCANNET_3D = os.path.join(OUTPUT_ROOT, "scannet_3d")
SPLITS_DIR = os.path.join(OUTPUT_ROOT, "splits")
META_DIR   = os.path.join(OUTPUT_ROOT, "meta")

AFFORDANCE_TO_ID = {
    "rotate": 0, "key_press": 1, "tip_push": 2, "hook_pull": 3, "pinch_pull": 4,
    "hook_turn": 5, "foot_push": 6, "plug_in": 7, "unplug": 8,
}
IGNORE_ID = 255

# ---------- LOG ----------
def info(msg: str):  print(f"[INFO] {msg}")
def warn(msg: str):  print(f"[WARN] {msg}")
def error(msg: str): print(f"[ERROR]--- {msg}")
def done(msg: str = ""): print(f"[DONE] {msg}" if msg else "[DONE]")
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# ---------- SMALL UTILS ----------
def find_repo_train_list() -> Optional[str]:
    cand = "benchmark_file_lists/train_scenes.txt"
    if os.path.isfile(cand): return os.path.abspath(cand)
    here = os.path.abspath(os.getcwd())
    for _ in range(4):
        p = os.path.join(here, "benchmark_file_lists", "train_scenes.txt")
        if os.path.isfile(p): return p
        here = os.path.dirname(here)
    return None

def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(rvec)
    if th < 1e-12: return np.eye(3, dtype=np.float64)
    k = rvec / th; kx, ky, kz = k
    K = np.array([[0, -kz, ky],[kz, 0, -kx],[-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)

def load_traj_w2c(traj_path: str):
    out = []
    with open(traj_path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 7: continue
            parts = parts[-7:]
            ts, rx, ry, rz, tx, ty, tz = parts
            R = rodrigues_to_R(np.array([float(rx), float(ry), float(rz)], dtype=np.float64))
            t = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)
            out.append((ts, R, t))
    return out

def w2c_to_c2w(R: np.ndarray, t: np.ndarray):
    Rt = R.T
    return Rt, -Rt @ t

def write_pose_txt(path: str, R: np.ndarray, t: np.ndarray):
    M = np.eye(4, dtype=np.float64); M[:3,:3] = R; M[:3,3] = t
    with open(path, "w") as f:
        for r in range(4):
            f.write(" ".join(f"{x:.6f}" for x in M[r]) + ("\n" if r < 3 else ""))

def list_keys_without_ext(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*")))
    out = []
    for p in files:
        if os.path.isdir(p): continue
        k, _ = os.path.splitext(os.path.basename(p))
        out.append(k)
    return out

def choose_common_keys(k_lists: List[List[str]]) -> List[str]:
    if not k_lists: return []
    base = k_lists[0]; sets = [set(ks) for ks in k_lists]
    return [k for k in base if all(k in s for s in sets)]

# ---- intrinsics helpers ----
def _parse_six_numbers_from_text(text: str):
    toks = [t for t in text.replace(",", " ").split() if t]
    vals = []
    for t in toks:
        try: vals.append(float(t))
        except Exception: pass
        if len(vals) >= 6: break
    if len(vals) < 6: return None
    W, H, fx, fy, cx, cy = vals[:6]
    return int(W), int(H), float(fx), float(fy), float(cx), float(cy)

def _load_intrinsic_file(path: str):
    try:
        if path.endswith(".npy"):
            arr = np.load(path).reshape(-1)
            if arr.size >= 6:
                W,H,fx,fy,cx,cy = arr[:6]; return int(W),int(H),float(fx),float(fy),float(cx),float(cy)
        else:
            with open(path, "r") as f: text = f.read()
            return _parse_six_numbers_from_text(text)
    except Exception:
        return None
    return None

def load_intrinsics_per_key_fuzzy(intrin_dir: str, key: str):
    pats = [
        os.path.join(intrin_dir, key + ".txt"),
        os.path.join(intrin_dir, key + ".npy"),
        os.path.join(intrin_dir, key + ".*"),
        os.path.join(intrin_dir, key + "*"),
    ]
    cand = []
    for pt in pats: cand += sorted(glob.glob(pt))
    for p in cand:
        if os.path.isdir(p): continue
        v = _load_intrinsic_file(p)
        if v is not None: return v
    return None

def copy_with_progress(src_paths, dst_paths, label: str, steps: int = 4):
    n = len(src_paths)
    if n == 0:
        info(f"{label}: 0 files"); return
    checkpoints = {int(n*s/steps) for s in range(steps+1)}
    for i,(s,d) in enumerate(zip(src_paths, dst_paths)):
        ensure_dir(os.path.dirname(d))
        try: shutil.copy2(s, d)
        except Exception as e: error(f"copy failed: {s} -> {d} ({e})")
        if i in checkpoints:
            pct = int(100 * (i / max(1, n-1))); info(f"{label}: {pct}%")

def try_read_ply_xyzrgb(ply_path: str):
    if not os.path.isfile(ply_path):
        error(f"missing file: {ply_path}"); return None, None
    if PlyData is None:
        error("plyfile not available; cannot read PLY"); return None, None
    try:
        ply = PlyData.read(ply_path); el = ply['vertex']
        coords = np.stack([np.asarray(el['x'], np.float32),
                           np.asarray(el['y'], np.float32),
                           np.asarray(el['z'], np.float32)], axis=1)
        rgb = None
        names = el.data.dtype.names
        if all(c in names for c in ("red","green","blue")):
            rgb = np.stack([np.asarray(el['red'], np.float32),
                            np.asarray(el['green'], np.float32),
                            np.asarray(el['blue'], np.float32)], axis=1)
        return coords, rgb
    except Exception as e:
        error(f"PLY read failed: {ply_path} ({e})"); return None, None

def build_labels_from_annotations(N_full: int, anno_path: str) -> np.ndarray:
    """labels over the ORIGINAL (unmasked) point count."""
    labels = np.full((N_full,), IGNORE_ID, dtype=np.int32)
    if not os.path.isfile(anno_path):
        warn(f"annotations not found: {anno_path} (labels set to IGNORE)")
        return labels
    try:
        with open(anno_path, "r") as f: data = json.load(f)
        anns = data.get("annotations", data)
        for a in anns:
            name = (a.get("label","") or "").strip()
            idxs = a.get("indices", [])
            if isinstance(idxs, str):
                if os.path.isfile(idxs) and idxs.endswith(".npy"):
                    idxs = np.load(idxs).astype(np.int64)
                else:
                    try: idxs = [int(x) for x in idxs.strip().split()]
                    except Exception: idxs = []
            idxs = np.asarray(idxs, dtype=np.int64)
            # bound check against ORIGINAL length:
            if idxs.size and (idxs.max() >= N_full or idxs.min() < 0):
                warn(f"indices out of range in annotations (max={int(idxs.max())}, N={N_full}); clipping")
                idxs = idxs[(idxs >= 0) & (idxs < N_full)]
            if name.lower() in ("exclude","ignore"):
                labels[idxs] = IGNORE_ID
            else:
                cid = AFFORDANCE_TO_ID.get(name, IGNORE_ID)
                if cid == IGNORE_ID and name: warn(f"unknown affordance '{name}' -> IGNORE")
                labels[idxs] = cid
    except Exception as e:
        warn(f"failed to parse annotations: {e} (labels set to IGNORE)")
    return labels

def write_classes_tables():
    ensure_dir(META_DIR)
    with open(os.path.join(META_DIR, "classes.csv"), "w") as f:
        f.write("id,name\n")
        for name, cid in sorted(AFFORDANCE_TO_ID.items(), key=lambda kv: kv[1]):
            f.write(f"{cid},{name}\n")
        f.write(f"{IGNORE_ID},ignore\n")
    with open(os.path.join(META_DIR, "classes.tsv"), "w") as f:
        for name, cid in sorted(AFFORDANCE_TO_ID.items(), key=lambda kv: kv[1]):
            f.write(f"{cid}\t{name}\n")
        f.write(f"{IGNORE_ID}\tignore\n")

def read_train_visits(path: Optional[str]) -> set:
    if not path or not os.path.isfile(path):
        warn("train_scenes.txt not found; all visits will be treated as val"); return set()
    s = set()
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: s.add(int(ln.split()[0]))
            except Exception: pass
    return s

def enumerate_all_scenes():
    if not os.path.isdir(INPUT_ROOT):
        error(f"input root not found: {INPUT_ROOT}"); sys.exit(1)
    scenes = []
    for vname in sorted(os.listdir(INPUT_ROOT)):
        vpath = os.path.join(INPUT_ROOT, vname)
        if not (os.path.isdir(vpath) and vname.isdigit()): continue
        visit = int(vname)
        vids = [int(x) for x in sorted(os.listdir(vpath)) if x.isdigit() and os.path.isdir(os.path.join(vpath,x))]
        if vids: scenes.append((visit, vids))
    return scenes

def assign_scene_names(scenes):
    mapping = {}; gid = 0
    for visit, videos in scenes:
        for j, vid in enumerate(videos):
            mapping[(visit, vid)] = f"scene{gid:04d}_{j:02d}"; gid += 1
    return mapping

def split_for_visit(visit_id: int, train_visits: set) -> str:
    return "train" if visit_id in train_visits else "val"

def append_split_file(scene_name: str, split: str):
    ensure_dir(SPLITS_DIR)
    p = os.path.join(SPLITS_DIR, f"{split}.txt")
    prev = set()
    if os.path.isfile(p):
        with open(p, "r") as f:
            for ln in f: prev.add(ln.strip())
    if scene_name not in prev:
        with open(p, "a") as f: f.write(scene_name + "\n")

def update_scene_mapping(scene_name: str, split: str, visit: int, vid: int,
                         rgb_count: int, depth_count: int, pose_count: int,
                         intr: Optional[Tuple[int,int,float,float,float,float]],
                         world_frame: str = "laser_scan", status: str = "ok"):
    ensure_dir(META_DIR)
    p = os.path.join(META_DIR, "scene_mapping.csv")
    write_header = not os.path.isfile(p)
    with open(p, "a") as f:
        if write_header:
            f.write("scene_name,split,visit_id,video_id,rgb_count,depth_count,pose_count,width,height,fx,fy,cx,cy,world_frame,status\n")
        if intr is None:
            f.write(f"{scene_name},{split},{visit},{vid},{rgb_count},{depth_count},{pose_count},,,,,,,{world_frame},{status}\n")
        else:
            W,H,fx,fy,cx,cy = intr
            f.write(f"{scene_name},{split},{visit},{vid},{rgb_count},{depth_count},{pose_count},{W},{H},{fx},{fy},{cx},{cy},{world_frame},{status}\n")

# ---------- CORE ----------
def process_single_scene(visit_id: int, video_id: int, scene_name: str, split: str):
    vdir  = os.path.join(INPUT_ROOT, str(visit_id))
    vscan = os.path.join(vdir, f"{visit_id}_laser_scan.ply")
    vmask = os.path.join(vdir, f"{visit_id}_crop_mask.npy")
    vanno = os.path.join(vdir, f"{visit_id}_annotations.json")

    sdir = os.path.join(vdir, str(video_id))
    color_in = os.path.join(sdir, "hires_wide")
    depth_in = os.path.join(sdir, "hires_depth")
    intri_in = os.path.join(sdir, "hires_wide_intrinsics")
    traj_in  = os.path.join(sdir, "hires_poses.traj")

    scene_2d_root = os.path.join(SCANNET_2D, scene_name)
    color_out = os.path.join(scene_2d_root, "color")
    depth_out = os.path.join(scene_2d_root, "depth")
    pose_out  = os.path.join(scene_2d_root, "pose")
    intr_out_dir = os.path.join(scene_2d_root, "intrinsic")

    info(f"Scene {scene_name} (visit={visit_id}, video={video_id}) split={split}")

    # required
    missing = False
    for req in [color_in, depth_in, intri_in, traj_in]:
        if not os.path.exists(req):
            error(f"missing file/dir: {req}"); missing = True
    if missing:
        error(f"skip scene: visit={visit_id} video={video_id}")
        update_scene_mapping(scene_name, split, visit_id, video_id, 0, 0, 0, None, status="skipped_missing"); return

    # keys
    keys_color = list_keys_without_ext(color_in)
    keys_depth = list_keys_without_ext(depth_in)
    keys_intr  = list_keys_without_ext(intri_in)
    common_keys = choose_common_keys([keys_color, keys_depth, keys_intr])
    if not common_keys:
        error("no common frames: color/depth/intrinsics mismatch")
        update_scene_mapping(scene_name, split, visit_id, video_id, 0, 0, 0, None, status="skipped_noframes"); return
    info(f"Frames: RGB={len(keys_color)} depth={len(keys_depth)} intr={len(keys_intr)} -> use {len(common_keys)} synced")

    # intrinsics decision
    sample_idxs = [0, len(common_keys)//2, len(common_keys)-1] if len(common_keys)>=3 else list(range(len(common_keys)))
    sample_vals, ok = [], True
    for idx in sample_idxs:
        v = load_intrinsics_per_key_fuzzy(intri_in, common_keys[idx])
        if v is None: ok = False; break
        sample_vals.append(v)
    per_frame_intrinsics = True
    intr_for_mapping = None
    if ok and sample_vals:
        per_frame_intrinsics = not all(v == sample_vals[0] for v in sample_vals)
        if not per_frame_intrinsics:
            ensure_dir(SCANNET_2D)
            W,H,fx,fy,cx,cy = sample_vals[0]
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], np.float64)
            np.savetxt(os.path.join(SCANNET_2D, "intrinsics.txt"), K)
            intr_for_mapping = sample_vals[0]
            info("Intrinsics: global intrinsics.txt written (identical across frames)")
        else:
            info("Intrinsics: per-frame (values vary)")
    else:
        # fallback: still try per-frame
        v0 = load_intrinsics_per_key_fuzzy(intri_in, common_keys[0])
        if v0 is None:
            error(f"cannot load intrinsics (per-frame) in {intri_in}")
            update_scene_mapping(scene_name, split, visit_id, video_id, 0, 0, 0, None, status="skipped_intrinsics"); return
        per_frame_intrinsics = True
        intr_for_mapping = v0
        info("Intrinsics: per-frame (fallback)")

    # 2D export
    def _first_with_any_ext(folder, key):
        cand = sorted(glob.glob(os.path.join(folder, key + ".*")))
        if not cand: cand = sorted(glob.glob(os.path.join(folder, key + "*")))
        return cand[0] if cand else None

    src_colors, src_depths = [], []
    for k in common_keys:
        c = _first_with_any_ext(color_in, k); d = _first_with_any_ext(depth_in, k)
        if c is None or d is None:
            error(f"missing rgb/depth for key {k}"); continue
        src_colors.append(c); src_depths.append(d)

    dst_colors = [os.path.join(color_out, f"{i:06d}.jpg") for i in range(len(src_colors))]
    dst_depths = [os.path.join(depth_out, f"{i:06d}.png") for i in range(len(src_depths))]

    info("2D export: color"); copy_with_progress(src_colors, dst_colors, "2D color", 4)
    info("2D export: depth"); copy_with_progress(src_depths, dst_depths, "2D depth", 4)

    # per-frame intrinsics export
    if per_frame_intrinsics:
        ensure_dir(intr_out_dir)
        checkpoints = {int(len(common_keys)*s/4) for s in range(5)}
        for i,k in enumerate(common_keys):
            v = load_intrinsics_per_key_fuzzy(intri_in, k)
            if v is None: error(f"cannot load intrinsics for key {k}"); continue
            W,H,fx,fy,cx,cy = v
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], np.float64)
            np.savetxt(os.path.join(intr_out_dir, f"{i:06d}.txt"), K)
            if i in checkpoints:
                pct = int(100 * (i / max(1, len(common_keys)-1))); info(f"Intrinsics export: {pct}%")

    # pose export
    try: traj = load_traj_w2c(traj_in)
    except Exception as e:
        error(f"failed to load traj: {traj_in} ({e})")
        update_scene_mapping(scene_name, split, visit_id, video_id, len(src_colors), len(src_depths), 0, intr_for_mapping, status="skipped_traj"); return
    n_pose = min(len(traj), len(src_colors))
    ensure_dir(pose_out); checkpoints = {int(n_pose*s/2) for s in range(3)}
    for i in range(n_pose):
        _, Rw2c, tw2c = traj[i]; Rc2w, tc2w = w2c_to_c2w(Rw2c, tw2c)
        write_pose_txt(os.path.join(pose_out, f"{i:06d}.txt"), Rc2w, tc2w)
        if i in checkpoints:
            pct = int(100 * (i / max(1, n_pose-1))); info(f"Pose export: {pct}%")

    # 3D export — FIX: ラベルは「元点群長」で作ってから、同じマスクで切る
    coords_full, rgb_full = try_read_ply_xyzrgb(vscan)
    if coords_full is None:
        update_scene_mapping(scene_name, split, visit_id, video_id, len(src_colors), len(src_depths), n_pose, intr_for_mapping, status="skipped_ply"); return
    N_full = coords_full.shape[0]
    labels_full = build_labels_from_annotations(N_full, vanno)

    # mask（あれば適用）
    mask = None
    if os.path.isfile(vmask):
        try:
            m = np.load(vmask); mask = (m.astype(bool) if m.dtype != np.bool_ else m)
            if mask.shape[0] != N_full:
                warn(f"crop_mask length mismatch: mask={mask.shape[0]} points={N_full} (ignore mask)")
                mask = None
        except Exception as e:
            warn(f"failed to load crop_mask: {e}"); mask = None

    if mask is None:
        coords, rgb, labels = coords_full, rgb_full, labels_full
    else:
        coords = coords_full[mask]
        rgb   = (rgb_full[mask] if rgb_full is not None else None)
        labels= labels_full[mask]

    # save 3D
    out_3d_base = os.path.join(SCANNET_3D, split, f"{scene_name}")
    ensure_dir(os.path.dirname(out_3d_base))
    if torch is not None:
        try:
            feat = np.zeros((coords.shape[0],3), np.float32) if rgb is None else ((rgb.astype(np.float32)/(255.0 if rgb.max()>1.5 else 1.0))*2-1)
            obj = {
                "coords": torch.from_numpy(coords.astype(np.float32)),
                "feat":   torch.from_numpy(feat.astype(np.float32)),
                "label":  torch.from_numpy(labels.astype(np.int32)),
            }
            torch.save(obj, out_3d_base + ".pth")
            info(f"3D export: coords={coords.shape[0]} pts -> {out_3d_base+'.pth'}")
        except Exception as e:
            error(f"failed to save .pth: {out_3d_base+'.pth'} ({e})")
    else:
        # フォールバック: .npz
        feat = np.zeros((coords.shape[0],3), np.float32) if rgb is None else ((rgb.astype(np.float32)/(255.0 if rgb.max()>1.5 else 1.0))*2-1)
        np.savez_compressed(out_3d_base + ".npz", coords=coords.astype(np.float32), feat=feat.astype(np.float32), label=labels.astype(np.int32))
        warn(f"torch not available; wrote {out_3d_base+'.npz'} instead of .pth")

    # splits & mapping
    append_split_file(scene_name, split)
    update_scene_mapping(scene_name, split, visit_id, video_id,
                         len(src_colors), len(src_depths), n_pose,
                         intr_for_mapping, status="ok")
    done(scene_name)

# ---------- ENTRY ----------
def main():
    parser = argparse.ArgumentParser(description="SceneFun3D -> OpenScene converter (fixed paths)")
    parser.add_argument("--id", type=str, default=None, help="Optional single scene: <visit_id>/<video_id>")
    args = parser.parse_args()

    info("Start OpenScene export")
    info(f"Input={INPUT_ROOT}  Output={OUTPUT_ROOT}")
    for d in [OUTPUT_ROOT, SCANNET_2D, SCANNET_3D, SPLITS_DIR, META_DIR]: ensure_dir(d)
    write_classes_tables()

    train_txt = find_repo_train_list()
    train_visits = read_train_visits(train_txt)

    all_scenes = enumerate_all_scenes()
    name_map = assign_scene_names(all_scenes)

    if args.id is None:
        info("Mode=ALL")
        targets = [(v, vid) for v, vids in all_scenes for vid in vids]
    else:
        info(f"Mode=SINGLE: {args.id}")
        if "/" not in args.id:
            error("invalid --id. expected <visit>/<video>"); sys.exit(1)
        v, w = args.id.split("/", 1)
        try: visit = int(v); vid = int(w)
        except Exception: error("invalid --id numbers"); sys.exit(1)
        if not os.path.isdir(os.path.join(INPUT_ROOT, v, w)):
            error(f"scene not found under INPUT: {INPUT_ROOT}/{v}/{w}"); sys.exit(1)
        targets = [(visit, vid)]
        if (visit, vid) not in name_map: name_map[(visit, vid)] = "scene0000_00"

    if not targets:
        error("no scenes to process"); sys.exit(1)

    for visit, vid in targets:
        scene_name = name_map[(visit, vid)]
        split = split_for_visit(visit, train_visits)
        process_single_scene(visit, vid, scene_name, split)

    done("All tasks")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        error("interrupted"); sys.exit(1)
