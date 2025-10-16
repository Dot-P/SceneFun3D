#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare one sample from two SceneFun3D/OpenScene-like roots.

- LEFT  : base_dir/left_root/<split>/**/*.{pth,pt,pth.tar}
- RIGHT : base_dir/right_root/<split>/**/*.{pth,pt,pth.tar}

PyTorch 2.6 の安全ロード既定 (weights_only=True) による古い pickle のロード失敗に備え、
--trust 指定時のみ allowlist + weights_only=False で再トライします（信頼できるファイル限定）。
"""

import os
import argparse
import glob
import json
import csv
import random
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
from os.path import join

# ------------------------ helpers ------------------------

def _first_present(d: dict, keys: List[str]):
    """dict d において最初に存在するキーの値を返す（存在しなければ None）。
    ※ Tensor を or で連結して真偽値評価しないための安全版。
    """
    for k in keys:
        if k in d:
            return d[k]
    return None

def _extract_coords(obj: Any) -> Optional[torch.Tensor]:
    """Return coords FloatTensor[N,3] if found, else None."""
    # 1) direct tensor
    if torch.is_tensor(obj):
        if obj.ndim >= 2 and obj.shape[1] >= 3:
            return obj[:, :3].float()
        return None

    # 2) dict with common keys
    if isinstance(obj, dict):
        v = _first_present(obj, ["coords", "locs_in", "locs", "xyz", "points", "pos"])
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] >= 3:
            return v[:, :3].float()
        # fallback: any 2D tensor with >=3 columns
        for vv in obj.values():
            if torch.is_tensor(vv) and vv.ndim >= 2 and vv.shape[1] >= 3:
                return vv[:, :3].float()
        return None

    # 3) list/tuple, take the first tensor-like
    if isinstance(obj, (list, tuple)) and obj:
        v = obj[0]
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] >= 3:
            return v[:, :3].float()

    return None

def _shape_str(x: Any) -> str:
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    try:
        return str(tuple(getattr(x, "shape")))
    except Exception:
        return ""

# ------------------------ summarization ------------------------

def _summarize_object(obj: Any) -> Dict[str, Any]:
    """Summarize Python object loaded from .pth for quick inspection."""
    out: Dict[str, Any] = {}

    # structure
    if torch.is_tensor(obj):
        out["type"] = "Tensor"
        out["shape_or_len"] = tuple(obj.shape)
        out["keys"] = ""
    elif isinstance(obj, dict):
        out["type"] = "Dict"
        ks = list(obj.keys())
        out["shape_or_len"] = f"keys={len(ks)}"
        out["keys"] = ", ".join(ks[:16]) + ("..." if len(ks) > 16 else "")
    elif isinstance(obj, (list, tuple)):
        out["type"] = "List" if isinstance(obj, list) else "Tuple"
        out["shape_or_len"] = f"len={len(obj)}"
        out["keys"] = ""
    else:
        out["type"] = type(obj).__name__
        out["shape_or_len"] = ""
        out["keys"] = ""

    # coords
    coords = _extract_coords(obj)
    if coords is None or coords.numel() == 0:
        out.update(dict(
            n_points="",
            coords_dtype="",
            min="",
            max="",
            mean=""
        ))
    else:
        out["n_points"] = coords.shape[0]
        out["coords_dtype"] = str(coords.dtype)
        out["min"] = tuple(torch.min(coords, dim=0).values.tolist())
        out["max"] = tuple(torch.max(coords, dim=0).values.tolist())
        out["mean"] = tuple(torch.mean(coords, dim=0).tolist())

    # extras (common fields, use _first_present to avoid Tensor boolean eval)
    if isinstance(obj, dict):
        rgb      = _first_present(obj, ["rgb"])
        normal   = _first_present(obj, ["normal", "normals"])
        label    = _first_present(obj, ["label", "labels", "semantic", "sem_label", "affordance_id"])
        instance = _first_present(obj, ["instance", "instance_id", "inst_label"])
        feat     = _first_present(obj, ["feat", "features"])

        out["has_rgb"] = rgb is not None
        out["has_normal"] = normal is not None
        out["has_label"] = label is not None
        out["has_instance"] = instance is not None
        out["has_feat"] = feat is not None

        out["rgb_shape"] = _shape_str(rgb)
        out["normal_shape"] = _shape_str(normal)
        out["label_shape"] = _shape_str(label)
        out["instance_shape"] = _shape_str(instance)
        out["feat_shape"] = _shape_str(feat)
        if torch.is_tensor(feat):
            out["feat_dtype"] = str(feat.dtype)
    else:
        for k in ["has_rgb","has_normal","has_label","has_instance","has_feat",
                  "rgb_shape","normal_shape","label_shape","instance_shape","feat_shape","feat_dtype"]:
            out[k] = "" if "has_" not in k else False

    return out

# ------------------------ safe load (PyTorch 2.6+) ------------------------

def _torch_load_safe(path: str) -> Union[Any, Exception]:
    """Try default torch.load (PyTorch 2.6+: weights_only=True)."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        return e

def _torch_load_trust(path: str) -> Union[Any, Exception]:
    """Retry with allowlist + weights_only=False. ONLY for trusted files."""
    try:
        # allowlist for legacy numpy reconstruct
        try:
            from torch.serialization import add_safe_globals, safe_globals
        except Exception:
            add_safe_globals = None
            safe_globals = None

        np_globals = ["numpy._core.multiarray._reconstruct", "numpy.core.multiarray._reconstruct"]
        if add_safe_globals is not None:
            for g in np_globals:
                try:
                    add_safe_globals([g])
                except Exception:
                    pass

        if safe_globals is not None:
            with safe_globals(np_globals):
                return torch.load(path, map_location="cpu", weights_only=False)
        else:
            return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return e

def _load_with_policy(path: str, trust: bool) -> Union[Any, Exception]:
    obj = _torch_load_safe(path)
    if isinstance(obj, Exception) and trust:
        obj = _torch_load_trust(path)
    return obj

# ------------------------ file discovery & picking ------------------------

def _glob_files(root_dir: str, split: str, exts: List[str], recursive: bool) -> List[str]:
    base = join(root_dir, split)
    patts = [join(base, "**", f"*{e}") if recursive else join(base, f"*{e}") for e in exts]
    out: List[str] = []
    for p in patts:
        out.extend(glob.glob(p, recursive=recursive))
    return sorted(out)

def _pick(files: List[str], mode: str, index: int) -> str:
    if not files:
        raise RuntimeError("No candidate files found.")
    if mode == "first":
        return files[0]
    if mode == "index":
        if index < 0 or index >= len(files):
            raise IndexError(f"--index out of range (0..{len(files)-1})")
        return files[index]
    if mode == "random":
        return random.choice(files)
    raise ValueError(f"Unknown pick mode: {mode}")

# ------------------------ summarize path ------------------------

def _summarize_path(path: str, trust: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": path}
    obj = _load_with_policy(path, trust=trust)
    if isinstance(obj, Exception):
        out.update({
            "load_ok": False,
            "error": repr(obj),
        })
        return out

    out["load_ok"] = True
    s = _summarize_object(obj)
    out.update(s)
    return out

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Pick one file from each dataset root and compare structures.")
    ap.add_argument("--base_dir", type=str, default="/nas/data_2/araake/scenefun3d/openscene/data")
    ap.add_argument("--left_root", type=str, default="scenefun_3d")
    ap.add_argument("--right_root", type=str, default="scenefun_3d_vg")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--exts", type=str, default=".pth,.pt,.pth.tar", help="Comma-separated extensions to search.")
    ap.add_argument("--recursive", action="store_true", help="Search recursively under split dir.")
    ap.add_argument("--left_mode", type=str, default="first", choices=["first","index","random"])
    ap.add_argument("--right_mode", type=str, default="first", choices=["first","index","random"])
    ap.add_argument("--left_index", type=int, default=0)
    ap.add_argument("--right_index", type=int, default=0)
    ap.add_argument("--trust", action="store_true",
                    help="Retry with allowlist + weights_only=False IF safe load fails. Use only for trusted files.")
    ap.add_argument("--csv", type=str, default="", help="Optional CSV report path.")
    ap.add_argument("--json", type=str, default="", help="Optional JSON report path.")
    args = ap.parse_args()

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    left_root  = join(args.base_dir, args.left_root)
    right_root = join(args.base_dir, args.right_root)

    left_files  = _glob_files(left_root,  args.split, exts, args.recursive)
    right_files = _glob_files(right_root, args.split, exts, args.recursive)

    print(f"[DEBUG] left_root={left_root} split={args.split} recursive={args.recursive} exts={exts} -> {len(left_files)} files")
    print(f"[DEBUG] right_root={right_root} split={args.split} recursive={args.recursive} exts={exts} -> {len(right_files)} files")

    if not left_files:
        raise SystemExit("No files on LEFT side. Check base/root/split/exts/recursive.")
    if not right_files:
        raise SystemExit("No files on RIGHT side. Check base/root/split/exts/recursive.")

    left_path  = _pick(left_files,  args.left_mode,  args.left_index)
    right_path = _pick(right_files, args.right_mode, args.right_index)

    L = _summarize_path(left_path,  trust=args.trust)
    R = _summarize_path(right_path, trust=args.trust)

    # ---- print result ----
    def _print_block(tag: str, D: Dict[str, Any]):
        print(f"\n# {tag}")
        for k in [
            "path","load_ok","error",
            "type","shape_or_len","keys",
            "n_points","coords_dtype","min","max","mean",
            "has_rgb","rgb_shape","has_normal","normal_shape",
            "has_label","label_shape","has_instance","instance_shape",
            "has_feat","feat_shape","feat_dtype",
        ]:
            if k in D and D[k] != "" and D[k] is not None:
                print(f"{k:14}: {D[k]}")

    _print_block("LEFT", L)
    _print_block("RIGHT", R)

    # quick diff (sizes & ranges) only if both have coords
    if L.get("n_points") and R.get("n_points"):
        lx = L["min"][0], L["max"][0]
        rx = R["min"][0], R["max"][0]
        print("\n# QUICK DIFF (counts & x-range)")
        print(f"n_points(left, right): {L['n_points']} vs {R['n_points']}")
        print(f"x-range(left, right): {lx[0]}..{lx[1]}  |  {rx[0]}..{rx[1]}")

    # optional reports
    if args.csv:
        rows = []
        L1 = {"side":"left"}; L1.update(L)
        R1 = {"side":"right"}; R1.update(R)
        rows.extend([L1, R1])
        fns = sorted(set(k for r in rows for k in r.keys()))
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[INFO] CSV report written to: {args.csv}")

    if args.json:
        data = {"left": L, "right": R}
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] JSON report written to: {args.json}")

if __name__ == "__main__":
    main()
