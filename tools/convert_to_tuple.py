#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from os.path import join, relpath, dirname, splitext
import glob
import torch
import numpy as np

def extract_coords(d):
    # dict から座標らしき (N,3+) を優先キー順に探索
    for k in ("coords", "locs_in", "locs", "xyz", "points", "pos"):
        v = d.get(k)
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] >= 3:
            return v[:, :3].float()
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] >= 3:
            return torch.as_tensor(v)[:, :3].float()  # いったん Tensor 化→後で NumPy に統一
    # フォールバック：最初に見つかった (N,3+) テンソル/配列
    for v in d.values():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] >= 3:
            return v[:, :3].float()
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] >= 3:
            return torch.as_tensor(v)[:, :3].float()
    raise KeyError("No (N,3) coords-like array/tensor found in dict")

def to_numpy_array(x):
    """Tensor/ndarray/スカラー/リストを NumPy 配列に正規化"""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    elif np.isscalar(x):
        x = np.asarray([x])
    else:
        # list/tuple 等は asarray で
        x = np.asarray(x)
    return x

def to_tuple_np(d, only_coords=False, order="coords,feat,label"):
    """dict から (coords, [feat], [label], ...) を **NumPy 配列**で返す"""
    coords_t = extract_coords(d)          # ここでは Tensor に整形
    coords = to_numpy_array(coords_t).astype(np.float32, copy=False)  # 保存は float32

    if only_coords:
        return (coords,)

    out = [coords]
    key_order = [s.strip() for s in order.split(",") if s.strip()]
    for k in key_order:
        if k == "coords":
            continue
        v = d.get(k)
        if v is not None:
            out.append(to_numpy_array(v))  # 追加要素も必ず NumPy に
    return tuple(out)

def convert_one(src, dst, only_coords=False, order="coords,feat,label", trust=False):
    # PyTorch 2.6+ 安全ロード → 失敗時のみ信頼ファイルとして再試行
    try:
        obj = torch.load(src, map_location="cpu")
    except Exception:
        if not trust:
            raise
        try:
            from torch.serialization import add_safe_globals, safe_globals
        except Exception:
            add_safe_globals = None
            safe_globals = None
        np_globals = [
            "numpy._core.multiarray._reconstruct",
            "numpy.core.multiarray._reconstruct",
        ]
        if add_safe_globals:
            try:
                add_safe_globals(np_globals)
            except Exception:
                pass
        if safe_globals:
            with safe_globals(np_globals):
                obj = torch.load(src, map_location="cpu", weights_only=False)
        else:
            obj = torch.load(src, map_location="cpu", weights_only=False)

    # 上位型ごとに **NumPy タプル**へ正規化
    if isinstance(obj, dict):
        tup = to_tuple_np(obj, only_coords=only_coords, order=order)
    elif isinstance(obj, (list, tuple)):
        # 既に順序タプルなら、中身を NumPy に
        tup = tuple(to_numpy_array(x) for x in obj)
    elif torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        tup = (to_numpy_array(obj),)
    else:
        raise TypeError(f"Unsupported top-level type: {type(obj)}")

    os.makedirs(dirname(dst), exist_ok=True)
    # **NumPy 配列のまま**保存（.pth 拡張子だが中身は ndarray）
    torch.save(tup, dst)
    return tup

def main():
    ap = argparse.ArgumentParser(description="Convert dict-style SceneFun3D .pth to tuple-style .pth (NumPy arrays)")
    ap.add_argument("--src_root", type=str, required=False,
                    default="/nas/data_2/araake/scenefun3d/openscene/data/scenefun_3d",
                    help="Source root (dict/various styles)")
    ap.add_argument("--dst_root", type=str, required=False,
                    default="/nas/data_2/araake/scenefun3d/openscene/data/scenefun_3d_tuple",
                    help="Destination root (tuple-style, saved as NumPy arrays)")
    ap.add_argument("--splits", type=str, default="train,val")
    ap.add_argument("--exts", type=str, default=".pth,.pt,.pth.tar")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--only-coords", action="store_true", help="Save as (coords,) only")
    ap.add_argument("--order", type=str, default="coords,feat,label",
                    help="Tuple order for extra fields if present (all saved as NumPy)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--trust", action="store_true",
                    help="Allow unsafe pickle (weights_only=False) for trusted files")
    ap.add_argument("--limit", type=int, default=0, help="Max files per split (0 = all)")
    args = ap.parse_args()

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    total = 0
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src_dir = join(args.src_root, sp)
        dst_dir = join(args.dst_root, sp)
        patterns = [join(src_dir, "**", f"*{e}") if args.recursive else join(src_dir, f"*{e}") for e in exts]
        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=args.recursive))
        files = sorted(files)
        if args.limit > 0:
            files = files[:args.limit]
        print(f"[{sp}] found: {len(files)} files")

        for i, src in enumerate(files, 1):
            rel = relpath(src, src_dir)
            base, _ = splitext(rel)
            dst = join(dst_dir, base + ".pth")  # 出力は .pth に統一（中身は NumPy）
            if (not args.overwrite) and os.path.exists(dst):
                print(f"  - skip exists: {dst}")
                continue
            try:
                tup = convert_one(src, dst,
                                  only_coords=args.only_coords,
                                  order=args.order,
                                  trust=args.trust)
                n = tup[0].shape[0] if hasattr(tup[0], "shape") and len(tup[0].shape) > 0 else "?"
                print(f"  - ok: {src} -> {dst}   (elements={len(tup)}, N={n})")
                total += 1
            except Exception as e:
                print(f"  - FAIL: {src}  ({type(e).__name__}: {e})")

    print(f"\n[done] converted: {total} files -> {args.dst_root}")

if __name__ == "__main__":
    main()
