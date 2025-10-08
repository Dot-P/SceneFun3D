#!/usr/bin/env python3
# vis_ply.py
import argparse, os, sys
from pathlib import Path

# リポジトリルートを import パスへ（tools/ の一つ上）
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_parser import DataParser
from utils.viz import viz_3d  # ← viz_masks は使わない

def main():
    ap = argparse.ArgumentParser(description="Visualize SceneFun3D PLY -> HTML (pyviz3d).")
    ap.add_argument("--data-root", required=True, help="SceneFun3D data root (…/scenefun3d/original)")
    ap.add_argument("--visit-id", required=True, help="Visit ID (e.g., 420673)")
    ap.add_argument("--out-dir", default="./out", help="Output dir for pyviz3d (default: ./out)")
    ap.add_argument("--apply-crop", action="store_true",
                    help="Physically subset points by crop_mask (preserve original colors).")
    ap.add_argument("--viz-tool", default="pyviz3d", choices=["pyviz3d", "open3d"])
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir   = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # pyviz3d はカレント直下に pyviz3d_output/ を作るため、出力先に一時 chdir
    old_cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        dp  = DataParser(data_root_path=str(data_root))
        pcd = dp.get_laser_scan(args.visit_id)  # XYZRGB 付きの PLY を読む想定

        if args.apply_crop:  # ←タイポ防止: '--apply-crop'
            # 1) マスクの保持インデックスを取得
            idx = dp.get_crop_mask(args.visit_id, return_indices=True)
            # 2) 点群を実際にサブセット化（RGB/法線も一緒にスライスされる）
            pcd = pcd.select_by_index(idx)
            # 3) 元の色を保ったまま可視化
            viz_3d([pcd], viz_tool=args.viz_tool)
        else:
            # マスク適用なしでそのまま可視化（元色のまま）
            viz_3d([pcd], viz_tool=args.viz_tool)

        print(f"[OK] Open: {out_dir}/pyviz3d_output/index.html")
    finally:
        os.chdir(old_cwd)

if __name__ == "__main__":
    main()
