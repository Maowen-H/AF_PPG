#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_npz_to_h5.py
Convert a large DeepBeat-like NPZ into an HDF5 with chunked+compressed datasets.

Usage:
  python npz_to_h5.py \
      --npz "/mnt/d/25 summer research/mobile health/train.npz" \
      --out "/home/wren/projects/Af_PPG/ppg_train.h5" \
      --workdir "/home/wren/projects/Af_PPG/" \
      --chunk-rows 1024 \
      --float32 \
      --subjects-keep-csv "/mnt/d/keep_subjects_val.csv"   # 可选

先在 validate/test 上跑通，再处理 train。
"""

import argparse
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np

def infer_members(npz_path: Path):
    """List member files inside NPZ (zip)."""
    with zipfile.ZipFile(npz_path) as zf:
        names = zf.namelist()
    return names

def extract_member(npz_path: Path, member: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"'{member}' not found in {npz_path}. Members: {zf.namelist()[:10]} ...")
        zf.extract(member, path=out_dir)
    return out_dir / member

def load_small_key(npz_path: Path, key: str):
    # 标签/参数相对小，可以直接读
    with np.load(npz_path, allow_pickle=True) as d:
        return d[key]

def to_subject_ids(parameters_obj_array: np.ndarray) -> np.ndarray:
    # parameters: shape (N, 3) object -> [timestamp, device, subject_id]
    subj = np.array([int(x) for x in parameters_obj_array[:, 2]], dtype=np.int32)
    return subj

def to_devices(parameters_obj_array: np.ndarray) -> np.ndarray:
    dev = parameters_obj_array[:, 1].astype(str)
    # 去除可能的前导空格
    return np.char.strip(dev)

def to_timestamps(parameters_obj_array: np.ndarray) -> np.ndarray:
    # 转成 ISO 字符串，写入 vlen string
    ts = parameters_obj_array[:, 0]
    # 某些元素是 pandas.Timestamp；统一成字符串
    ts_str = np.array([str(x) for x in ts], dtype=object)
    return ts_str

# def load_subjects_keep(csv_path: Path):
#     if not csv_path or not csv_path.exists():
#         return None
#     keep = set()
#     with open(csv_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line=line.strip()
#             if not line: 
#                 continue
#             try:
#                 keep.add(int(line.split(",")[0]))
#             except Exception:
#                 # 支持纯数字或逗号分隔第一列
#                 pass
#     return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=str, help="Path to input .npz")
    ap.add_argument("--out", required=True, type=str, help="Path to output .h5")
    ap.add_argument("--workdir", required=True, type=str, help="Dir to extract big npy (temp workspace)")
    ap.add_argument("--chunk-rows", type=int, default=1024, help="HDF5 chunk rows (per-sample chunk)")
    ap.add_argument("--float32", action="store_true", help="Store signal as float32 (recommended)")
    ap.add_argument("--subjects-keep-csv", type=str, default=None,
                    help="Optional CSV with one subject_id per line to keep (use to purge overlaps)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)
    workdir  = Path(args.workdir)

    print(f"[i] NPZ: {npz_path}")
    print(f"[i] OUT: {out_path}")
    print(f"[i] WORKDIR: {workdir}")

    # 1) 读取小键
    rhythm = load_small_key(npz_path, "rhythm")
    params = load_small_key(npz_path, "parameters")
    N = rhythm.shape[0]
    print(f"[i] N (samples) = {N}")

    subject_id = to_subject_ids(params)
    device     = to_devices(params)
    timestamp  = to_timestamps(params)

    # 可选：仅保留给定 subject 的样本（例如去除 val/test 重叠）
    # keep_set = load_subjects_keep(Path(args.subjects_keep_csv)) if args.subjects_keep_csv else None
    # if keep_set is not None:
    #     mask = np.array([sid in keep_set for sid in subject_id], dtype=bool)
    #     kept = mask.sum()
    #     print(f"[i] Filtering by subjects-keep list: keep {kept}/{N} samples")
    # else:
    #     mask = None

    # 2) 找到并解出 signal.npy
    members = infer_members(npz_path)
    sig_member = "signal.npy"
    if sig_member not in members:
        raise FileNotFoundError(f"'signal.npy' not in {members[:10]} ...")
    sig_npy = extract_member(npz_path, sig_member, workdir)
    print(f"[i] extracted: {sig_npy}")

    # 3) 用 memmap 只读打开，推回每窗口长度 L
    sig_mm = np.lib.format.open_memmap(sig_npy, mode="r")
    total_len = sig_mm.size
    if N <= 0 or total_len % N != 0:
        raise RuntimeError(f"signal.size={total_len} not divisible by N={N}")
    L = total_len // N
    print(f"[i] total points={total_len}, per-window length L={L}")

    # 4) 创建 H5，写入压缩+分块的数据集
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        os.remove(out_path)

    dtype_signal = np.float32 if args.float32 else sig_mm.dtype
    with h5py.File(out_path, "w") as f:
        # 建 vlen string dtype
        str_dtype = h5py.string_dtype(encoding="utf-8")

        # 先根据 mask 确定输出大小
        # if mask is None:
        #     out_N = N
        # else:
        #     out_N = int(mask.sum())
        out_N = N

        # 创建 datasets
        dset_sig = f.create_dataset(
            "signal", shape=(out_N, L), dtype=dtype_signal,
            chunks=(min(args.chunk_rows if hasattr(args, 'chunk-rows') else args.chunk_rows, out_N), L),
            compression="gzip", shuffle=True
        )
        dset_y   = f.create_dataset("rhythm",     shape=(out_N,2), dtype=np.uint8)
        dset_sid = f.create_dataset("subject_id", shape=(out_N,), dtype=np.int32)
        dset_dev = f.create_dataset("device",     shape=(out_N,), dtype=str_dtype)
        dset_ts  = f.create_dataset("timestamp",  shape=(out_N,), dtype=str_dtype)

        # 分块搬运
        B = 200_000  # 每次搬多少“窗口”（按内存/磁盘吞吐调整）
        write_ptr = 0
        for i in range(0, N, B):
            j = min(i + B, N)
            # 当前块的索引（应用mask）
            idx = slice(i, j)
            out_idx = slice(write_ptr, write_ptr + (j - i))
            # if mask is None:
            #     idx = slice(i, j)
            #     out_idx = slice(write_ptr, write_ptr + (j - i))
            # else:
            #     sel = np.nonzero(mask[i:j])[0]
            #     if sel.size == 0:
            #         continue
            #     idx = i + sel
            #     out_idx = slice(write_ptr, write_ptr + sel.size)

            # 取 signal 视图（不进内存），再降精度
            block = sig_mm.reshape(N, L)[idx]
            if dtype_signal == np.float32 and block.dtype != np.float32:
                block = block.astype(np.float32, copy=False)

            # 写入
            dset_sig[out_idx] = block
            dset_y[out_idx]   = rhythm[idx].astype(np.uint8, copy=False)
            dset_sid[out_idx] = subject_id[idx]
            dset_dev[out_idx] = device[idx]
            dset_ts[out_idx]  = timestamp[idx]

            write_ptr = out_idx.stop
            print(f"[i] wrote rows {out_idx.start}..{out_idx.stop - 1}")

        # 元数据（可选）
        f.attrs["source_npz"] = str(npz_path)

    # 5) 收尾：删除临时 signal.npy（避免双份占盘）
    try:
        os.remove(sig_npy)
        print(f"[i] removed temp file: {sig_npy}")
    except Exception as e:
        print(f"[w] failed to remove temp npy: {e}")

    print(f"[✓] Done. H5 saved at: {out_path}")

if __name__ == "__main__":
    main()
