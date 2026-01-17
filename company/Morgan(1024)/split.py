import numpy as np
import pandas as pd
import json
from pathlib import Path

# ================== 手动配置 ==================
CSV_DIR = Path("csv")
NPZ_DIR = Path("features")
OUT_DIR = Path("npy_data")
# ==============================================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_prefix(split: str, task: str):
    """
    自动匹配 {split}_{task}_*.csv 和 *.npz
    返回 prefix = {split}_{task}_{message}
    """
    csv_files = list(CSV_DIR.glob(f"{split}_{task}_*.csv"))
    npz_files = list(NPZ_DIR.glob(f"{split}_{task}_*.npz"))

    if len(csv_files) != 1 or len(npz_files) != 1:
        raise ValueError(
            f"❌ 文件匹配异常 ({split}, {task})\n"
            f"CSV: {csv_files}\nNPZ: {npz_files}"
        )

    csv_stem = csv_files[0].stem
    npz_stem = npz_files[0].stem

    if csv_stem != npz_stem:
        raise ValueError(
            f"❌ CSV / NPZ 前缀不一致:\n{csv_stem}\n{npz_stem}"
        )

    return csv_stem


def process_one(split: str, task: str):
    """
    split: Train / Test
    task: sif / sgf
    """

    prefix = find_prefix(split, task)
    csv_path = CSV_DIR / f"{prefix}.csv"
    npz_path = NPZ_DIR / f"{prefix}.npz"

    print(f"\n处理 {prefix}")

    # ================== 加载数据 ==================
    df = pd.read_csv(csv_path)
    data = np.load(npz_path, allow_pickle=True)

    X = data["X"]

    # ================== 一致性检查 ==================
    if len(df) != X.shape[0]:
        raise ValueError(
            f"❌ 行数不一致: CSV={len(df)}, NPZ={X.shape[0]} ({prefix})"
        )

    # ================== 选择任务列 ==================
    minute_col = "SIF_minutes" if task == "sif" else "SGF_minutes"

    indices = []
    y_minutes = []
    y_label = []
    source_name = []

    # ================== 按行号顺序对齐 ==================
    for idx, row in df.iterrows():
        minute = row[minute_col]

        if minute == -1 or pd.isna(minute):
            continue

        indices.append(idx)
        y_minutes.append(minute)
        y_label.append(row["label"])
        source_name.append(row["source_name"])

    # ================== 输出目录 ==================
    out_dir = OUT_DIR / task.upper() / split
    ensure_dir(out_dir)

    # ================== 保存 ==================
    np.save(out_dir / f"x_{split.lower()}_{task}.npy", X[indices])
    np.save(out_dir / f"y_{split.lower()}_{task}.npy", np.array(y_minutes))
    np.save(
        out_dir / f"y_{split.lower()}_{task}_label.npy",
        np.array(y_label)
    )
    np.save(
        out_dir / f"{split.lower()}_{task}_source_name.npy",
        np.array(source_name, dtype=object)
    )

    print(f"  样本数: {len(indices)}")

    return data["feature_names"]


def main():
    feature_names = None

    for split, task in [
        ("Train", "sif"),
        ("Test",  "sif"),
        ("Train", "sgf"),
        ("Test",  "sgf"),
    ]:
        fn = process_one(split, task)
        if feature_names is None:
            feature_names = fn

    # ================== 保存 feature names ==================
    ensure_dir(OUT_DIR)

    np.save(OUT_DIR / "feature_names.npy", feature_names)

    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f, indent=2, ensure_ascii=False)

    print("\n✓ npy_data 构建完成")


if __name__ == "__main__":
    main()
