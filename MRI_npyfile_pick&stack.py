from pathlib import Path
import numpy as np
import re

def _natural_key(s: str):
    """用于'patient2' < 'patient10'的人性化排序键。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def stack_patient_npys(
    root_dir: str,
    npy_name: str,                 # 例: 'images.npy' / 'labels.npy' / 'block64_256.npy'
    out_name: str | None = None,   # 例: 'images_all.npy'，缺省则自动生成
    skip_missing: bool = True,     # 缺失文件是否跳过（否则报错）
    strict_shape: bool = True      # 是否严格要求 shape[1:] 完全一致
) -> str:
    """
    在 root_dir 下按患者文件夹依次读取 <patient>/<npy_name>，
    沿 axis=0 进行拼接，并保存为 out_name（默认 '<stem>_all.npy'）。

    返回：写出的 .npy 路径
    """
    root = Path(root_dir)
    assert root.is_dir(), f"根目录不存在: {root}"

    # 决定输出文件名
    if out_name is None:
        if npy_name.endswith(".npy"):
            out_name = npy_name[:-4] + "_all.npy"
        else:
            out_name = npy_name + "_all.npy"
    out_path = root / out_name

    # 收集患者子目录（仅目录），并人性化排序
    patients = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
    print(f"[Info] 在 {root} 下找到 {len(patients)} 个患者文件夹。")
    arrays = []
    base_shape_tail = None  # 用于 strict_shape 检查：除 axis=0 外的其余维度

    loaded_count = 0
    for i, p in enumerate(patients, 1):
        f = p / npy_name
        if not f.exists():
            msg = f"[Skip] {i}/{len(patients)}: {p.name} 缺少文件 {npy_name}"
            if skip_missing:
                print(msg)
                continue
            else:
                raise FileNotFoundError(msg)

        try:
            arr = np.load(f, allow_pickle=False)
        except Exception as e:
            msg = f"[Error] {i}/{len(patients)}: 读取失败 {f}，原因：{e}"
            if skip_missing:
                print(msg)
                continue
            else:
                raise

        if base_shape_tail is None:
            base_shape_tail = arr.shape[1:]
            print(f"[Info] 参考形状（去除axis=0）：{base_shape_tail}")

        if strict_shape and arr.shape[1:] != base_shape_tail:
            msg = (f"[ShapeMismatch] {i}/{len(patients)}: {p.name} 形状 {arr.shape} 与参考 (*,{base_shape_tail}) 不一致")
            if skip_missing:
                print(msg + "，已跳过。")
                continue
            else:
                raise ValueError(msg)

        arrays.append(arr)
        loaded_count += 1
        print(f"[Load] {i}/{len(patients)}: {p.name} -> {f.name} 形状 {arr.shape}")

    if not arrays:
        raise RuntimeError("未成功加载任何数组，无法拼接。")

    print(f"[Info] 成功加载 {loaded_count} / {len(patients)} 个样本，开始沿 axis=0 拼接……")
    stacked = np.concatenate(arrays, axis=0)  # 关键：在 axis=0 堆叠
    np.save(out_path, stacked)
    print(f"[Done] 写出：{out_path}  形状：{stacked.shape}  dtype：{stacked.dtype}")
    return str(out_path)


if __name__ == "__main__":
    # 示例：
    # 根目录结构：
    # /root/
    #   ├─ P0001/images.npy
    #   ├─ P0002/images.npy
    #   └─ ...
    path_out = stack_patient_npys("/root", "images.npy", out_name="images_all.npy")
    print("Saved to:", path_out)
