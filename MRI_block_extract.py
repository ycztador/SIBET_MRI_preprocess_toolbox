from pathlib import Path
import numpy as np
import SimpleITK as sitk
import cv2

def build_block64_from_mri(
    img_path: str,
    mask_path: str,
    out_path: str | None = None,
    target_hw: tuple[int, int] = (256, 256),
    center_mode: str = "com",   # 'com' | 'max_area' | 'mid_range'
    padding: str = "edge",      # 'edge' | 'zero'
    normalize: str | None = None  # None | 'volume01' | 'slice01'
) -> str:
    """
    基于mask定位中心，向上31层、向下32层，共取64层切片，统一为256x256。
    输出为 4D npy，shape = [1, 64, 256, 256]（N=1，D=64，H=256，W=256）。
    另保存 *_zindices.npy 记录原/使用的Z索引。
    """
    pin_img = Path(img_path)
    pin_msk = Path(mask_path)

    if out_path is None:
        stem = pin_img.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = str(pin_img.with_name(f"{stem}_block64_256.npy"))
    meta_path = str(Path(out_path).with_name(Path(out_path).stem + "_zindices.npy"))

    # 读取为 [Z,Y,X]
    img_itk = sitk.ReadImage(str(pin_img))
    msk_itk = sitk.ReadImage(str(pin_msk))
    if img_itk.GetDimension() != 3 or msk_itk.GetDimension() != 3:
        raise ValueError("仅支持3D NIfTI。")
    img = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    msk = sitk.GetArrayFromImage(msk_itk).astype(np.uint8)
    if img.shape != msk.shape:
        raise ValueError(f"影像与标签尺寸不匹配：{img.shape} vs {msk.shape}")

    Z, H, W = img.shape
    Ht, Wt = int(target_hw[0]), int(target_hw[1])

    # 有效层与中心层
    roi_sums = msk.reshape(Z, -1).sum(axis=1)
    valid_idx = np.where(roi_sums > 0)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("未检测到ROI（mask全为0）。")

    if center_mode == "com":
        z_coords = np.arange(Z, dtype=np.float64)
        center_z = int(np.round(np.average(z_coords, weights=roi_sums.clip(min=0.0))))
    elif center_mode == "max_area":
        center_z = int(valid_idx[np.argmax(roi_sums[valid_idx])])
    elif center_mode == "mid_range":
        center_z = int(np.round((valid_idx[0] + valid_idx[-1]) / 2.0))
    else:
        raise ValueError("center_mode 必须是 'com' / 'max_area' / 'mid_range'。")
    center_z = int(np.clip(center_z, 0, Z - 1))

    # 目标64层
    offsets = np.arange(-31, 33, dtype=int)  # [-31..32]
    z_indices_raw = center_z + offsets
    if padding == "edge":
        z_indices_used = np.clip(z_indices_raw, 0, Z - 1)
    elif padding == "zero":
        z_indices_used = z_indices_raw
    else:
        raise ValueError("padding 必须是 'edge' 或 'zero'。")

    # 组装 [64, Ht, Wt]
    block = np.empty((64, Ht, Wt), dtype=np.float32)
    for i, z in enumerate(z_indices_raw):
        if 0 <= z < Z:
            sl = img[z]
        else:
            sl = img[np.clip(z, 0, Z - 1)] if padding == "edge" else np.zeros((H, W), dtype=np.float32)
        block[i] = cv2.resize(sl, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    # 可选归一化
    if normalize == "volume01":
        mn, mx = float(block.min()), float(block.max())
        block = np.zeros_like(block) if mx <= mn else (block - mn) / (mx - mn)
    elif normalize == "slice01":
        for i in range(block.shape[0]):
            sl = block[i]
            mn, mx = float(sl.min()), float(sl.max())
            block[i] = np.zeros_like(sl) if mx <= mn else (sl - mn) / (mx - mn)

    # 扩维到 [1,64,256,256]
    block4d = block[None, ...].astype(np.float32)
    assert block4d.shape == (1, 64, Ht, Wt), f"unexpected shape: {block4d.shape}"

    # 保存
    np.save(out_path, block4d)
    meta = np.stack([z_indices_raw, np.clip(z_indices_used, 0, Z - 1)], axis=1).astype(np.int32)
    np.save(meta_path, meta)

    return out_path

# 示例
if __name__ == "__main__":
    img_file = r"/data/case001/t2.nii.gz"
    msk_file = r"/data/case001/t2_mask.nii.gz"
    saved = build_block64_from_mri(img_file, msk_file, center_mode="com", padding="edge", normalize=None)
    print("Saved to:", saved)
