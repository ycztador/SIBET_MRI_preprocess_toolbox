from pathlib import Path
import numpy as np
import SimpleITK as sitk
import cv2

def _to_float01_per_slice(sl: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sl = sl.astype(np.float32)
    mn, mx = float(sl.min()), float(sl.max())
    if mx - mn < eps:
        return np.zeros_like(sl, dtype=np.float32)
    return (sl - mn) / (mx - mn + eps)

def save_roi_slices_to_npy(
    img_path: str,
    lbl_path: str,
    target_hw: tuple[int, int] = (256, 256),
    out_dir: str | None = None,
    normalize_per_slice: bool = True,
) -> tuple[str, str]:
    """
    读取 MRI 与标签（NIfTI），筛选出 ROI>0 的切片，统一到 target_hw，并保存为 .npy

    参数
    ----
    img_path : str
        MRI 的 .nii 或 .nii.gz 路径（3D）
    lbl_path : str
        标签的 .nii 或 .nii.gz 路径（与 MRI 同几何尺寸，3D）
    target_hw : (H, W)
        统一输出的空间尺寸，默认 (256,256)
    out_dir : str | None
        输出目录；若为 None，则与 MRI 同目录
    normalize_per_slice : bool
        是否对图像每个切片做 0-1 归一化（基于该切片 min/max）

    返回
    ----
    (images_npy_path, labels_npy_path)
        写出的图像与标签 .npy 文件路径
    """
    img_p = Path(img_path)
    lbl_p = Path(lbl_path)
    if out_dir is None:
        outdir = img_p.parent
    else:
        outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取
    img_itk = sitk.ReadImage(str(img_p))
    lbl_itk = sitk.ReadImage(str(lbl_p))
    if img_itk.GetDimension() != 3 or lbl_itk.GetDimension() != 3:
        raise ValueError("仅支持 3D NIfTI。")

    img_arr = sitk.GetArrayFromImage(img_itk)  # [Z, Y, X]
    lbl_arr = sitk.GetArrayFromImage(lbl_itk)  # [Z, Y, X]
    if img_arr.shape != lbl_arr.shape:
        raise ValueError(f"图像与标签尺寸不匹配：{img_arr.shape} vs {lbl_arr.shape}")

    Z = img_arr.shape[0]
    Ht, Wt = int(target_hw[0]), int(target_hw[1])

    imgs_out, lbls_out = [], []
    keep_indices = []

    for z in range(Z):
        lbl_slice = lbl_arr[z]
        if np.any(lbl_slice > 0):  # ROI 存在
            img_slice = img_arr[z].astype(np.float32)

            # 归一化（可选）
            if normalize_per_slice:
                img_slice = _to_float01_per_slice(img_slice)

            # resize 到目标大小（图像线性插值，标签最近邻）
            img_res = cv2.resize(img_slice, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
            lbl_res = cv2.resize(lbl_slice.astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_NEAREST)

            # 标签保存为 int（保持原有标签值，非二值也可）
            imgs_out.append(img_res.astype(np.float32))
            lbls_out.append(lbl_res.astype(np.int16))
            keep_indices.append(z)

    if len(imgs_out) == 0:
        raise RuntimeError("未找到包含 ROI 的切片（标签全为 0）。")

    imgs_np = np.stack(imgs_out, axis=0)   # [N, H, W]
    lbls_np = np.stack(lbls_out, axis=0)   # [N, H, W]

    # 生成输出文件名：原名 + 后缀
    stem = img_p.name.replace(".nii.gz", "").replace(".nii", "")
    images_npy = outdir / f"{stem}_images.npy"
    labels_npy = outdir / f"{stem}_labels.npy"
    idxs_npy   = outdir / f"{stem}_slice_indices.npy"  # 记录原始 Z 索引，便于回溯

    np.save(images_npy, imgs_np)
    np.save(labels_npy, lbls_np)
    np.save(idxs_npy, np.array(keep_indices, dtype=np.int32))

    return str(images_npy), str(labels_npy)


if __name__ == "__main__":
    # 使用示例
    img_file = r"1_001_t2.nii"
    lbl_file = r"1_001_mask.nii"
    out_img_npy, out_lbl_npy = save_roi_slices_to_npy(img_file, lbl_file)
    print("Saved:", out_img_npy, out_lbl_npy)
