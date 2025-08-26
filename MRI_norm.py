from pathlib import Path
import numpy as np
import SimpleITK as sitk

def _build_out_path_with_norm_suffix(in_path: Path) -> Path:
    name = in_path.name
    if name.endswith(".nii.gz"):
        return in_path.with_name(name[:-7] + "_norm.nii.gz")  # 去掉 .nii.gz 再加 _norm.nii.gz
    elif name.endswith(".nii"):
        return in_path.with_name(name[:-4] + "_norm.nii")     # 去掉 .nii 再加 _norm.nii
    else:
        # 不常见后缀，保留原名并追加 _norm.nii.gz
        return in_path.with_name(name + "_norm.nii.gz")

def slicewise_znorm_to01(in_path: str,
                         out_path: str | None = None,
                         ignore_zeros: bool = False,
                         eps: float = 1e-8) -> str:
    pin = Path(in_path)
    if out_path is None:
        pout = _build_out_path_with_norm_suffix(pin)
    else:
        pout = Path(out_path)
    pout.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.ReadImage(str(pin))
    if img.GetDimension() != 3:
        raise ValueError(f"仅支持3D NIfTI，当前维度为 {img.GetDimension()}D：{in_path}")

    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # 形状 [z, y, x]
    if arr.ndim != 3:
        raise ValueError(f"期望 3D 数组，得到形状 {arr.shape}")

    out = np.empty_like(arr, dtype=np.float32)

    for z in range(arr.shape[0]):
        sl = arr[z]
        if ignore_zeros:
            mask = sl != 0
            if mask.sum() < 10:        # 若有效像素太少，退化为全体计算
                mask = np.ones_like(sl, dtype=bool)
        else:
            mask = np.ones_like(sl, dtype=bool)

        # z-norm
        mu = sl[mask].mean()
        sd = sl[mask].std()
        if sd < eps:
            zn = np.zeros_like(sl, dtype=np.float32)
        else:
            zn = (sl - mu) / (sd + eps)

        # 线性缩放到 [0,1]（使用该层的 min/max）
        mn = zn.min()
        mx = zn.max()
        if mx - mn < eps:
            norm01 = np.zeros_like(zn, dtype=np.float32)
        else:
            norm01 = (zn - mn) / (mx - mn + eps)

        # 最终保险：clip 到 [0,1]
        out[z] = np.clip(norm01, 0.0, 1.0)

    # 写回 NIfTI，保持原 header 的几何信息
    out_img = sitk.GetImageFromArray(out)
    out_img.SetSpacing(img.GetSpacing())
    out_img.SetOrigin(img.GetOrigin())
    out_img.SetDirection(img.GetDirection())
    sitk.WriteImage(out_img, str(pout))
    return str(pout)


if __name__ == "__main__":
    # 示例
    in_file = r"reample.nii"
    saved = slicewise_znorm_to01(in_file, ignore_zeros=False)
    print("Saved to:", saved)
