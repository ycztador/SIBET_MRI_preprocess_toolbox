from pathlib import Path
import numpy as np
import SimpleITK as sitk
import cv2

def _to_float01(sl: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sl = sl.astype(np.float32)
    mn, mx = float(sl.min()), float(sl.max())
    if mx - mn < eps:
        return np.zeros_like(sl, dtype=np.float32)
    return (sl - mn) / (mx - mn + eps)

def save_all_slices_to_npy(
    img_path: str,
    out_path: str | None = None,
    target_hw: tuple[int, int] = (256, 256),
    normalize_per_slice: bool = False,
) -> str:

    pin = Path(img_path)
    if out_path is None:
        stem = pin.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = str(pin.with_name(f"{stem}_stack.npy"))

    # 读取并转为数组（SimpleITK: [Z, Y, X]）
    img_itk = sitk.ReadImage(str(pin))
    if img_itk.GetDimension() != 3:
        raise ValueError(f"仅支持 3D NIfTI，当前为 {img_itk.GetDimension()}D：{img_path}")
    arr = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # [Z, Y, X]

    Z = arr.shape[0]
    Ht, Wt = int(target_hw[0]), int(target_hw[1])

    outs = []
    for z in range(Z):
        sl = arr[z]
        if normalize_per_slice:
            sl = _to_float01(sl)
        # 注意：cv2.resize 的 size为  (width, height)
        sl_res = cv2.resize(sl, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        outs.append(sl_res.astype(np.float32))

    stacked = np.stack(outs, axis=0)  # [N, H, W]
    np.save(out_path, stacked)
    return out_path


if __name__ == "__main__":
    # 使用示例
    in_file = r"1_001_t2.nii"
    saved = save_all_slices_to_npy(in_file, normalize_per_slice=False)
    print("Saved to:", saved)
