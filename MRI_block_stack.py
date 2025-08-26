from pathlib import Path
import numpy as np
import SimpleITK as sitk
import cv2

def save_center_block64_npy(
    img_path: str,
    out_path: str | None = None,
    target_hw: tuple[int, int] = (256, 256),
    padding: str = "edge",    # 'edge' 边界复制；'zero' 零填充
    dtype=np.float32
) -> str:
    """
    读取 3D MRI（.nii/.nii.gz），以 Z 轴中点为中心，采样 64 层（上31、下32），
    每层 resize 到 target_hw，输出形状 [1,64,H,W] 的 .npy。

    参数
    ----
    img_path : str
        影像 NIfTI 路径（3D）
    out_path : str | None
        输出 .npy 路径；None 时使用原文件名追加 '_block64_256.npy'
    target_hw : (H, W)
        每层目标大小（默认 256×256）
    padding : {'edge','zero'}
        采样越界时的补齐策略
    dtype : numpy dtype
        保存的数据类型

    返回
    ----
    str : 写出的 .npy 路径
    """
    pin = Path(img_path)
    if out_path is None:
        stem = pin.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = str(pin.with_name(f"{stem}_block64_256.npy"))

    # 读入为 [Z, Y, X]
    img_itk = sitk.ReadImage(str(pin))
    if img_itk.GetDimension() != 3:
        raise ValueError(f"仅支持 3D NIfTI，当前为 {img_itk.GetDimension()}D：{img_path}")
    vol = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # [Z,Y,X]

    Z, H, W = vol.shape
    Ht, Wt = map(int, target_hw)

    # 以中点为中心，得到 64 层窗口
    # 先尽量把窗口放在体内；Z<64 时不可避免越界，后续按 padding 处理
    center = Z // 2
    win = 64
    half_up, half_down = 31, 32  # 上31、下32
    start = center - half_up
    if Z >= win:
        start = max(0, min(start, Z - win))
    z_indices = np.arange(start, start + win, dtype=int)  # 期望的层索引（可能越界）

    # 生成块 [64, Ht, Wt]
    block = np.empty((win, Ht, Wt), dtype=np.float32)
    for i, z in enumerate(z_indices):
        if 0 <= z < Z:
            sl = vol[z]
        else:
            if padding == "edge":
                sl = vol[np.clip(z, 0, Z - 1)]
            elif padding == "zero":
                sl = np.zeros((H, W), dtype=np.float32)
            else:
                raise ValueError("padding 必须是 'edge' 或 'zero'。")
        # resize（cv2 接收尺寸为 (W,H)）
        sl_res = cv2.resize(sl, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        block[i] = sl_res

    # 扩到 [1,64,H,W] 并保存
    block4d = block[None, ...].astype(dtype)
    np.save(out_path, block4d)
    return out_path


if __name__ == "__main__":
    in_file = r"/data/case001/t2.nii.gz"
    saved = save_center_block64_npy(in_file, padding="edge")
    print("Saved to:", saved)
