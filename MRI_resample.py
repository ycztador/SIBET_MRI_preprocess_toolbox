from pathlib import Path
import SimpleITK as sitk

def resample_nii_to_isotropic(in_path: str,
                              out_path: str,
                              target_spacing=(1.0, 1.0, 1.0),
                              is_label: bool = False) -> str:

    in_path = str(in_path)
    out_path = str(out_path)

    img = sitk.ReadImage(in_path)
    if img.GetDimension() != 3:
        raise ValueError(f"仅支持3D NIfTI，当前维度为 {img.GetDimension()}D：{in_path}")

    orig_spacing = img.GetSpacing()      # (sx, sy, sz)
    orig_size    = img.GetSize()         # (nx, ny, nz)

    # 计算新尺寸： n_new = round(n_old * (s_old / s_new))
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / float(target_spacing[i]))))
        for i in range(3)
    ]

    # 选择插值器
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline

    # 配置重采样
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(tuple(float(s) for s in target_spacing))
    rs.SetSize(new_size)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetTransform(sitk.Transform())  # 恒等变换
    rs.SetInterpolator(interpolator)
    rs.SetOutputPixelType(img.GetPixelID())

    resampled = rs.Execute(img)

    # 确保输出目录存在
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, out_path)

    return out_path


if __name__ == "__main__":

    in_file  = r"test.nii"
    out_file = r"reample.nii"
    resample_nii_to_isotropic(in_file, out_file, target_spacing=(1.0,1.0,1.0), is_label=False)
    print("Saved to:", out_file)
