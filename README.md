# SIBET_MRI_preprocess_toolbox
MR影像的标准化预处理流程和工具包
MRI_resample.py：把 3D NIfTI 重采样到 spacing=[1,1,1]，图像用线性/BSpline、标签用最近邻，保留原点与方向信息。

MRI_norm.py：对体数据逐层 z-norm并线性缩放到 [0,1]，输出带 _norm 后缀的新 NIfTI。

MRI_slice_extract.py：基于 mask 筛出含 ROI 的切片，统一为 256×256，按病例保存为切片堆叠的 images.npy / labels.npy（并记录原始 Z 索引）。

MRI_block_extract.py：以 mask 的 Z 轴中心为基准取 64 层（上31/下32），每层 256×256，导出 [1,64,256,256].npy 的三维块。

MRI_slice_stack.py：将各病例的切片 .npy（如 images.npy / labels.npy）按患者排序在 axis=0 拼接，得到数据集级的 images_stack.npy / masks_stack.npy。

MRI_stack.py：通用堆叠器——在根目录下按患者文件夹排序读取指定类型的 .npy，沿 axis=0 合并为 <type>_all.npy。

npyfile_pick&stack.py：从已生成的多个 stack（如 images_stack.npy、masks_stack.npy、images_block_stack.npy）中选择/对齐/合并，产出新的组合数据集。
