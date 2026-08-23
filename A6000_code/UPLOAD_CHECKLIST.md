# 数据上传清单（Mac → 新电脑 RTX A6000）

> 源位置以 Mac 上的实际路径为准；上传后按 README_A6000.md 第 2 节放置。

## Tier 1 —— 必传（训练必需）
| # | 项目 | Mac 源位置 | 大小 | 用途 |
|---|---|---|---|---|
| 1 | A6000_code/（本文件夹） | /Volumes/code_copy/科研工作2026/A6000_code | ~2MB | 已适配 CUDA 的完整代码 |
| 2 | cache_sda_real256/ | SDADiff/cache_sda_real256 | 25GB | 训练缓存（1367 样本，已烘焙真实 ERA5，无需 ERA5 源） |
| 3 | cache_sda_full/ | SDADiff/cache_sda_full | 2.7GB | 验证缓存（45 个留出站，val.npz） |

## Tier 2 —— 评估与画图（强烈推荐）
| # | 项目 | Mac 源位置 | 大小 | 用途 |
|---|---|---|---|---|
| 4 | Data2/aux/ | StormSurgeCastNet-main/Data2/aux | 7.8GB | splits_ids.npy（官方测试 710 站划分）、stats.npy（归一化参数）等 |
| 5 | Data2/combined_gesla_surge.nc | StormSurgeCastNet-main/Data2/ | 10GB | 官方测试实测增水（E2 评估）与图验证 |
| 6 | cache_gulf/ | SDADiff/cache_gulf | 1.6GB | 墨西哥湾案例图（id=528 等） |
| 7 | results_sda/real_era5_256/ | SDADiff/results_sda/real_era5_256 | 107MB | 旧模型权重 best_sda.pth.tar（对照评估） |
| 8 | results_sda/real_era5_256_v3/ | SDADiff/results_sda/real_era5_256_v3 | 429MB | 可选：v3 续训权重（若想在新机器接着训） |

## Tier 3 —— 仅当要重建缓存（当前不需要）
| 项目 | Mac 源位置 | 大小 | 说明 |
|---|---|---|---|
| Data2/GESLA | StormSurgeCastNet-main/Data2/GESLA | 41GB | 原始验潮站（重建缓存用） |
| Data2/GTSM | StormSurgeCastNet-main/Data2/GTSM | 28GB | 粗分辨率模型场 |
| Data2/cache | StormSurgeCastNet-main/Data2/cache | 28GB | 原始 val.npy/train.npy 对象缓存 |
| ERA5 源 | StormSurgeCastNet-main/Data2/ERA5 | ~7MB | ⚠️ 已损坏缺失；**缓存已烘焙真实 ERA5，训练无需 ERA5 源**；重建缓存则需要 |

## 合计
- 仅训练（Tier 1）：**约 28GB**
- 训练 + 评估 + 图（Tier 1+2）：**约 48GB**
- 全量（含 Tier 3）：约 143GB（不建议，ERA5 源缺失无法重建缓存）

## 上传后自检
```
cd A6000_code
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from util.utils import get_device; print(get_device())"   # 应输出 cuda
ls cache_sda_real256 | head -3    # 应有 part_*.npz
ls cache_sda_full                 # 应有 val.npz
```
