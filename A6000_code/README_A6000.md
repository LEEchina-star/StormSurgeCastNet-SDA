# StormSurgeCastNet-SDA —— RTX A6000 (Windows/CUDA) 适配版

> ## 0. 在线获取（无需手动拷数据）
> 本文件夹已同步到 GitHub：`github.com/LEEchina-star/StormSurgeCastNet-SDA`（目录 `A6000_code/`）。
> 新机器上只需两样东西，用 `download_data_a6000.py` 一键下载（自动校验 SHA256）：
> ```
> python download_data_a6000.py --code-only     # 代码（GitHub，已在线）
> python download_data_a6000.py --models <Release资产直链>   # 模型权重（上传后）
> python download_data_a6000.py --caches --zenodo <记录ID>  # 训练缓存（上传 Zenodo 后）
> python download_data_a6000.py --verify        # 完整性校验
> ```
> 待上传的两样（见 UPLOAD_CHECKLIST.md）：① 缓存 ~29GB → Zenodo（免费 50GB/数据集）；② 模型权重 → GitHub Release。
>

本目录由 Mac (MPS) 版本移植：**设备自动选择 CUDA 优先**（RTX A6000），回退 MPS/CPU。
与原版唯一差异在设备处理与 A6000 加速选项，**训练协议（float32、batch 4、256x256、30 epochs）与论文一致**。

## 1. 环境安装

### 方式 A: conda（推荐，仓库自带 environment.yml，pytorch 2.2 + cuda 11.8）
```
conda env create -f environment.yml
conda activate surgecast
```

### 方式 B: pip 手动
```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib xarray netcdf4 pandas einops scikit-image rasterio
```

## 2. 数据放置（重要）

Windows 无符号链接，直接复制文件夹到 A6000_code/ 下：

```
A6000_code/
├── cache_sda_real256/            # 训练缓存 ~25GB（已烘焙真实 ERA5）
├── cache_sda_full/               # 验证缓存 ~2.7GB（val.npz, 45 样本）
├── cache_gulf/                   # (可选) 墨西哥湾样本 ~1.6GB（画图）
├── Data2/
│   ├── aux/                      # splits_ids.npy / stats.npy 等 ~7.8GB（评估用）
│   ├── combined_gesla_surge.nc   # ~10GB（官方测试/图验证）
│   └── ...
└── results_sda/                  # (可选) 旧权重 / 续训
```

## 3. 训练

### 全新训练（30 epochs，与论文协议一致）
```
python train_sda_cache.py --cache_dir cache_sda_real256 --val_cache_dir cache_sda_full --out results_sda/real_era5_256 --epochs 30 --batch_size 4
```

### 断点续训（同一条命令，自动从 last.pth.tar 的 epoch 继续）
```
python train_sda_cache.py --cache_dir cache_sda_real256 --val_cache_dir cache_sda_full --out results_sda/real_era5_256 --epochs 30 --batch_size 4
```

### A6000 加速选项
- `--amp`：fp16 混合精度（约快 1.5-2x；默认关闭，保持 float32 论文数值）
- `--batch_size 8`：48GB 显存完全够（默认 4，显存占用约 8GB）
- 首次启动加载 28 个缓存分片（~25GB），需几分钟，属正常

## 4. 评估

```
# 官方测试协议（batch=4, seed=0, 256, steps=25, ensemble=4）
python test_sda_cache.py --cache_dir cache_sda_full ^
    --checkpoint results_sda/real_era5_256/best_sda.pth.tar ^
    --out results_sda/test_report --resize 0 --ensemble 4 --sample_steps 25 ^
    --sda_guidance 1.0 --obs_noise 0.1

# E2 densification / E5 输入消融 / 真实协议对比
python eval_densification.py results_sda/real_era5_256/best_sda.pth.tar
python eval_input_ablation.py results_sda/real_era5_256/best_sda.pth.tar
python eval_forecast_protocols.py
```

## 5. Windows 注意事项
- 路径正斜杠/反斜杠均可（代码统一用 os.path）
- 中文输出乱码：`chcp 65001` 或设 `PYTHONIOENCODING=utf-8`
- `num_workers` 默认 0（Windows 最稳）；想提速可改 train_sda_cache.py 第 ~100 行为 `num_workers=2`（脚本已有 `if __name__=="__main__"` 保护）
- 不要用软链接放 Data2，直接复制文件夹
- 训练日志在 `results_sda/<out>/train.log`（nohup 时）与 `trainlog.json`（每 epoch）

## 6. 预期速度（RTX A6000, float32, batch 4, 256x256）
- 单 epoch 约 30-60 分钟（Mac MPS 为 ~5.9 小时）
- 30 epochs 约 1-2 天；加 `--amp` 可再快约 1.5-2 倍
