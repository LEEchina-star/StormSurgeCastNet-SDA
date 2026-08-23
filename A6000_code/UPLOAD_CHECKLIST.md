# 数据上传清单（在线下载方案，2026-08 更新）

## 现状：代码已在 GitHub，只需上传 2 样东西

| # | 内容 | 大小 | 上传到 | 状态 |
|---|---|---|---|---|
| 1 | 代码 A6000_code/ | 47MB | GitHub 仓库（目录 A6000_code/） | ✅ 已在线 |
| 2 | 缓存 cache_sda_*（3 个文件夹） | 29GB | Zenodo 新记录（免费 50GB） | ⬜ 待上传 |
| 3 | 模型权重 best_sda.pth.tar + v3/last.pth.tar | 536MB | GitHub Release 资产 | ⬜ 待上传 |

## 你需要做的两步

### ① 上传缓存到 Zenodo（一次，~29GB）
1. 打开 https://zenodo.org/deposit/new （需登录，免费）
2. 拖入 3 个文件夹（不用打包，直接传文件夹内文件）：
   - `cache_sda_real256/`（28 个 part_*.npz，25GB）
   - `cache_sda_full/`（val.npz，2.7GB）
   - `cache_gulf/`（1.6GB）
3. 填标题（如 "StormSurgeCastNet-SDA training caches"）、选 Open Access
4. 发布后得到记录 ID（网址 zenodo.org/records/<ID>），把 ID 发给我
   → 新机器：`python download_data_a6000.py --caches --zenodo <ID>`

> 缓存无法从网上下载重建的原因：其中已烘焙真实 ERA5（Mac 上 ERA5 源已损坏缺失，
> Zenodo 原始记录 12067776/11846592 只有 aux 文件，不含 28GB 原始缓存）。

### ② 上传模型权重到 GitHub Release（~536MB）
1. 打开 https://github.com/LEEchina-star/StormSurgeCastNet-SDA/releases/new
2. 填 Tag（如 v1.0）与标题
3. 拖入：
   - `results_sda/real_era5_256/best_sda.pth.tar`（107MB，旧模型对照）
   - `results_sda/real_era5_256_v3/last.pth.tar`（429MB，续训起点）
4. 发布 → 复制每个资产的直链发给我（或自行使用）
   → 新机器：`python download_data_a6000.py --models <直链1> <直链2>`

## 新机器一条龙（上传完成后）
```
# 0) 装好 conda 环境（README 第 1 节）
# 1) 下载代码
python download_data_a6000.py --code-only
# 2) 下载模型与缓存
python download_data_a6000.py --models <直链...>
python download_data_a6000.py --caches --zenodo <ID>
python download_data_a6000.py --verify          # 全部校验通过后即可训练
```
下载后目录结构自动对齐 README 第 2 节。模型放入 `models_upload/` 后，
用 `--checkpoint models_upload/best_sda.pth.tar` 评估；续训则拷到
`results_sda/real_era5_256_v3/last.pth.tar` 再跑训练命令。

## 备选方案（Zenodo 上传慢/不方便时）
- 百度网盘/阿里云盘传 3 个缓存文件夹（国内最快），把分享链接给我，脚本加 `--caches --netdisk <链接>` 或手动下载后跑 `--verify`
- 局域网/U盘直接拷贝（25GB 约 10-20 分钟）
