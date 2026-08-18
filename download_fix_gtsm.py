import os
import re
import cdsapi

# ===================== 配置（直接用） =====================
STORE_AT = "DATA_temp"  # 你的数据根目录
FAIL_LOG = "/Volumes/code_copy/科研工作2026/PINN- STGNN/nc_missing_files_1950_2024.txt"
GTSM_FINAL_DIR = os.path.join(STORE_AT, 'GTSM')
os.makedirs(GTSM_FINAL_DIR, exist_ok=True)

# ============== 从 txt 读取需要下载的 年/月 ==============
def parse_failed_years_months():
    """解析失败记录，提取 (year, month) 列表"""
    pattern = r"reanalysis_surge_hourly_(\d{4})_(\d{2})"
    failed = []

    if not os.path.exists(FAIL_LOG):
        print(f"❌ {FAIL_LOG} 不存在")
        return failed

    with open(FAIL_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                year = match.group(1)
                month = match.group(2)
                failed.append((year, month))

    # 去重
    failed = sorted(list(set(failed)))
    print(f"✅ 需要修复下载的文件数：{len(failed)}")
    for y, m in failed:
        print(f"   → {y} 年 {m} 月")
    return failed

# ===================== CDS 下载 =====================
def download_gtsm_month(year, month, target_dir):
    c = cdsapi.Client()

    filename = f"reanalysis_surge_hourly_{year}_{month}_v3.nc"
    save_path = os.path.join(target_dir, filename)
    zip_path = os.path.join(target_dir, f"temp_{year}_{month}.zip")

    print(f"\n🚀 下载：{year}-{month}")

    c.retrieve(
        'sis-water-level-change-timeseries-cmip6',
        {
            'variable': 'storm_surge_residual',
            'experiment': 'reanalysis',
            'temporal_aggregation': 'hourly',
            'year': year,
            'month': month,
            'format': 'zip',
            "version": ["v3"]
        },
        zip_path
    )

    # 解压 + 删除压缩包
    os.system(f"unzip -q -o {zip_path} -d {target_dir}")
    os.remove(zip_path)

    print(f"✅ 完成：{save_path}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    missing = parse_failed_years_months()

    if not missing:
        print("\n🎉 没有需要修复的文件！")
        exit()

    print("\n开始批量修复下载缺失的 GTSM 数据...")
    for y, m in missing:
        download_gtsm_month(y, m, GTSM_FINAL_DIR)

    print("\n🎉 所有缺失文件修复下载完成！")