#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
file_path = "Data2/GTSM/reanalysis_surge_hourly_1950_01_v3.nc"
ds = xr.open_dataset(file_path)
#%%
# ======================
# 你的文件路径
# ======================
# file_path = "/Volumes/EXTERNAL_USB/ERA5_WIND/199301.nc"
file_path = '/Volumes/CYGNSSSAR/ERA5-GTSM/temp/era5_gtsm_surge_hourly_1978_08_v3.nc'
# "Data2/GTSM/reanalysis_surge_hourly_1978_08_v3.nc"
# ======================
# 读取 NetCDF 数据
# ======================
# print("正在读取 ERA5 文件...")
ds = xr.open_dataset(file_path)
#%%
# ======================
# 打印文件基本信息
# ======================
print("\n===== 文件信息 =====")
print(ds)

print("\n===== 变量列表 =====")
print(list(ds.variables))

print("\n===== 维度信息 =====")
print(ds.dims)

# ======================
# 查看每个变量的形状
# ======================
print("\n===== 变量形状 =====")
for var in ds.data_vars:
    print(f"{var}: {ds[var].shape}")

# ======================
# 读取经纬度 + 时间
# ======================
lon = ds.longitude.values
lat = ds.latitude.values
time = ds.time.values

print(f"\n经度范围: {lon.min():.2f} ~ {lon.max():.2f}")
print(f"纬度范围 : {lat.min():.2f} ~ {lat.max():.2f}")
print(f"时间步长 : {len(time)} 个时刻")
print(f"时间范围 : {time[0]}  ~  {time[-1]}")

# ======================
# 读取第一个时刻的风速（如果存在 u10, v10）
# # ======================
# if "u10" in ds:
#     u10 = ds.u10.values
#     print(f"\nu10 形状 (time, lat, lon): {u10.shape}")

# if "v10" in ds:
#     v10 = ds.v10.values
#     print(f"v10 形状 (time, lat, lon): {v10.shape}")

# if "msl" in ds:
#     msl = ds.msl.values
#     print(f"msl 形状 (time, lat, lon): {msl.shape}")

# # ======================
# # 画一张图（第一个时间步的 u10）
# # ======================
# try:
#     plt.figure(figsize=(12, 6))
#     plt.pcolormesh(lon, lat, u10[0], cmap="jet")
#     plt.colorbar()
#     plt.title("ERA5 10m U wind component (first time step)")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.show()
# except:
#     print("\n无法绘图（可能没有 u10 变量 或 无 matplotlib）")

print("\n✅ 读取完成！文件正常！")
# %%
