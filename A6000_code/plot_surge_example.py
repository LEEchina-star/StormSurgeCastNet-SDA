import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer  # 用于投影坐标转经纬度

# ====================== 1. 读取NetCDF文件 ======================
file_path = "Data2/GTSM/reanalysis_surge_hourly_1950_01_v3.nc"
ds = xr.open_dataset(file_path)

# ====================== 2. 关键信息解读（适配stations维度） ======================
print("="*50 + " 修正后的数据核心信息 " + "="*50)
# 时间维度
time_arr = ds["time"].values
print(f"时间范围：{time_arr[0]} 至 {time_arr[-1]}（共{len(time_arr)}小时）")
# 站点维度
n_stations = ds.dims["stations"]
print(f"站点总数：{n_stations}个")
# 风暴增水统计
surge_arr = ds["surge"].values
print(f"风暴增水范围：{np.nanmin(surge_arr):.3f} ~ {np.nanmax(surge_arr):.3f} m")
print(f"风暴增水均值：{np.nanmean(surge_arr):.3f} m（剔除缺失值）")

# ====================== 3. 投影坐标转经纬度（核心修正） ======================
# GTSM的station_x/y是RD新坐标（EPSG:28992），需转为WGS84经纬度（EPSG:4326）
def convert_coords(x, y):
    """将GTSM的RD坐标（EPSG:28992）转为WGS84经纬度（EPSG:4326）"""
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat

# 提取站点投影坐标并转换为经纬度
x_coords = ds["station_x_coordinate"].values
y_coords = ds["station_y_coordinate"].values
lon, lat = convert_coords(x_coords, y_coords)

# 过滤异常坐标（避免可视化出错）
valid_mask = (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
lon_valid = lon[valid_mask]
lat_valid = lat[valid_mask]
station_idx_valid = np.where(valid_mask)[0]  # 有效站点的索引

print(f"\n有效站点数：{len(lon_valid)}个（剔除异常坐标后）")
print(f"经纬度范围：\n  经度：{lon_valid.min():.2f} ~ {lon_valid.max():.2f}°E\n  纬度：{lat_valid.min():.2f} ~ {lat_valid.max():.2f}°N")

# ====================== 4. 提取PINN-STGNN所需核心变量 ======================
# 4.1 静态特征（站点级，不随时间变）
station_features = {
    "lon": lon_valid,          # 经度（GNN节点空间特征）
    "lat": lat_valid,          # 纬度（GNN节点空间特征）
    "station_idx": station_idx_valid  # 有效站点的原始索引
}

# 4.2 动态特征（时间+站点级，模型输入/标签）
# 提取有效站点的风暴增水（适配PINN-STGNN的标签）
surge_valid = ds["surge"].values[:, station_idx_valid]  # shape: (744, 有效站点数)
# 时间特征（转为小时数，便于时序窗口构建）
time_hours = (ds["time"] - ds["time"].isel(time=0)).dt.total_seconds().values / 3600

# 示例：提取1950-01-01 12:00的风暴增水（第12个时间步）
surge_t12 = surge_valid[12, :]
print(f"\n1950-01-01 12:00的风暴增水：\n  均值：{np.nanmean(surge_t12):.3f} m\n  最大值：{np.nanmax(surge_t12):.3f} m（极端增水站点）")

# ====================== 5. 可视化验证（修正坐标后） ======================
def plot_surge_stations(ds, surge_data, lon, lat, time_idx=12):
    """绘制指定时间点的站点风暴增水分布（适配经纬度）"""
    plt.figure(figsize=(14, 8))
    # 全球投影，聚焦主要海区
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="gray")
    ax.set_global()  # 显示全球
    
    # 掩膜缺失值（只绘制有效增水数据）
    valid_surge_mask = ~np.isnan(surge_data)
    lon_plot = lon[valid_surge_mask]
    lat_plot = lat[valid_surge_mask]
    surge_plot = surge_data[valid_surge_mask]
    
    # 绘制站点增水（颜色区分增水大小，点大小区分权重）
    sc = ax.scatter(
        lon_plot, lat_plot, 
        c=surge_plot, cmap="RdBu_r", 
        vmin=-0.5, vmax=0.5,  # 增水范围（可根据数据调整）
        s=0.5, alpha=0.8,     # 点大小和透明度（避免重叠）
        transform=ccrs.PlateCarree()
    )
    
    # 添加颜色条和标题
    cbar = plt.colorbar(sc, ax=ax, label="Storm Surge (m)", shrink=0.6, pad=0.05)
    time_label = ds["time"].isel(time=time_idx).dt.strftime("%Y-%m-%d %H:%M").values
    ax.set_title(f"ERA5-GTSM Hourly Storm Surge at {time_label}", fontsize=12)
    
    # 保存并显示
    plt.tight_layout()
    plt.savefig("surge_stations_global.png", dpi=300, bbox_inches="tight")
    # plt.show()

# 绘制1950-01-01 12:00的全球站点增水分布
plot_surge_stations(ds, surge_t12, lon_valid, lat_valid, time_idx=12)

# ====================== 6. 构建GNN图结构的前置准备（关键） ======================
def build_station_adjacency(lon, lat, k=10):
    """
    为站点构建邻接矩阵（适配GNN）：每个站点取最近的k个邻居
    :param lon/lat: 站点经纬度
    :param k: 每个站点的邻居数
    :return: 邻接索引（PyG格式：edge_index）
    """
    from scipy.spatial import KDTree
    # 构建经纬度的KD树（快速找近邻）
    coords = np.vstack([lon, lat]).T
    tree = KDTree(coords)
    # 找每个站点的k个近邻
    distances, indices = tree.query(coords, k=k+1)  # +1是排除自身
    # 构建边索引（source -> target）
    source = np.repeat(np.arange(len(lon)), k)
    target = indices[:, 1:].flatten()  # 排除自身（第0个是自己）
    edge_index = np.vstack([source, target])
    return edge_index

# 示例：为前1000个站点构建邻接（全量4万+站点需调整k或降采样）
n_sample = 1000
edge_index = build_station_adjacency(lon_valid[:n_sample], lat_valid[:n_sample], k=8)
print(f"\nGNN邻接索引形状：{edge_index.shape}（适配PyTorch Geometric）")
print(f"前5条边：source={edge_index[0, :5]}, target={edge_index[1, :5]}")

# ====================== 7. 关闭数据集 ======================
ds.close()