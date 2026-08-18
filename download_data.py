import os
import cdsapi
c = cdsapi.Client()

import numpy as np

# 修复版：下载 ERA5、GTSM、GESLA、辅助数据
store_at = "Data2"  # 直接放在当前代码文件夹下的 Data2

dl_era5  = False
dl_gtsm  = True
dl_gesla = False
dl_aux   = False

if __name__ == "__main__":

    if not os.path.exists(store_at):
        os.makedirs(store_at)
    
    # ====================== 下载 ERA5 ======================
    if dl_era5:
        era5_dir = os.path.join(store_at, 'ERA5', 'stormSurge_hourly_79_18')
        if not os.path.exists(era5_dir):
            os.makedirs(era5_dir)
        print(f'下载 ERA5 到: {era5_dir}')

        var_long  = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']
        var_short = ['u10', 'v10', 'msl']

        start, end, bins = 2017, 2018, 13
        yearIntervals = np.array_split(np.arange(start, end+1), bins)

        for idx, interval in enumerate(yearIntervals):
            for jdx, var_name in enumerate(var_long):
                print(f'下载 {var_name} {interval}')
                
                target_file = os.path.join(era5_dir, f"{var_short[jdx]}_{idx+1:02d}.nc")

                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [var_name],
                        'month': ['01','02','03','04','05','06','07','08','09','10','11','12'],
                        'day': [f'{d:02d}' for d in range(1,32)],
                        'time': [f'{h:02d}:00' for h in range(24)],
                        'year': [str(year) for year in interval],
                    },
                    target_file
                )

    # ====================== 下载 GTSM ======================
    if dl_gtsm: 
        gtsm_dir = os.path.join(store_at, 'GTSM', 'reanalysis')
        if not os.path.exists(gtsm_dir):
            os.makedirs(gtsm_dir)
        print(f'下载 GTSM 到: {gtsm_dir}')

        for month in range(1, 13):
            print(f'下载月份 {month:02d}')
            
            zip_file = os.path.join(gtsm_dir, f'download{month:02d}.zip')

            c.retrieve(
                'sis-water-level-change-timeseries-cmip6',
                {
                    'variable': 'storm_surge_residual',
                    'experiment': 'reanalysis',
                    'temporal_aggregation': ['hourly'],
                    'year': [str(y) for y in range(1979, 2019)],
                    'month': f"{month:02d}",
                    'format': 'zip',
                    "version": ["v3"]
                },
                zip_file
            )
            os.system(f'unzip -o {zip_file} -d {gtsm_dir} && rm {zip_file}')
    
    # ====================== 下载 GESLA ======================
    if dl_gesla:
        gesla_dir = os.path.join(store_at, 'GESLA')
        if not os.path.exists(gesla_dir):
            os.makedirs(gesla_dir)
        print(f'GESLA 需要手动下载到: {gesla_dir}')

    # ====================== 下载辅助数据 ======================
    if dl_aux: 
        aux_dir = os.path.join(store_at, 'aux')
        if not os.path.exists(aux_dir):
            os.makedirs(aux_dir)
        print(f'下载辅助数据到: {aux_dir}')
