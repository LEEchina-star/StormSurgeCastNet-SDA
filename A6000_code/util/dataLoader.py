import os
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata

import shapely
import rasterio
import rioxarray
import xarray as xr
import geopandas as gpd
from global_land_mask import globe

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import dask
dask.config.set(scheduler='synchronous')

from util.GeslaDataset import gesla
from torch.utils.data import Dataset
from skimage.morphology import binary_dilation, binary_erosion

def try_sel_time(xar, q):
    try:    return xar.sel(date_time=q).date_time.values
    except: return np.datetime64('NaT')
            
def shape2raster(roi, shape_geo, r_size, fill_empty=0):
    transform = rasterio.transform.from_bounds(*roi.bounds, width=r_size, height=r_size)
    out_r  = np.full([r_size, r_size], float("inf"))
    raster = rasterio.features.rasterize(shape_geo,
                                        out_shape=2*[r_size],
                                        out=out_r,
                                        transform=transform)
    changed = out_r != np.full([r_size, r_size], float("inf"))
    changed_nums = raster[changed]
    changed_idx  = changed.nonzero()
    raster[raster==float("inf")] = fill_empty
    return raster, changed_nums, changed_idx

def rasterize_gauges(roi, r_size, gauges, fill_empty=0): 
    lat = gauges.latitude.values.reshape((-1,))
    lon = gauges.longitude.values.reshape((-1,))
    gauges_geo = shapely.points(coords=list(zip(lon, lat)))
    grid_zip = ((geom,value) for geom, value in zip(gauges_geo, gauges.sea_level.values.flatten()))
    raster_gesla, _, _ = shape2raster(roi, grid_zip, r_size, fill_empty)
    return raster_gesla

def rasterize_dense(ds, roi, r_size, val_key='surge', coord_x_key='station_x_coordinate', coord_y_key='station_y_coordinate', method='nearest'):
    values = ds[val_key].values
    if len(values.shape) == 1:
        values = values[None]
    transform = rasterio.transform.from_bounds(*roi.bounds, width=r_size, height=r_size)
    coords = np.array(list(zip(ds[coord_x_key].values, ds[coord_y_key].values)))
    pixel_coords = np.array(~transform * coords.T).T
    X, Y = np.meshgrid(np.arange(r_size), np.arange(r_size))
    rasters = []
    for t in range(values.shape[0]):
        rasters.append(griddata(pixel_coords, values[t], (X, Y), method=method))
    return np.stack(rasters, axis=0)

def unindexed_search_bounds(file, bounds, idx_key, x_key, y_key, expand=0):
    minx, miny, maxx, maxy = bounds
    in_bounds = (file[x_key] >= minx-expand) & (file[x_key] <= maxx+expand) \
              & (file[y_key] >= miny-expand) & (file[y_key] <= maxy+expand)
    idx = file[idx_key][in_bounds]
    data_in_range = [file.sel(**{idx_key: idx})]
    if minx < file[x_key].min():
        in_bounds = (file[x_key] >= minx-expand+360) & (file[x_key] <= maxx+expand+360) \
                  & (file[y_key] >= miny-expand)     & (file[y_key] <= maxy+expand)
        idx = file[idx_key][in_bounds]
        extra = file.sel(**{idx_key: idx})
        extra.assign_coords(**{x_key: file[x_key]-360})
        data_in_range.append(extra)
    if maxx > file[x_key].max():
        in_bounds = (file[x_key] >= minx-expand-360) & (file[x_key] <= maxx+expand-360) \
                  & (file[y_key] >= miny-expand)     & (file[y_key] <= maxy+expand)
        idx = file[idx_key][in_bounds]
        extra = file.sel(**{idx_key: idx})
        extra.assign_coords(**{x_key: file[x_key]+360})
        data_in_range.append(extra)
    return xr.merge(data_in_range)

def indexed_search_bounds(file, bounds, x_key, y_key, expand=0):
    minx, miny, maxx, maxy = bounds
    data_in_range = [file.sel(**{
        x_key: slice(minx-expand, maxx+expand),
        y_key: slice(maxy+expand, miny-expand)
    }).compute()]
    if minx < file[x_key].min():
        extra = file.sel(**{
            x_key: slice(minx-expand+360, maxx+expand+360),
            y_key: slice(maxy+expand, miny-expand)
        }).compute()
        extra = extra.assign_coords(**{x_key: extra[x_key] - 360})
        data_in_range.insert(0, extra)
    if maxx > file[x_key].max():
        extra = file.sel(**{
            x_key: slice(minx-expand-360, maxx+expand-360),
            y_key: slice(maxy+expand, miny-expand)
        }).compute()
        extra = extra.assign_coords(**{x_key: extra[x_key] + 360})
        data_in_range.append(extra)
    return xr.concat(data_in_range, x_key)

def gtsm_in_bounds(gtsm, bounds, impute_value=None):
    subset = unindexed_search_bounds(gtsm, bounds, 'stations', 'station_x_coordinate', 'station_y_coordinate')
    if impute_value is None:
        impute_value = np.nanmean(subset.surge.values)
    nanfree_subset = subset.fillna(impute_value)
    return nanfree_subset

def resize_box(shp, res=0.1, width=256):
    minx, miny, maxx, maxy = shp.bounds
    con = width//2
    x, y = (minx + maxx)/2, (miny + maxy)/2
    lg_minx, lg_maxx = x-res*con, x+res*con
    lg_miny, lg_maxy = y-res*con, y+res*con
    return shapely.box(lg_minx, lg_miny, lg_maxx, lg_maxy)

toTimestamp = lambda datestr, sep: datetime.strptime(datestr.split(' ')[0], f"%Y{sep}%m{sep}%d")

def sample_roi(rng, stations, n_samples=10, scale=1, res=0.1, con=128):
    sub_stations = rng.choice(stations, size=n_samples, replace=True)
    sample_centers = np.full([n_samples, 2], np.nan)
    for sample in range(0, n_samples):
        invalid = False
        mean_sub_sample = [sub_stations[sample].x, sub_stations[sample].y]
        draws = rng.multivariate_normal(mean=mean_sub_sample , cov=[[scale**2, 0], [0, scale**2]], size=50)
        drawsF = draws[(-180 <= draws[:,0]) & (draws[:,0] <= 180) & (-90 <= draws[:,1]) & (draws[:,1] <= 90)]
        is_in_ocean = globe.is_ocean(drawsF[:,1], drawsF[:,0])
        drawsO = drawsF[is_in_ocean]
        if len(drawsO) == 0: 
            sample_centers[sample, :] = mean_sub_sample
        else: 
            sample_centers[sample, :] = drawsO[0]
    rois = roi_around(sample_centers, res, con)
    return rois

def roi_around(centroids, res=0.1, con=128, incl_center=False):
    dist = con*res
    if incl_center:
        dist += 0.5*res
    rois = [shapely.geometry.box(x-dist, y-dist, x+dist, y+dist, ccw=True) for x, y in centroids]
    return rois

def get_gtsm_database(root_path, years=None):
    gtsm_path  = os.path.join(root_path, 'GTSM', '')
    paths    = glob.glob(os.path.join(gtsm_path, 'reanalysis_surge_hourly_*_v3.nc'))
    if years is not None:
        y0, y1 = years
        paths = [p for p in paths if y0 <= int(os.path.basename(p).split('_')[3]) <= y1]
    if len(paths) == 0:
        raise FileNotFoundError(f'No GTSM files found in {gtsm_path} (years={years})')
    rootgrp  = xr.open_mfdataset(paths, combine='by_coords', parallel=False, data_vars='minimal', coords='minimal', compat='override',engine='netcdf4')
    t0_data  = rootgrp.surge.sel(time=rootgrp.time[0]) 
    coords   = zip(t0_data.station_x_coordinate.values, t0_data.station_y_coordinate.values)
    stations = shapely.points(coords=list(coords))
    return rootgrp, stations

def get_raster_resample(file, roi, r_size, pixel_size, bands=None, method="nearest", x_key="x", y_key="y"):
    data_in_range = indexed_search_bounds(file, roi.bounds, x_key, y_key, expand=pixel_size)
    minx, miny, maxx, maxy = roi.bounds
    i_kwargs = {"method": method, "kwargs": {"fill_value": "extrapolate"}}
    resampled = data_in_range \
        .interp(**{x_key: np.linspace(minx, maxx, r_size), **i_kwargs}) \
        .interp(**{y_key: np.linspace(maxy, miny, r_size), **i_kwargs})
    if bands is not None:
        raster = np.stack([resampled[band] for band in bands], axis=1)
    else:
        raster = resampled.values
    return raster

def get_era5(file, roi, r_size, method="nearest"):
    return get_raster_resample(file, roi, r_size, pixel_size=0.25, bands=["msl", "u10", "v10"], method=method, x_key="longitude", y_key="latitude")

def get_lsm(mask, roi, r_size, pixel_size):
    mask = get_raster_resample(mask, roi, r_size, pixel_size=pixel_size)
    mask = binary_erosion(binary_erosion(binary_erosion(binary_dilation(binary_dilation(binary_dilation(mask))))))
    return mask
    

class coastalLoader(Dataset):
    def __init__(self, root, split="all", hyperlocal=False, splits_ids=None, stats=None, stats_ibtracs=None, input_len=12, context_window=128, res=0.025, drop_in=0.25, lead_time=None, center_gauge=False, only_series=False, no_gesla_context=False, seed=None, gtsm_years=(1993, 2018)):
        
        assert split in ['all', 'train', 'val', 'test']
        self.start = datetime(year=1993, month=1, day=1)

        self.root_dir       = root              
        self.split          = split
        self.hyperlocal     = hyperlocal

        self.dbg            = False
        self.only_series    = only_series
        self.no_gesla_context = no_gesla_context
        self.res            = res               
        self.input_length   = input_len         
        self.lead_time      = lead_time         
        self.center_gauge   = center_gauge      
        self.max_rnd_lead   = 12
        self.con            = context_window    
        self.drop_in        = drop_in           
        self.close_thresh   = 100
        self.gtsm_years     = gtsm_years

        self.fract_holdout = 0.2                     
        self.fract_val     = 0.1                     

        self.splits_ids  = splits_ids
        # ===================== 已改为 1993 =====================
        self.total_start, self.holdout_start, self.total_end = '1993-01-01', '2014-04-03', '2018-12-31'
        self.start_date  = self.holdout_start if split=='test' else self.total_start
        self.end_date    = self.total_end if split=='test' else '2014-04-01' 
        if split=='all': self.start_date, self.end_date = self.total_start, self.total_end 

        self.min_rec_len = self.input_length + int(np.nanmax([np.nan if self.lead_time is None else self.lead_time, self.max_rnd_lead]))

        print(f"Setting up {self.split} split:")

        gesla_path, gesla_meta, self.pts_gesla, ids_gesla, self.meta = self.get_gesla3_coords()
        self.geslaD  = gesla.GeslaDataset(gesla_meta, gesla_path)
        self.ncGESLA = xr.open_dataset(filename_or_obj=os.path.join(self.root_dir, 'combined_gesla_surge.nc'), engine='netcdf4')
        
        missing_in_xr = set(self.meta.index.values) - set(np.intersect1d(self.ncGESLA.station.values, self.meta.index.values))
        missing_in_xr = missing_in_xr.union(set([self.meta[self.meta['FILE NAME'] == 'stouts_creek-8533365-usa-noaa'].index.values[0], 
                                                 self.meta[self.meta['FILE NAME'] == 'brighton-btn-gbr-cco'].index.values[0],
                                                 self.meta[self.meta['FILE NAME'] == 'moda-8775283-usa-noaa'].index.values[0],
                                                 self.meta[self.meta['FILE NAME'] == 'ucluelet_bc-8595-can-meds'].index.values[0],
                                                 self.meta[self.meta['FILE NAME'] == 'constanta-con-rou-cmems'].index.values[0]
                                                 ]))
        self.meta.drop(missing_in_xr, inplace=True)
        missing_in_meta = set(self.ncGESLA.station.values) - set(np.intersect1d(self.ncGESLA.station.values, self.meta.index.values))
        self.ncGESLA    = self.ncGESLA.sel(station=list(set(self.ncGESLA.station.values)-set(missing_in_meta)))
        
        self.compute_stats_ibtracs(stats_ibtracs, compute=True)
        
        if self.splits_ids is None: self.get_splits()

        if split=='all': self.split_idx     = self.meta.index.values
        if split=='train': self.split_idx   = self.splits_ids['train']
        if split=='val': self.split_idx     = self.splits_ids['val']
        if split=='test': self.split_idx    = self.splits_ids['test']

        self.split_idx    = np.intersect1d(self.ncGESLA.station.values, self.split_idx)
        self.split_gesla  = self.pts_gesla[self.split_idx]

        self.ncGESLA      = self.ncGESLA.sel(date_time=slice(self.start_date, self.end_date), station=self.split_idx)
        try:
            print(f'Loading GESLA data into memory at {datetime.now()}')
            self.ncGESLA = self.ncGESLA.load()
            print(f'Done loading GESLA data at {datetime.now()}')
        except:
            chunks={'date_time': (int(self.end_date[:4])-int(self.start_date[:4]))*12 * 24, 'station': int(len(self.ncGESLA.station)/100)}            
            self.ncGESLA = self.ncGESLA.chunk(chunks=chunks)
        
        outlier_magnitude   = 2
        self.surge_mean     = self.ncGESLA.mean(dim='date_time', skipna=True).sea_level.values
        self.surge_std      = self.ncGESLA.std(dim='date_time', skipna=True).sea_level.values
        self.outliers_bool  = (self.ncGESLA.sea_level > np.expand_dims((self.surge_mean + outlier_magnitude*self.surge_std), 1))
        self.outliers_data  = self.ncGESLA.where(self.outliers_bool.compute(), drop=True)

        n_nanmean, n_nanstd = np.isnan(self.surge_mean).sum(), np.isnan(self.surge_std).sum()
        if n_nanmean:
            print(f'Detected {n_nanmean} NaN entries in mean statistics, data may be missing. Imputing with 0.')
            self.surge_mean[np.isnan(self.surge_mean)] = 0
        if n_nanstd:
            print(f'Detected {n_nanstd} NaN entries in standard deviation statistics, data may be missing. Imputing with 1.')
            self.surge_mean[np.isnan(self.surge_mean)] = 1

        self.ncGTSM = None
        self.ncERA5 = None

        mask_path = os.path.join(self.root_dir, 'aux', 'landWater2020_1000m.tif')
        self.mask = rioxarray.open_rasterio(mask_path)
        try:
            self.mask = self.mask.load()
        except:
            chunks  = {'y': 22, 'x': 43}
            self.mask = self.mask.chunk(chunks=chunks)

        self.n_samples = len(self.split_gesla)

        if seed is None:
            self.hashes = [None]  * self.n_samples
        else:
            rng = np.random.default_rng(seed)
            self.hashes = [rng.integers(1000000, 9999999) for _ in range(self.n_samples)]

        self.compute_stats(stats, compute=False)

        self.storm_dates, self.storm_count = 0, 0

        print(f'Done setting up {self.split} split.\n')


    def __getitem__(self, pdx):

        if self.hashes[pdx] is None:
            rng = np.random.default_rng(np.random.randint(10000, 99999))
        else:
            rng = np.random.default_rng(self.hashes[pdx])
        
        if self.ncGTSM is None:
            self.ncGTSM, self.stations = get_gtsm_database(self.root_dir, years=self.gtsm_years)
            era5_path = os.path.join(self.root_dir, 'ERA5')
            paths     = sorted(glob.glob(os.path.join(era5_path, '**', '*.nc'), recursive=True))
            if len(paths) == 0:
                print('[dataLoader] WARNING: no ERA5 files found under', era5_path,
                      '-> substituting zeros (normalized mean) for the smoke test / missing data runs.')
                self.ncERA5 = None
            else:
                self.ncERA5 = xr.open_mfdataset(paths)

        gauge_id     = self.split_idx[pdx] 
        gauge_coords = self.split_gesla[pdx]

        n_samples, scale = 1, 1
        if self.center_gauge:
            roi = roi_around([(gauge_coords.x, gauge_coords.y)], self.res, self.con, incl_center=True)[0]
            r_size = 2*self.con + 1
        else:
            roi = sample_roi(rng, [gauge_coords], n_samples, scale, self.res, self.con)[0]
            r_size = 2*self.con
        minx, miny, maxx, maxy = roi.bounds

        mask = get_lsm(self.mask, roi, r_size, pixel_size=0.025)

        lead_ranges     = np.arange(0, self.max_rnd_lead+2)
        lead_time       = self.lead_time if self.lead_time is not None else rng.choice(lead_ranges, 1, p=lead_ranges[::-1]/lead_ranges.sum())[0]
        len_timeseries  = self.input_length + lead_time

        target_time = np.datetime64(self.start_date)+np.timedelta64(len_timeseries,'h')

        if self.split in ['val', 'test']:
            qdx = np.where(self.meta.index.values==gauge_id)[0][0]
            is_close = self.stats_ibtracs['thresh'].values[:, qdx]
            close_times = self.stats_ibtracs['times'][is_close]
            valid_times = close_times[~np.isnan(close_times)]       
            gauge_id_xr = self.ncGESLA.sel(station=gauge_id)  
            times = np.array([try_sel_time(gauge_id_xr, t) for t in valid_times])
            try:
                close_enough = times[~np.isnat(times)]  
                close_enough = close_enough[~np.isnan(self.ncGESLA.sel(station=gauge_id, date_time=close_enough).sea_level.values)]
                target_time     = rng.choice(close_enough).astype('datetime64[h]')
                self.storm_dates += len(close_enough)
                self.storm_count += 1
            except: 
                close_enough = []
            
        if self.split in ['all', 'train'] or len(close_enough)==0:
            gauge_id_t = self.ncGESLA.sel(station=gauge_id)
            prob_to_sample_extremes = 0.5
            sample_extremes         = rng.uniform(0, 1) > (1-prob_to_sample_extremes)
            try:
                if not sample_extremes: raise Exception
                gauge_id_outlier_t = gauge_id_t.date_time[self.outliers_bool.sel(station=gauge_id).values]
                if not len(gauge_id_outlier_t.date_time): raise Exception
                target_time = gauge_id_outlier_t.isel(date_time=rng.choice(range(0, len(gauge_id_outlier_t)), 1))[0].values
            except:
                gauge_id_t_dropna  = gauge_id_t.dropna(dim='date_time')
                if len(gauge_id_t_dropna.date_time) > len_timeseries:
                    target_time = gauge_id_t_dropna.isel(date_time=rng.integers(len_timeseries, high=len(gauge_id_t_dropna.date_time), size=1)).date_time[0].values
                else: 
                    print(f"Time series (index: {pdx}, ID: {gauge_id}) is too short. SKIPPING.")
                    return None

        target_time = max(np.datetime64(self.start_date)+np.timedelta64(len_timeseries,'h'), target_time)
        try:
            converted_target_t_idx  = self.ncGESLA.indexes["date_time"].get_loc(target_time)
        except:
            return None

        start_time              = self.ncGESLA.date_time[converted_target_t_idx-len_timeseries+1]
        end_time                = self.ncGESLA.date_time[converted_target_t_idx-lead_time]

        gesla_series_in  = self.ncGESLA.sel(date_time=slice(start_time, end_time), station=gauge_id).sea_level.values
        if len(gesla_series_in) != self.input_length:
            return None

        gesla_series_in  = (gesla_series_in-self.stats['mean']['GESLA'])/self.stats['std']['GESLA']
        gesla_series_out = self.ncGESLA.sel(date_time=target_time, station=gauge_id).sea_level.values
        gesla_series_out = ((gesla_series_out-self.stats['mean']['GESLA'])/self.stats['std']['GESLA'])

        if self.only_series:
            gauges_in = self.ncGESLA.sel(date_time=slice(start_time, end_time), station=gauge_id)
            sample = {'input': {'series': gesla_series_in[:, None].astype('float32'),
                                'ls_mask': mask.astype('float32'),
                                'dates2int': pd.to_datetime(gauges_in.date_time).astype(np.int64).values,
                                'td': np.array((pd.to_datetime(gauges_in.date_time) - self.start).total_seconds() // 3600, np.float32),                                                                                            
                                'td_lead': np.array(lead_time).astype('float32'),                                                                                    
                                'lon': gauges_in.longitude.values,
                                'lat': gauges_in.latitude.values
                                },
                    'target': {'series': gesla_series_out.reshape((1,1)).astype('float32'),
                                'id': np.array(gauge_id)
                                },
                        }
            return sample
    
        gauges_in, gauges_target, coarse_in, coarse_target, weather_in = self.get_data(start_time, end_time, target_time)
        try:
            coarse_in = coarse_in.compute()
            coarse_target = coarse_target.compute()
        except:
            return None

        if self.no_gesla_context:
            in_idx = set([gauge_id])
            subset_gauges_in = gauges_in.sel(station=gauge_id)
        else:
            sub_meta_in     = self.geslaD.meta.iloc[gauges_in.station.values]
            in_bound        = (sub_meta_in.longitude>=minx) & (sub_meta_in.longitude<=maxx) & (sub_meta_in.latitude>=miny) & (sub_meta_in.latitude<=maxy)
            in_gauge_idx     = sub_meta_in[in_bound].index.to_numpy()

            if self.split == 'train':
                in_idx     = np.array(set(in_gauge_idx) - set(self.splits_ids['val']) - set(self.splits_ids['test']))
                in_idx     = in_gauge_idx[rng.uniform(0, 1, len(in_gauge_idx)) > self.drop_in]
            elif self.split in ['val', 'test'] and not self.hyperlocal:
                if self.split == 'val': in_idx = np.array(set(in_gauge_idx) - set(self.splits_ids['test']))
                in_idx      = list(set(in_gauge_idx) - set([gauge_id]))
            else:
                in_idx    = in_gauge_idx
            subset_gauges_in = gauges_in.sel(station=in_idx)

        if self.split in ['val', 'test'] or self.no_gesla_context:
            target_bound, select_target_gauges = np.ones(1, bool), np.array([gauge_id])
        else:
            sub_meta_target     = self.geslaD.meta.iloc[gauges_target.station.values]
            target_bound        = (sub_meta_target.longitude>=minx) & (sub_meta_target.longitude<=maxx) & (sub_meta_target.latitude>=miny) & (sub_meta_target.latitude<=maxy)
            select_target_gauges = sub_meta_target[target_bound].index.to_numpy()
        subset_gauges_target = gauges_target.sel(station=select_target_gauges)

        if len(in_idx) == 0:
            gesla_in  = np.full((self.input_length, r_size, r_size), np.nan)
        else:
            try:
                gesla_in  = np.array([rasterize_gauges(roi, r_size, subset_gauges_in.sel(date_time=gin), fill_empty=np.nan) for gin in subset_gauges_in.date_time], np.float32)
            except:
                return None

        if target_bound.sum() == 0: 
            gesla_out = np.full((1, r_size, r_size), np.nan)
        else:
            try:
                gesla_out = np.array([rasterize_gauges(roi, r_size, subset_gauges_target, fill_empty=np.nan)], np.float32)
            except:
                return None

        if self.center_gauge:
            gesla_in[:, self.con, self.con]  = gauges_in.sel(station=gauge_id).sea_level.values.flatten()
            gesla_out[:, self.con, self.con] = gauges_target.sel(station=gauge_id).sea_level.values.flatten()

        lg_box               = resize_box(roi, self.res, width=256)
        subset_coarse_in     = gtsm_in_bounds(coarse_in, lg_box.bounds)
        subset_coarse_target = gtsm_in_bounds(coarse_target, lg_box.bounds)

        try:
            gtsm_in = rasterize_dense(subset_coarse_in, roi, r_size)
            gtsm_out = rasterize_dense(subset_coarse_target, roi, r_size)
        except:
            gtsm_in  = np.zeros((self.input_length, r_size, r_size))
            gtsm_out = np.zeros((1, r_size, r_size))
            mask     = np.ones_like(mask, bool)
        gtsm_out_unmasked = gtsm_out.copy()

        if self.ncERA5 is not None:
            try:
                era5_in  = get_era5(weather_in, roi, r_size).astype(np.float32)
            except:
                return None
        else:
            era5_in = np.zeros((self.input_length, 3, r_size, r_size), np.float32)

        td_in   = np.array((pd.to_datetime(gauges_in.date_time) - self.start).total_seconds() // 3600, np.float32)
        loc_lon, loc_lat   = tuple(roi.centroid.coords)[0]

        era5_mean   = np.stack([self.stats['mean']['ERA5']['msl'], self.stats['mean']['ERA5']['u10'], self.stats['mean']['ERA5']['v10']])[None,:,None,None]
        era5_std    = np.stack([self.stats['std']['ERA5']['msl'], self.stats['std']['ERA5']['u10'], self.stats['std']['ERA5']['v10']])[None,:,None,None]

        gesla_in    = (gesla_in-self.stats['mean']['GESLA'])/self.stats['std']['GESLA']
        nan_mask    = np.isnan(gesla_in)
        gesla_in[nan_mask] = 0

        mask[~nan_mask.min(axis=0,keepdims=True)] = False
        gtsm_out[mask]  = np.nan

        sample = {'input': {'era5': ((era5_in-era5_mean)/era5_std).astype('float32'),                                       
                            'sparse': gesla_in[:,None].astype('float32'),
                            'series': gesla_series_in[:, None].astype('float32'),
                            'gtsm': ((gtsm_in-self.stats['mean']['GTSM'])/self.stats['std']['GTSM'])[:,None,...].astype('float32'),
                            'ls_mask': mask.astype('float32'),
                            'valid_mask': (1-nan_mask).astype('float32')[:,None,...].astype('float32'),
                            'td': td_in,
                            'td_lead': np.array(lead_time).astype('float32'),
                            'lon': loc_lon,
                            'lat': loc_lat,
                            },
                'target': {'sparse': ((gesla_out-self.stats['mean']['GESLA'])/self.stats['std']['GESLA']).astype('float32'),
                           'series': gesla_series_out.reshape((1,1)).astype('float32'),
                           'gtsm': ((gtsm_out-self.stats['mean']['GTSM'])/self.stats['std']['GTSM']).astype('float32'),   
                           'gtsm_out_unmasked': ((gtsm_out_unmasked-self.stats['mean']['GTSM'])/self.stats['std']['GTSM']).astype('float32'),
                           'id': np.array(gauge_id),
                           'lon_gauge': gauges_target.sel(station=gauge_id).longitude.values,
                           'lat_gauge': gauges_target.sel(station=gauge_id).latitude.values
                           },
                    }
        
        # ===================== 终极修复：任何异常/空数据直接返回None =====================
        if sample['input']['series'].shape[0] != self.input_length:
            return None
        if 'era5' in sample['input'] and sample['input']['era5'].shape[0] != self.input_length:
            return None
        
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def get_data(self, start_time, end_time, target_time):
        gauges_in     = self.ncGESLA.sel(date_time=slice(start_time, end_time))
        gauges_target = self.ncGESLA.sel(date_time=target_time)
        coarse_in     = self.ncGTSM.sel(time=slice(start_time, end_time))
        coarse_target = self.ncGTSM.sel(time=target_time)
        weather_in = self.ncERA5.sel(time=slice(start_time, end_time)) if self.ncERA5 is not None else None
        return gauges_in, gauges_target, coarse_in, coarse_target, weather_in
    
    
    def get_gesla3_coords(self):
        gesla_path = os.path.join(self.root_dir, 'GESLA', 'GESLA3', '')
        gesla_meta = os.path.join(self.root_dir, 'GESLA', 'GESLA3_ALL 2.csv')
        gdf = gpd.read_file(gesla_meta)
        pts_gesla = shapely.points(coords=[(float(gdf['LONGITUDE'][idx]), float(gdf['LATITUDE'][idx])) for idx in gdf.index])
        ids_gesla = np.array([idx for idx in gdf.index])
        gdf = gdf[gdf['GAUGE TYPE']=='Coastal']
        gdf = gdf[~(np.array([toTimestamp(tdx, '/') for tdx in gdf['END DATE/TIME']])    < timedelta(hours=self.min_rec_len) + toTimestamp(self.total_start, '-'))]
        gdf = gdf[~(np.array([toTimestamp(tdx, '/') for tdx in gdf['START DATE/TIME']])  > toTimestamp(self.total_end, '-'))]
        return gesla_path, gesla_meta, pts_gesla, ids_gesla, gdf

    def compute_stats(self, stats, compute=True):
        if stats is not None:
            self.stats = stats
        else:
            if compute and self.split == 'train':
                    print('Computing mean and std statistics on the train split.')
                    sub_p, lon, lat, t = 0.001, len(self.ncERA5.longitude), len(self.ncERA5.latitude), len(self.ncERA5.time)
                    sub_ERA5 = self.ncERA5.isel(longitude=np.linspace(0, lon-1, num=int(sub_p*lon), dtype=int),latitude=np.linspace(0, lat-1, num=int(sub_p*lat), dtype=int), time=np.linspace(0, t-1, num=int(sub_p*t), dtype=int))
                    print(f'Computing GESLA statistics.')
                    mean_GESLA  = float(self.ncGESLA.sea_level.mean(skipna=True).compute().values)
                    std_GESLA   = float(self.ncGESLA.sea_level.std(skipna=True).compute().values)
                    print(f'Computing GTSM statistics.')
                    mean_GTSM   = float(self.ncGTSM.surge.mean(skipna=True).compute().values)
                    std_GTSM    = float(self.ncGTSM.surge.std(skipna=True).compute().values)
                    print(f'Computing ERA5 statistics.')
                    mean_msl    = float(sub_ERA5.msl.mean(skipna=True).compute().values)
                    mean_u10    = float(sub_ERA5.u10.mean(skipna=True).compute().values)
                    mean_v10    = float(sub_ERA5.v10.mean(skipna=True).compute().values)
                    std_msl     = float(sub_ERA5.msl.std(skipna=True).compute().values)
                    std_u10     = float(sub_ERA5.u10.std(skipna=True).compute().values)
                    std_v10     = float(sub_ERA5.v10.std(skipna=True).compute().values)
                    
                    self.stats = {'mean': {'GESLA': mean_GESLA,
                                           'GTSM': mean_GTSM,
                                           'ERA5': {'msl': mean_msl,
                                                    'u10': mean_u10,
                                                    'v10': mean_v10}},
                                  'std': {'GESLA': std_GESLA,
                                          'GTSM': std_GTSM,
                                          'ERA5': {'msl': std_msl,
                                                   'u10': std_u10,
                                                   'v10': std_v10}}}
                    print(f'Done computing statistics.')
            else:
                self.stats = {'mean': {'GESLA': 0, 'GTSM': 0, 'ERA5': {'msl': 0, 'u10': 0, 'v10': 0}},
                              'std':  {'GESLA': 1, 'GTSM': 1, 'ERA5': {'msl': 1, 'u10': 1, 'v10': 1}}}
                
   
    def compute_stats_ibtracs(self, stats_ibtracs, compute=True):
        if stats_ibtracs is not None:
            self.stats_ibtracs = stats_ibtracs
        elif compute:
                print('Computing coincidences of IBTrACS with tidal gauges.')
                self.ncGESLA = self.ncGESLA.sel(date_time=slice(self.total_start, self.total_end))
                ibtracs      = xr.open_dataset(filename_or_obj= os.path.join(self.root_dir, 'aux', 'IBTrACS.ALL.v04r00.nc'), engine='netcdf4')
                self.ncGESLA = self.ncGESLA.load()
                ibtracs      = ibtracs.load()

                time_filter   = np.logical_and(ibtracs.time>=np.datetime64(self.total_start) + np.timedelta64(10, 'D'), ibtracs.time<np.datetime64(self.total_end) - np.timedelta64(10, 'D')) 
                self.ibtracs  = ibtracs.where(time_filter, drop=False).dropna(dim='storm', how='all')
                s_x_t_matrix = np.array([(lon, lat) for lon, lat in zip(self.ibtracs.lon.values, self.ibtracs.lat.values)]).swapaxes(1,-1)
                nan_idx      = np.isnan(s_x_t_matrix).mean(axis=-1).astype(bool)
                s_x_t_points = shapely.points(coords=[tuple(row) for row in s_x_t_matrix[~nan_idx,:]])
                gesla_points = shapely.points(coords=[(lon[~np.isnan(lon)][0], lat[~np.isnan(lat)][0]) for lon, lat in zip(self.ncGESLA.longitude.values, self.ncGESLA.latitude.values)])
                gdf_1, gdf_2 = gpd.GeoDataFrame(geometry=s_x_t_points), gpd.GeoDataFrame(geometry=gesla_points)
                dist_gauge_x_s_x_t = gdf_1.geometry.apply(lambda g: gdf_2.distance(g))
                thresh_mat        = dist_gauge_x_s_x_t*100 < self.close_thresh
                s_x_t_times       = self.ibtracs.time.values[~nan_idx].astype('datetime64[h]')
                self.stats_ibtracs = {'thresh': thresh_mat, 'times': s_x_t_times}

    def get_splits(self):
        print('Computing assignmend of gauge sites to splits.')
        n_total       = len(self.meta)
        n_holdout     = int(n_total * self.fract_holdout)
        n_val         = int((n_total - n_holdout) * self.fract_val)
        n_train       = int(n_total-n_holdout-n_val)

        put_to_holdout = self.meta[np.array([toTimestamp(tdx, '/') >= toTimestamp(self.holdout_start, '-') - timedelta(days=1) for tdx in self.meta['START DATE/TIME']])].index.values
        put_to_holdout = np.unique(list(put_to_holdout)+list([3161, 3165, 3169, 3173, 3202, 3206, 3208, 3203, 3219, 3220, 3225, 3250, 3259, 3256, 3269, 3345, 3355, 3361, 
                                                              3373, 3374, 3380, 3399, 1416, 1604, 1908, 1913, 1930, 1931, 1932, 1938, 1943, 1949, 1950, 1951, 1961, 1965, 
                                                              1966, 1967, 1976, 1979, 1999, 2054, 2091, 2149, 2168, 2193, 2237, 2259, 2353, 4856]))
        end_b4_holdout = self.meta[np.array([toTimestamp(tdx, '/') < timedelta(hours=self.min_rec_len) + toTimestamp(self.holdout_start, '-') for tdx in self.meta['END DATE/TIME']])].index.values
        n_test_left    = n_holdout - len(put_to_holdout)

        if self.stats_ibtracs is not None:
            time_replicate  = np.repeat(self.stats_ibtracs['times'][:,None], self.stats_ibtracs['thresh'].shape[-1], axis=-1)        
            hits_after_holdout_start = time_replicate > toTimestamp(self.holdout_start, '-')
            time_replicate[~self.stats_ibtracs['thresh'] | ~hits_after_holdout_start] = None
            rowIsAllNaN_mask = np.all(np.isnan(time_replicate), axis=1)
            time_replicate = time_replicate[~rowIsAllNaN_mask]
            filter_gauges = False*np.ones_like(time_replicate).astype(bool)
            filter_gauges[:, [idx in end_b4_holdout for idx in self.meta.index.values]] = True
            time_replicate[filter_gauges] = None
            hits_per_gauge  = (~np.isnat(time_replicate)).sum(axis=0)
            hitmost_idx     = list(np.argsort(hits_per_gauge)[::-1])
            hitmost_idx     = self.meta.iloc[hitmost_idx].index.values
            filter_putTo    = sorted(set(hitmost_idx) - set(put_to_holdout), key=list(hitmost_idx).index)
            add2test_ids    = sorted(filter_putTo[:n_test_left])

        else:
            holdout_left2draw = set(self.meta.index.values) - set(end_b4_holdout) - set(put_to_holdout)
            add2test_ids      = random.sample(list(holdout_left2draw), n_test_left)

        test_idx      = np.array(sorted(list(put_to_holdout) + add2test_ids))
        noHoldout_idx = set(self.meta.index.values) - set(test_idx)
        val_idx       = np.array(sorted(random.sample(list(noHoldout_idx), n_val)))
        train_idx     = np.array(sorted(list(set(noHoldout_idx)-set(val_idx))))
        
        self.splits_ids = {'all': self.meta.index.values, 'train': train_idx, 'val': val_idx, 'test': test_idx,
                           'total_start': self.total_start, 'holdout_start': self.holdout_start, 'total_end': self.total_end}