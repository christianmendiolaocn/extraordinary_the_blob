from typing import List, Tuple, Optional, Dict
import xarray as xr
import seaborn as sns
import numpy as np
import xarray as xr
from functools import wraps
from scipy.interpolate import interp1d
from skimage.measure import find_contours
from haversine import haversine, Unit
from skimage.morphology import convex_hull_image
from typing import List, Tuple
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import pickle
import json
import inspect
import LENSfunctions as funcs
import measuresfuncs as measures
import warnings
def log_execution_time(toggle_attr='use_decorators'):
    """Decorator to log execution time, which can be toggled on/off using a class attribute."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, toggle_attr, True):  # Check if decorators are enabled
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
                return result
            else:
                return func(self, *args, **kwargs)  # Run function normally if disabled
        return wrapper
    return decorator

class ShapeMeasures:
    def __init__(self, lat_resolution: float = 110.574, lon_resolution: float = 111.320, use_decorators: bool = True):
        """
        A class to compute shape-based metrics for geospatial objects.

        Parameters:
            lat_resolution (float): Resolution of latitude in km per degree.
            lon_resolution (float): Resolution of longitude in km per degree at the equator.
            use_decorators (bool): Toggle for using decorators like execution time logging.
        """
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.use_decorators = use_decorators  # Control decorator execution
    
    @log_execution_time()
    def calculate_area(self, lats: List[float], lons: List[float]) -> float:
        """Computes area in square kilometers."""
        y, x = np.array(lats), np.array(lons)
        dlon = np.cos(np.radians(y)) * self.lon_resolution
        dlat = self.lat_resolution * np.ones(len(dlon))
        return np.sum(dlon * dlat)

    @log_execution_time()
    def calculate_spatial_extents(self, one_obj: xr.Dataset) -> dict:
        """Computes spatial extents and summary statistics."""
        spatial_extents = []
        coords_full = []

        for i in range(len(one_obj.time)):
            stacked = one_obj[i, :, :].stack(zipcoords=['lat', 'lon'])
            intermed = stacked.dropna(dim='zipcoords').zipcoords.values

            if len(intermed) == 0:
                coords_full.append([])
                spatial_extents.append(0.0)
                continue

            lats, lons = zip(*intermed)
            coords = list(zip(lats, lons))
            coords_full.append(coords)
            spatial_extents.append(self.calculate_area(lats, lons))
            
        return {
            'coords_full': coords_full,
            'spatial_extents': spatial_extents,
            'max_spatial_extent': max(spatial_extents, default=0.0),
            'max_spatial_extent_time': np.argmax(spatial_extents) if spatial_extents else -1,
            'mean_spatial_extent': np.mean(spatial_extents) if spatial_extents else 0.0,
            'cumulative_spatial_extent': np.sum(spatial_extents) if spatial_extents else 0.0,
        }

def intensity_measures(sstafilepath , mhwfilepath):

    ens_memb_ind = range(100)
    da_ssta = xr.open_mfdataset(sstafilepath, combine='nested', concat_dim='ensemble_member')
    da_mhwobj = xr.open_mfdataset(mhwfilepath, combine='nested', concat_dim='ensemble_member')
    
    ssta_mean_data_ls_all = []
    ssta_max_data_ls_all = []
    ssta_90percentile_data_ls_all = []
    ssta_90percentile_test_data_ls_all = []
    ssta_stdpertimestep_data_ls_all = []
    ssta_std_data_ls_all = []
    
    for ens_memb in ens_memb_ind:
        print('ENS MEMB:',ens_memb)
    
        ssta_mean_data_ens_ls_all = []
        ssta_max_data_ens_ls_all = []
        ssta_90percentile_ens_data_ls_all = []
        ssta_90percentile_test_ens_data_ls_all = []
        ssta_stdpertimestep_data_ens_ls_all = []
        ssta_std_data_ens_ls_all = []
        ssta_notrend = da_ssta.__xarray_dataarray_variable__[ens_memb,:,:,:].compute()
        blobs = da_mhwobj.labels[ens_memb,:,:,:].compute()
        unique_labels = np.unique(blobs.max(dim=('lat','lon')).data)[:-1]
    
        for object_id in unique_labels:
            print(' OBJ ID:', object_id)
    
            object_count_per_time = (blobs == object_id).sum(dim=['lat', 'lon'])
            true_time_steps = object_count_per_time.time.where(object_count_per_time > 0, drop=True)
            one_obj = blobs.sel(time=true_time_steps.time)
            one_obj_ones = xr.where(one_obj > 0, 1., 0)
            one_obj_ssta = ssta_notrend.sel(time=true_time_steps.time)
            masked_one_obj_ssta = one_obj_ssta*one_obj_ones
            masked_one_obj_ssta_nans = xr.where(masked_one_obj_ssta >0, masked_one_obj_ssta, np.nan)
    
            mean_ssta = masked_one_obj_ssta_nans.mean(dim=('lat','lon'))
            max_ssta = masked_one_obj_ssta_nans.max(dim=('lat','lon'))
            percentile_90 = masked_one_obj_ssta_nans.quantile(0.9, dim='time')
            percentile_90_per_timestep = masked_one_obj_ssta_nans.quantile(0.9, dim=('lat', 'lon'))
            ssta_std_per_timestep = masked_one_obj_ssta_nans.std(dim=('lat', 'lon'))
            ssta_std = masked_one_obj_ssta_nans.std()
    
            ssta_mean_data_ens_ls_all.append(mean_ssta.data)
            ssta_max_data_ens_ls_all.append(max_ssta.data)
            ssta_90percentile_ens_data_ls_all.append(percentile_90.data)
            ssta_90percentile_test_ens_data_ls_all.append(percentile_90_per_timestep.data)
            ssta_stdpertimestep_data_ens_ls_all.append(ssta_std_per_timestep.data)
            ssta_std_data_ens_ls_all.append(ssta_std.data)
    
        ssta_mean_data_ls_all.append(ssta_mean_data_ens_ls_all)
        ssta_max_data_ls_all.append(ssta_max_data_ens_ls_all)
        ssta_90percentile_data_ls_all.append(ssta_90percentile_ens_data_ls_all)
        ssta_90percentile_test_data_ls_all.append(ssta_90percentile_test_ens_data_ls_all)
        ssta_stdpertimestep_data_ls_all.append(ssta_stdpertimestep_data_ens_ls_all)
        ssta_std_data_ls_all.append(ssta_std_data_ens_ls_all)
    
    
    final_data_ssta = {
        "Mean SSTA": ssta_mean_data_ls_all,
        "Max SSTA": ssta_max_data_ls_all,
        "90th Percentile": ssta_90percentile_data_ls_all,
        "90th Percentile Per Timestep": ssta_90percentile_test_data_ls_all,
        "Standard Deviation per Timestep": ssta_stdpertimestep_data_ls_all,
        "Standard Deviation": ssta_std_data_ls_all,
    }