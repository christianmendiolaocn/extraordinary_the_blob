import os
import re
import bottleneck
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from tqdm import tqdm
import cmocean
from scipy.interpolate import interp1d
import imageio
import json
import pop_tools
import pickle
import inspect
import dask
import dask_jobqueue
import distributed
import dask.array as da
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xarray as xr
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

"""
This module applies all Ocetrac methods for labeling objects in 3 dimensions.
"""
import dask.array as dsa
import numpy as np
import scipy.ndimage
import xarray as xr

from skimage.measure import label as label_np
from skimage.measure import regionprops


def _apply_mask(binary_images, mask):
    binary_images_with_mask = binary_images.where(mask == 1, drop=False, other=0)
    return binary_images_with_mask

class Tracker:
    """
    Tracker object for applying Ocetrac
    """

    def __init__(
        self, da, mask, radius, min_size_quartile, timedim, xdim, ydim, positive=True
    ):
        self.da = da
        self.mask = mask
        self.radius = radius
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.ydim = ydim
        self.positive = positive

        if (timedim, ydim, xdim) != da.dims:
            try:
                da = da.transpose(timedim, ydim, xdim)
            except:
                raise ValueError(
                    f"Ocetrac currently only supports 3D DataArrays. The dimensions should only contain ({timedim}, {xdim}, and {ydim}). Found {list(da.dims)}"
                )

    def track(self):
        """
        Label and track image features.

        Parameters
        ----------
        da : xarray.DataArray
            The data to label.

        mask : xarray.DataArray
            The mask of points to ignore. Must be binary where 1 = true point and 0 = background to be ignored.

        radius : int
            The size of the structuring element used in morphological opening and closing. Radius specified by the number of grid units.
            
            Structuring elements are defined such that cells are included if their distance from the origin in index space is
            strictly less than the radius. For example, a radius of 1 means that the structuring element includes just an
            individual pixel (such that applying the morhological closing and opening just returns the original binary array),
            while a radius of 2 would additionally include the eight cells cells adjacent to the origin. As the radius increases,
            the shape of the structure element asymptotes to a circle centered on the origin.

        min_size_quartile : float
            The quantile used to define the threshold of the smallest area object retained in tracking. Value should be between 0 and 1.
            A value of exactly 0 means objects of any size are retained while a value of 1 retains just the single largest event.
            Higher values of `min_size_quartile` result in improved performance because less events are stored in memory and need to be
            compared.

        timedim : str
            The name of the time dimension

        xdim : str
            The name of the x dimension

        ydim : str
            The name of the y dimension

        positive : bool
            True if da values are expected to be positive, false if they are negative. Default argument is True

        Returns
        -------
        labels : xarray.DataArray
            Integer labels of the connected regions.
        """


        if (self.mask == 0).all():
            raise ValueError(
                "Found only zeros in `mask` input. The mask should indicate valid regions with values of 1"
            )

        # Convert data to binary, define structuring element, and perform morphological closing then opening
        binary_images = self._morphological_operations()

        # Apply mask
        binary_images_with_mask = _apply_mask(
            binary_images, self.mask
        )  # perhaps change to method? JB

        # Filter area
        area, min_area, binary_labels, N_initial = self._filter_area(
            binary_images_with_mask
        )

        # Label objects
        labels, num = self._label_either(binary_labels, return_num=True, connectivity=3)

        # Wrap labels
        grid_res = abs(self.da[self.xdim][1] - self.da[self.xdim][0])
        if self.da[self.xdim][-1] - self.da[self.xdim][0] >= 360 - grid_res:
            labels_wrapped, N_final = self._wrap(labels)
        else:
            labels_wrapped = labels
            N_final = np.max(labels)

        # Final labels to DataArray
        new_labels = xr.DataArray(
            labels_wrapped, dims=self.da.dims, coords=self.da.coords
        )
        new_labels = new_labels.where(new_labels != 0, drop=False, other=np.nan)

        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(np.sum(area.values))

        reject_area = area.where(area <= min_area, drop=True)
        sum_reject_area = int(np.sum(reject_area.values))
        percent_area_reject = sum_reject_area / sum_tot_area

        accept_area = area.where(area > min_area, drop=True)
        sum_accept_area = int(np.sum(accept_area.values))
        percent_area_accept = sum_accept_area / sum_tot_area

        new_labels = new_labels.rename("labels")
        new_labels.attrs["initial objects identified"] = int(N_initial)
        new_labels.attrs["final objects tracked"] = int(N_final)
        new_labels.attrs["radius"] = self.radius
        new_labels.attrs["size quantile threshold"] = self.min_size_quartile
        new_labels.attrs["min area"] = min_area
        new_labels.attrs["percent area reject"] = percent_area_reject
        new_labels.attrs["percent area accept"] = percent_area_accept

        print("initial objects identified \t", int(N_initial))
        print("final objects tracked \t", int(N_final))

        return new_labels

    ### PRIVATE METHODS - not meant to be called by user ###

    def _morphological_operations(self):
        """Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        Parameters
        ----------
        da     : xarray.DataArray
                The data to label
        radius : int
                Length of grid spacing that defines the radius of the structuring element used in morphological closing and opening.
                
                Structuring elements are defined such that cells are included if their distance from the origin in index space is
                strictly less than the radius. For example, a radius of 1 means that the structuring element includes just an
                individual pixel (such that applying the morhological closing and opening just returns the original binary array),
                while a radius of 2 would additionally include the eight cells cells adjacent to the origin. As the radius increases,
                the shape of the structure element asymptotes to a circle centered on the origin.

        """

        # Convert images to binary. All positive values == 1, otherwise == 0
        if self.positive == True:
            bitmap_binary = self.da.where(self.da > 0, drop=False, other=0)

        elif self.positive == False:
            bitmap_binary = self.da.where(self.da < 0, drop=False, other=0)

        bitmap_binary = bitmap_binary.where(bitmap_binary == 0, drop=False, other=1)

        # Define structuring element
        diameter = self.radius * 2
        x = np.arange(-self.radius, self.radius + 1)
        x, y = np.meshgrid(x, x)
        r = x**2 + y**2
        se = r < self.radius**2

        def binary_open_close(bitmap_binary):
            bitmap_binary_padded = np.pad(
                bitmap_binary, ((diameter, diameter), (diameter, diameter)), mode="wrap"
            )
            # If the radius is equal to 1, the structuring element is just an individual pixel, so
            # it is faster to just skip the calls to `binary_closing` and `binary_opening`
            if self.radius == 1:
                s2 = bitmap_binary_padded
            elif self.radius > 1:
                s1 = scipy.ndimage.binary_closing(bitmap_binary_padded, se, iterations=1)
                s2 = scipy.ndimage.binary_opening(s1, se, iterations=1)
            else:
                raise ValueError("radius must be an integer greater than or equal to 1")
                
            unpadded = s2[diameter:-diameter, diameter:-diameter]
            return unpadded

        mo_binary = xr.apply_ufunc(
            binary_open_close,
            bitmap_binary,
            input_core_dims=[[self.ydim, self.xdim]],
            output_core_dims=[[self.ydim, self.xdim]],
            output_dtypes=[bitmap_binary.dtype],
            vectorize=True,
            dask="parallelized",
        )
        return mo_binary

    # def _filter_area(self, binary_images):
    #     """calculatre area with regionprops"""

    #     def get_labels(binary_images):
    #         blobs_labels = self._label_either(binary_images, background=0)
    #         return blobs_labels

    #     labels = xr.apply_ufunc(
    #         get_labels,
        #     binary_images,
        #     input_core_dims=[[self.ydim, self.xdim]],
        #     output_core_dims=[[self.ydim, self.xdim]],
        #     output_dtypes=[binary_images.dtype],
        #     vectorize=True,
        #     dask="parallelized",
        # )

        # labels = xr.DataArray(
        #     labels, dims=binary_images.dims, coords=binary_images.coords
        # )
        # labels = labels.where(labels > 0, drop=False, other=np.nan)

        # # The labels are repeated each time step, therefore we relabel them to be consecutive
        # max_id = 0
        # for i in range(1, labels.shape[0]):
        #     max_id = np.nanmax([max_id, labels[i - 1, :, :].max().values])
        #     labels[i, :, :] = labels[i, :, :].values + max_id

        # labels = labels.where(labels > 0, drop=False, other=0)
        # labels_wrapped, N_initial = self._wrap(np.array(labels))

        # # Calculate Area of each object and keep objects larger than threshold
        # props = regionprops(labels_wrapped.astype("int"))

        # labelprops = [p.label for p in props]
        # labelprops = xr.DataArray(
        #     labelprops, dims=["label"], coords={"label": labelprops}
        # )

        # area = xr.DataArray(
        #     [p.area for p in props], dims=["label"], coords={"label": labelprops}
        # )  # Number of pixels of the region.

        # if area.size == 0:
        #     raise ValueError(
        #         f"No objects were detected. Try changing radius or min_size_quartile parameters."
        #     )

        # min_area = np.percentile(area, self.min_size_quartile * 100)
        # print(f"minimum area: {min_area}")

        # keep_labels = labelprops.where(area >= min_area, drop=True)
        # keep_where = np.isin(labels_wrapped, keep_labels)
        # out_labels = xr.DataArray(
        #     np.where(keep_where == False, 0, labels_wrapped),
        #     dims=binary_images.dims,
        #     coords=binary_images.coords,
        # )

        # # Convert images to binary. All positive values == 1, otherwise == 0
        # binary_labels = out_labels.where(out_labels == 0, drop=False, other=1)

        # return area, min_area, binary_labels, N_initial

    def _filter_area(self, binary_images):
        """Calculate area with regionprops and filter objects based on size."""

        def get_labels(binary_images):
            blobs_labels = self._label_either(binary_images, background=0)
            return blobs_labels
    
        labels = xr.apply_ufunc(
            get_labels,
            binary_images,
            input_core_dims=[[self.ydim, self.xdim]],
            output_core_dims=[[self.ydim, self.xdim]],
            output_dtypes=[binary_images.dtype],
            vectorize=True,
            dask="parallelized",
        )
    
        labels = xr.DataArray(
            labels, dims=binary_images.dims, coords=binary_images.coords
        )
        labels = labels.where(labels > 0, drop=False, other=np.nan)
    
        # The labels are repeated each time step, therefore we relabel them to be consecutive
        max_id = 0
        for i in range(labels.shape[0]):
            # Check if the current timestep has any features
            if np.nanmax(labels[i, :, :].values) > 0:
                max_id = np.nanmax([max_id, labels[i - 1, :, :].max().values])
                labels[i, :, :] = labels[i, :, :].values + max_id
            else:
                # If no features are detected, keep the labels as 0
                labels[i, :, :] = 0
    
        labels = labels.where(labels > 0, drop=False, other=0)
        labels_wrapped, N_initial = self._wrap(np.array(labels))
    
        # Calculate Area of each object and keep objects larger than threshold
        props = regionprops(labels_wrapped.astype("int"))
    
        labelprops = [p.label for p in props]
        labelprops = xr.DataArray(
            labelprops, dims=["label"], coords={"label": labelprops}
        )
    
        area = xr.DataArray(
            [p.area for p in props], dims=["label"], coords={"label": labelprops}
        )  # Number of pixels of the region.
    
        if area.size == 0:
            raise ValueError(
                f"No objects were detected. Try changing radius or min_size_quartile parameters."
            )

        min_area = np.percentile(area, self.min_size_quartile * 100)
        print(f"minimum area: {min_area}")
    
        keep_labels = labelprops.where(area >= min_area, drop=True)
        keep_where = np.isin(labels_wrapped, keep_labels)
        out_labels = xr.DataArray(
            np.where(keep_where == False, 0, labels_wrapped),
            dims=binary_images.dims,
            coords=binary_images.coords,
        )
    
        # Convert images to binary. All positive values == 1, otherwise == 0
        binary_labels = out_labels.where(out_labels == 0, drop=False, other=1)
    
        return area, min_area, binary_labels, N_initial
    
    def _label_either(self, data, **kwargs):
        if isinstance(data, dsa.Array):
            try:
                from dask_image.ndmeasure import label as label_dask

                def label_func(a, **kwargs):
                    ids, num = label_dask(a, **kwargs)
                    return ids

            except ImportError:
                raise ImportError(
                    "Dask_image is required to use this function on Dask arrays. "
                    "Either install dask_image or else call .load() on your data."
                )
        else:
            label_func = label_np
        return label_func(data, **kwargs)

    def _wrap(self, labels):
        """Impose periodic boundary and wrap labels"""
        first_column = labels[..., 0]
        last_column = labels[..., -1]

        unique_first = np.unique(first_column[first_column > 0])

        # This loop iterates over the unique values in the first column, finds the location of those values in
        # the first columnm and then uses that index to replace the values in the last column with the first column value
        for i in enumerate(unique_first):
            first = np.where(first_column == i[1])
            last = last_column[first[0], first[1]]
            bad_labels = np.unique(last[last > 0])
            replace = np.isin(labels, bad_labels)
            labels[replace] = i[1]

        labels_wrapped = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)

        # recalculate the total number of labels
        N = np.max(labels_wrapped)

        return labels_wrapped, N

def get_var_paths(directory, var):
    # Directory containing the files; 'var' is a variable name placeholder
    # directory = f'/glade/campaign/cgd/cesm/CESM2-LE/{comp}/proc/tseries/month_1/{var}/'
    
    # Prefixes to match for future and historical datasets
    prefixes_to_match_fut = ['b.e21.BSSP370cmip6.', 'b.e21.BSSP370smbb.']
    prefixes_to_match_hist = ['b.e21.BHISTcmip6.', 'b.e21.BHISTsmbb.']
    
    # Sets to store unique prefixes for future and historical filenames
    prefixes_fut = set()
    prefixes_hist = set()
    
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check and add prefixes for future scenario files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_fut) and filename.endswith('.nc'):
            prefixes_fut.add(filename.rsplit('.', 3)[0])
        
        # Check and add prefixes for historical scenario files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_hist) and filename.endswith('.nc'):
            prefixes_hist.add(filename.rsplit('.', 3)[0])
    
    # Convert sets to lists for further use
    path_intermed_fut = list(prefixes_fut)
    path_intermed_hist = list(prefixes_hist)
    return path_intermed_hist, path_intermed_fut

def get_ds_var(var, comp, index_hist):
    directory = f'/glade/campaign/cgd/cesm/CESM2-LE/{comp}/proc/tseries/month_1/{var}/'
    path_intermed_hist, path_intermed_fut = get_var_paths(directory, var)
    filename_identifier = '.'.join(path_intermed_hist[index_hist].rsplit('.', 5)[1:4])
    index_fut = find_identifier_with_index(path_intermed_hist, filename_identifier)[0][1]
    hist_file_paths = get_hist_file_paths(var,directory, path_intermed_hist, index_hist)
    fut_file_paths = get_fut_file_paths(var, directory, path_intermed_fut, index_fut)
    ds_var_fut = file_path_to_var_ds(fut_file_paths)
    ds_var_hist = file_path_to_var_ds(hist_file_paths)
    return ds_var_hist, ds_var_fut

def find_identifier_with_index(prefixes, identifier):
    """
    Find prefixes that contain a specific identifier and their indices.

    Parameters:
        prefixes (list): A list of prefixes to search through.
        identifier (str): The identifier to search for in the prefixes.

    Returns:
        list: A list of tuples containing matching prefixes and their indices.
    """
    matching_prefixes_with_indices = []  # Initialize a list to store matches and their indices

    for index, prefix in enumerate(prefixes):  # Use enumerate to get both index and prefix
        if identifier in prefix:  # Check if the identifier is in the prefix
            matching_prefixes_with_indices.append((prefix, index))  # Add the prefix and index as a tuple

    return matching_prefixes_with_indices  # Return the list of matching prefixes and indices
    
def get_hist_file_paths(var, directory, path_intermed_hist, index):
    attrib_title = path_intermed_hist[index]
    file_paths = []
    for start_year in range(1850, 2010, 10):
        end_year = start_year + 9
        file_path = f'{directory}{attrib_title}.{var}.{start_year}01-{end_year}12.nc'
        file_paths.append(file_path)
    last_file_path = f'{directory}{attrib_title}.{var}.201001-201412.nc'
    file_paths.append(last_file_path)
    return file_paths

def get_fut_file_paths(var, directory, path_intermed_fut, index):
    attrib_title = path_intermed_fut[index]
    file_paths = []
    for start_year in range(2015, 2095, 10):
        end_year = start_year + 9
        file_path = f'{directory}{attrib_title}.{var}.{start_year}01-{end_year}12.nc'
        file_paths.append(file_path)
    last_file_path = f'{directory}{attrib_title}.{var}.209501-210012.nc'
    file_paths.append(last_file_path)
    return file_paths

def file_path_to_var_ds(file_paths):
    var_ds = xr.open_mfdataset(file_paths, 
                                 concat_dim='time', 
                                 combine='nested', 
                                 parallel=True)
    return var_ds

def get_var_paths(directory, var):
    # Prefixes to match for future and historical datasets
    prefixes_to_match_fut = ['b.e21.BSSP370cmip6.', 'b.e21.BSSP370smbb.']
    prefixes_to_match_hist = ['b.e21.BHISTcmip6.', 'b.e21.BHISTsmbb.']
        
    # Sets to store unique prefixes for future and historical filenames
    prefixes_fut = list()
    prefixes_hist = list()
        
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check and add prefixes for future scenario files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_fut) and filename.endswith('.nc'):
            prefixes_fut.append(filename.rsplit('.', 3)[0])
            
        # Check and add prefixes for historical scenario files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_hist) and filename.endswith('.nc'):
            prefixes_hist.append(filename.rsplit('.', 3)[0])
        
    prefixes_hist_set = set(prefixes_hist)
    sorted_unique_list_hist = sorted(prefixes_hist_set)
    
    prefixes_fut_set = set(prefixes_fut)
    sorted_unique_list_fut = sorted(prefixes_fut_set)
    
    path_intermed_fut = sorted_unique_list_fut
    path_intermed_hist = sorted_unique_list_hist

    return path_intermed_hist, path_intermed_fut

def get_ds_var(directory, var, comp, index_hist):
    path_intermed_hist, path_intermed_fut = get_var_paths(directory, var)
    filename_identifier = '.'.join(path_intermed_hist[index_hist].rsplit('.', 5)[1:4])
    index_fut = find_identifier_with_index(path_intermed_hist, filename_identifier)[0][1]
    hist_file_paths = get_hist_file_paths(var,directory, path_intermed_hist, index_hist)
    fut_file_paths = get_fut_file_paths(var, directory, path_intermed_fut, index_fut)
    ds_var_fut = file_path_to_var_ds(fut_file_paths)
    ds_var_hist = file_path_to_var_ds(hist_file_paths)
    return ds_var_hist, ds_var_fut

####################################### TRENDS
def calc_anomalies_features_removelineartrend(ds):
    '''
    remove linear trend, seasonal cycle, and mean
    '''
    sst = ds
    dyr = ds.time.dt.year + ds.time.dt.month/12
    # Our 6 coefficient model is composed of the mean, trend, annual sine and cosine harmonics, & semi-annual sine and cosine harmonics
    model = np.array([np.ones(len(dyr))] + [dyr-np.mean(dyr)] + [np.sin(2*np.pi*dyr)] + [np.cos(2*np.pi*dyr)] + [np.sin(4*np.pi*dyr)] + [np.cos(4*np.pi*dyr)])
    
    # Take the pseudo-inverse of model to 'solve' least-squares problem
    pmodel = np.linalg.pinv(model)
    
    # Convert model and pmodel to xaray DataArray
    model_da = xr.DataArray(model.T, dims=['time','coeff'], coords={'time':sst.time.values, 'coeff':np.arange(1,7,1)}) 
    pmodel_da = xr.DataArray(pmodel.T, dims=['coeff','time'], coords={'coeff':np.arange(1,7,1), 'time':sst.time.values})  
    
    # resulting coefficients of the model
    sst_mod = xr.DataArray(pmodel_da.dot(sst), dims=['coeff','lat','lon'], coords={'coeff':np.arange(1,7,1), 'lat':sst.lat.values, 'lon':sst.lon.values})
    
    # Construct mean, trend, and seasonal cycle
    mean = model_da[:,0].dot(sst_mod[0,:,:])
    trend = model_da[:,1].dot(sst_mod[1,:,:])
    seas = model_da[:,2:].dot(sst_mod[2:,:,:])
    
    # compute anomalies by removing all  the model coefficients 
    ssta_notrend = sst-model_da.dot(sst_mod)
    
    # Use the 90th percentile as a threshold and find anomalies that exceed it. 
    if ssta_notrend.chunks:
        ssta_notrend = ssta_notrend.chunk({'time': -1})
    
    threshold = ssta_notrend.quantile(.9, dim=('time'))
    features_notrend = ssta_notrend.where(ssta_notrend>=threshold, other=np.nan)
    return mean, trend, seas, features_notrend, ssta_notrend

def calc_anomalies_features_removeseasonalcycle(ds):
    '''
    remove seasonal cycle and mean
    '''
    sst = ds
    dyr = ds.time.dt.year + ds.time.dt.month/12
    # Our 6 coefficient model is composed of the mean, trend, annual sine and cosine harmonics, & semi-annual sine and cosine harmonics
    model = np.array([np.ones(len(dyr))] + [dyr-np.mean(dyr)] + [np.sin(2*np.pi*dyr)] + [np.cos(2*np.pi*dyr)] + [np.sin(4*np.pi*dyr)] + [np.cos(4*np.pi*dyr)])
    
    # Take the pseudo-inverse of model to 'solve' least-squares problem
    pmodel = np.linalg.pinv(model)
    
    # Convert model and pmodel to xarray DataArray
    model_da = xr.DataArray(model.T, dims=['time','coeff'], coords={'time':sst.time.values, 'coeff':np.arange(1,7,1)}) 
    pmodel_da = xr.DataArray(pmodel.T, dims=['coeff','time'], coords={'coeff':np.arange(1,7,1), 'time':sst.time.values})  
    
    # resulting coefficients of the model
    sst_mod = xr.DataArray(pmodel_da.dot(sst), dims=['coeff','lat','lon'], coords={'coeff':np.arange(1,7,1), 'lat':sst.lat.values, 'lon':sst.lon.values})
    
    # Construct mean, trend, and seasonal cycle
    mean = model_da[:,0].dot(sst_mod[0,:,:])
    trend = model_da[:,1].dot(sst_mod[1,:,:])
    seas = model_da[:,2:].dot(sst_mod[2:,:,:])
    
    # compute anomalies by removing only the mean and seasonal cycle (keeping the trend)
    ssta_withtrend = sst - (mean + seas)
    
    # Use the 90th percentile as a threshold and find anomalies that exceed it. 
    if ssta_withtrend.chunks:
        ssta_withtrend = ssta_withtrend.chunk({'time': -1})
    
    threshold = ssta_withtrend.quantile(.9, dim=('time'))
    features_withtrend = ssta_withtrend.where(ssta_withtrend>=threshold, other=np.nan)
    return mean, trend, seas, features_withtrend, ssta_withtrend

def calc_anomalies_features_removeonlylineartrend(ds):
    '''
    remove linear trend (retain seasonal cycle)
    '''
    sst = ds
    dyr = ds.time.dt.year + ds.time.dt.month/12
    # Our 6 coefficient model is composed of the mean, trend, annual sine and cosine harmonics, & semi-annual sine and cosine harmonics
    model = np.array([np.ones(len(dyr))] + [dyr-np.mean(dyr)] + [np.sin(2*np.pi*dyr)] + [np.cos(2*np.pi*dyr)] + [np.sin(4*np.pi*dyr)] + [np.cos(4*np.pi*dyr)])
    
    # Take the pseudo-inverse of model to 'solve' least-squares problem
    pmodel = np.linalg.pinv(model)
    
    # Convert model and pmodel to xarray DataArray
    model_da = xr.DataArray(model.T, dims=['time','coeff'], coords={'time':sst.time.values, 'coeff':np.arange(1,7,1)}) 
    pmodel_da = xr.DataArray(pmodel.T, dims=['coeff','time'], coords={'coeff':np.arange(1,7,1), 'time':sst.time.values})  
    
    # resulting coefficients of the model
    sst_mod = xr.DataArray(pmodel_da.dot(sst), dims=['coeff','lat','lon'], coords={'coeff':np.arange(1,7,1), 'lat':sst.lat.values, 'lon':sst.lon.values})
    
    # Construct mean, trend, and seasonal cycle
    mean = model_da[:,0].dot(sst_mod[0,:,:])
    trend = model_da[:,1].dot(sst_mod[1,:,:])
    seas = model_da[:,2:].dot(sst_mod[2:,:,:])
    
    # compute anomalies by removing only the mean and seasonal cycle (keeping the trend)
    ssta_withtrend = sst - (trend)
    
    # Use the 90th percentile as a threshold and find anomalies that exceed it. 
    if ssta_withtrend.chunks:
        ssta_withtrend = ssta_withtrend.chunk({'time': -1})
    
    threshold = ssta_withtrend.quantile(.9, dim=('time'))
    features_withtrend = ssta_withtrend.where(ssta_withtrend>=threshold, other=np.nan)
    return mean, trend, seas, features_withtrend, ssta_withtrend

# Calculate anomalies by removing quadratic trend and seasonal cycle
def calc_anomalies_features_removequadratictrend(ds):
    '''
    Remove quadratic trend, seasonal cycle (annual and semiannual harmonics), and mean.
    Seasonal cycle removed using same harmonic approach as linear detrending function.
    '''
    sst = ds
    dyr = ds.time.dt.year + ds.time.dt.month/12
    time_numeric = dyr - dyr.mean()  # Center for better numerical stability


    # --- Step 1: Remove quadratic trend ---
    def fit_quadratic_trend(time_vals, data_vals):
        if np.isnan(data_vals).all():
            return np.full_like(time_vals, np.nan)
        mask = ~np.isnan(data_vals)
        if np.sum(mask) < 3:
            return np.full_like(time_vals, np.nan)
        try:
            coeffs = np.polyfit(time_vals[mask], data_vals[mask], 2)
            return np.polyval(coeffs, time_vals)
        except:
            return np.full_like(time_vals, np.nan)


    quadratic_trend = xr.apply_ufunc(
        fit_quadratic_trend,
        time_numeric, sst,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[sst.dtype]
    )


    sst_detrended = sst - quadratic_trend


    # --- Step 2: Remove seasonal cycle using annual and semiannual harmonics ---
    # Same 4-coefficient harmonic model as linear detrending (no mean or trend terms)
    harmonics = np.array(
        [np.sin(2 * np.pi * dyr)] +
        [np.cos(2 * np.pi * dyr)] +
        [np.sin(4 * np.pi * dyr)] +
        [np.cos(4 * np.pi * dyr)]
    )


    pharmonics = np.linalg.pinv(harmonics)


    harmonics_da = xr.DataArray(
        harmonics.T,
        dims=['time', 'coeff'],
        coords={'time': sst.time.values, 'coeff': np.arange(1, 5, 1)}
    )
    pharmonics_da = xr.DataArray(
        pharmonics.T,
        dims=['coeff', 'time'],
        coords={'coeff': np.arange(1, 5, 1), 'time': sst.time.values}
    )


    # Fit harmonics to detrended SST
    sst_mod = xr.DataArray(
        pharmonics_da.dot(sst_detrended),
        dims=['coeff', 'lat', 'lon'],
        coords={'coeff': np.arange(1, 5, 1), 'lat': sst.lat.values, 'lon': sst.lon.values}
    )


    # Reconstruct and remove seasonal cycle
    seas = harmonics_da.dot(sst_mod)
    ssta = sst_detrended - seas


    # --- Step 3: Threshold and detect features ---
    if ssta.chunks:
        ssta = ssta.chunk({'time': -1})


    threshold = ssta.quantile(.9, dim=('time'))
    features = ssta.where(ssta >= threshold, other=np.nan)


    return features, ssta

def access_ds(
    DIRECTORY,
    ens_memb_index, 
    UPPER_LAT, 
    LOWER_LAT,
    LEFT_LON,
    RIGHT_LON
):

    ds_var_hist_SST, ds_var_fut_SST = get_ds_var(
        DIRECTORY, 'SST','atm', ens_memb_index)
    
    CESMLENS_SST_NEP_hist = ds_var_hist_SST.SST.sel(
        lon=slice(LEFT_LON, RIGHT_LON), lat=slice(LOWER_LAT,UPPER_LAT))
    
    CESMLENS_SST_NEP_fut = ds_var_fut_SST.SST.sel(
        lon=slice(LEFT_LON, RIGHT_LON), lat=slice(LOWER_LAT,UPPER_LAT))
    
    CESMLENS_SST_NEP_hist_time_slice = CESMLENS_SST_NEP_hist.sel(
        time=slice('1979-01-01', '2015-01-01'))
    
    CESMLENS_SST_NEP_fut_time_slice = CESMLENS_SST_NEP_fut.sel(
        time=slice('2015-02-01', '2022-12-01'))
    
    CESMLENS_SST_NEP_ds = xr.concat(
        [CESMLENS_SST_NEP_hist_time_slice, CESMLENS_SST_NEP_fut_time_slice], 
        dim='time')
    
    CESMLENS_SST_NEP_ds = CESMLENS_SST_NEP_ds.compute()
    
    CESMLENS_SST_NEP_ds_no_nan = CESMLENS_SST_NEP_ds.where(
        CESMLENS_SST_NEP_ds != 0, np.nan)
    return CESMLENS_SST_NEP_ds_no_nan

def apply_ocetrac(features_notrend, mean, RAD_VAL=2):
    full_mask_land = features_notrend
    full_masked = full_mask_land.where(full_mask_land != 0)
    binary_out_afterlandmask=np.isfinite(full_masked)
    newmask = np.isfinite(mean[:,:,:][:])
        
    binary_out_afterlandmask = binary_out_afterlandmask.compute()
    newmask = newmask.compute()

    obj_Tracker = Tracker(
        binary_out_afterlandmask[:,:,:],
        newmask, 
        radius=RAD_VAL,
        min_size_quartile= 0.75, # check these values
        timedim = 'time', 
        xdim = 'lon', 
        ydim='lat', 
        positive=True)
    blobs = obj_Tracker.track()
    blobs.attrs
    mo = obj_Tracker._morphological_operations()
    return blobs

def process_blobs_ssta_features(CESMLENS_SST_NEP_ds_no_nan, region_ensemble_mean_ds, ens_memb_index):

    
    ############ Remove linear trend, seasonal cycle, mean
    print('=================== 1 ===================')
    mean, trend, seas, features_notrend, ssta_notrend = calc_anomalies_features_removelineartrend(
        ds=CESMLENS_SST_NEP_ds_no_nan)

    # -------------
    blobs_1 = apply_ocetrac(features_notrend, mean, RAD_VAL = 1)
    print()
    blobs_2 = apply_ocetrac(features_notrend, mean, RAD_VAL = 2)
    print()
    blobs_3 = apply_ocetrac(features_notrend, mean, RAD_VAL = 3)
    print()
    blobs_4 = apply_ocetrac(features_notrend, mean, RAD_VAL = 4)
    print()
    blobs_5 = apply_ocetrac(features_notrend, mean, RAD_VAL = 5)
    print()

    combined_ensemble = xr.concat([blobs_1, blobs_2, blobs_3, blobs_4, blobs_5], dim='radius')
    combined_ensemble.to_netcdf(
        '/glade/work/cmendiola/data_conv_lin_trend/ens_{}_mhwobj.nc'.format(ens_memb_index))
    
    # # -------------
    # blobs_1.to_netcdf(
    #     'data_conv_lin_trend/ens_{}_mhwobj_rad1.nc'.format(ens_memb_index))
    # blobs_2.to_netcdf(
    #     'data_conv_lin_trend/ens_{}_mhwobj_rad2.nc'.format(ens_memb_index))
    # blobs_3.to_netcdf(
    #     'data_conv_lin_trend/ens_{}_mhwobj_rad3.nc'.format(ens_memb_index))
    # blobs_5.to_netcdf(
    #     'data_conv_lin_trend/ens_{}_mhwobj_rad5.nc'.format(ens_memb_index))
    
    # # -------------
    ssta_notrend.to_netcdf(
        '/glade/work/cmendiola/data_conv_lin_trend/ens_{}_ssta.nc'.format(ens_memb_index))


    ############ No ensemble mean
    print('=================== 2 ===================')
    cesmlens_sst_detrend_using_ensmean = CESMLENS_SST_NEP_ds_no_nan - region_ensemble_mean_ds
    threshold = cesmlens_sst_detrend_using_ensmean.quantile(.9, dim=('time'))
    
    features_noensmean = cesmlens_sst_detrend_using_ensmean.where(
        cesmlens_sst_detrend_using_ensmean >= threshold, other=np.nan)
    
    # -------------
    blobs_1 = apply_ocetrac(features_noensmean, mean, RAD_VAL = 1)
    print()
    blobs_2 = apply_ocetrac(features_noensmean, mean, RAD_VAL = 2)
    print()
    blobs_3 = apply_ocetrac(features_noensmean, mean, RAD_VAL = 3)
    print()
    blobs_4 = apply_ocetrac(features_noensmean, mean, RAD_VAL = 4)
    print()
    blobs_5 = apply_ocetrac(features_noensmean, mean, RAD_VAL = 5)

    combined_ensemble = xr.concat([blobs_1, blobs_2, blobs_3, blobs_4, blobs_5], dim='radius')
    combined_ensemble.to_netcdf(
        '/glade/work/cmendiola/data_ens_mean/ens_{}_mhwobj.nc'.format(ens_memb_index))
    
    # # -------------
    # blobs_1.to_netcdf(
    #     'data_ens_mean/ens_{}_mhwobj_rad1.nc'.format(ens_memb_index))
    # blobs_2.to_netcdf(
    #     'data_ens_mean/ens_{}_mhwobj_rad2.nc'.format(ens_memb_index))
    # blobs_3.to_netcdf(
    #     'data_ens_mean/ens_{}_mhwobj_rad3.nc'.format(ens_memb_index))
    # blobs_5.to_netcdf(
    #     'data_ens_mean/ens_{}_mhwobj_rad5.nc'.format(ens_memb_index))
    
    # -------------
    cesmlens_sst_detrend_using_ensmean.to_netcdf(
        '/glade/work/cmendiola/data_ens_mean/ens_{}_ssta.nc'.format(ens_memb_index))
    
    ############ Remove seasonal cycle and mean
    print('=================== 3 ===================')
    mean, trend, seas, features_notrend, ssta_notrend = calc_anomalies_features_removeseasonalcycle(
        ds=CESMLENS_SST_NEP_ds_no_nan)
    
    # -------------
    blobs_1 = apply_ocetrac(features_notrend, mean, RAD_VAL = 1)
    print()
    blobs_2 = apply_ocetrac(features_notrend, mean, RAD_VAL = 2)
    print()
    blobs_3 = apply_ocetrac(features_notrend, mean, RAD_VAL = 3)
    print()
    blobs_4 = apply_ocetrac(features_notrend, mean, RAD_VAL = 4)
    print()
    blobs_5 = apply_ocetrac(features_notrend, mean, RAD_VAL = 5)
    print()

    combined_ensemble = xr.concat([blobs_1, blobs_2, blobs_3, blobs_4, blobs_5], dim='radius')
    combined_ensemble.to_netcdf(
        '/glade/work/cmendiola/data_rm_seasonalcycle_mean/ens_{}_mhwobj.nc'.format(ens_memb_index))
    
    # # -------------
    # blobs_1.to_netcdf(
    #     'data_rm_seasonalcycle_mean/ens_{}_mhwobj_rad1.nc'.format(ens_memb_index))
    # blobs_2.to_netcdf(
    #     'data_rm_seasonalcycle_mean/ens_{}_mhwobj_rad2.nc'.format(ens_memb_index))
    # blobs_3.to_netcdf(
    #     'data_rm_seasonalcycle_mean/ens_{}_mhwobj_rad3.nc'.format(ens_memb_index))
    # blobs_5.to_netcdf(
    #     'data_rm_seasonalcycle_mean/ens_{}_mhwobj_rad5.nc'.format(ens_memb_index))
    
    # -------------
    ssta_notrend.to_netcdf(
        '/glade/work/cmendiola/data_rm_seasonalcycle_mean/ens_{}_ssta.nc'.format(ens_memb_index))
    
    ############ Remove linear trend only
    print('=================== 4 ===================')
    mean, trend, seas, features_notrend, ssta_notrend = calc_anomalies_features_removeonlylineartrend(
        ds=CESMLENS_SST_NEP_ds_no_nan)
    
    # -------------
    blobs_1 = apply_ocetrac(features_notrend, mean, RAD_VAL = 1)
    print()
    blobs_2 = apply_ocetrac(features_notrend, mean, RAD_VAL = 2)
    print()
    blobs_3 = apply_ocetrac(features_notrend, mean, RAD_VAL = 3)
    print()
    blobs_4 = apply_ocetrac(features_notrend, mean, RAD_VAL = 4)
    print()
    blobs_5 = apply_ocetrac(features_notrend, mean, RAD_VAL = 5)
    print()

    combined_ensemble = xr.concat([blobs_1, blobs_2, blobs_3, blobs_4, blobs_5], dim='radius')
    combined_ensemble.to_netcdf(
        '/glade/work/cmendiola/data_rm_only_lin_trend/ens_{}_mhwobj.nc'.format(ens_memb_index))
    
    # # -------------
    # blobs_1.to_netcdf(
    #     'data_rm_only_lin_trend/ens_{}_mhwobj_rad1.nc'.format(ens_memb_index))
    # blobs_2.to_netcdf(
    #     'data_rm_only_lin_trend/ens_{}_mhwobj_rad2.nc'.format(ens_memb_index))
    # blobs_3.to_netcdf(
    #     'data_rm_only_lin_trend/ens_{}_mhwobj_rad3.nc'.format(ens_memb_index))
    # blobs_5.to_netcdf(
    #     'data_rm_only_lin_trend/ens_{}_mhwobj_rad5.nc'.format(ens_memb_index))
    
    # -------------
    ssta_notrend.to_netcdf(
        '/glade/work/cmendiola/data_rm_only_lin_trend/ens_{}_ssta.nc'.format(ens_memb_index))
    
    ########### Remove quadratic trend and seasonal cycle and mean 
    print('=================== 5 ===================')
    # could be funcs.calc_anomalies_features_removequadratictrend if loaded correctly
    features_noquadratic, ssta_noquadratic = calc_anomalies_features_removequadratictrend(
        ds=CESMLENS_SST_NEP_ds_no_nan)
    
    # -------------
    blobs_1 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 1)
    print()
    blobs_2 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 2)
    print()
    blobs_3 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 3)
    print()
    blobs_4 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 4)
    print()
    blobs_5 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 5)
    print()

    combined_ensemble = xr.concat([blobs_1, blobs_2, blobs_3, blobs_4, blobs_5], dim='radius')
    combined_ensemble.to_netcdf(
        '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj.nc'.format(ens_memb_index))
    
    # # -------------
    # blobs_1.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad1.nc'.format(ens_memb_index))
    # blobs_2.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad2.nc'.format(ens_memb_index))
    # blobs_3.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad3.nc'.format(ens_memb_index))
    # blobs_5.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad5.nc'.format(ens_memb_index))
    
    # -------------
    ssta_noquadratic.to_netcdf(
        '/glade/work/cmendiola/data_quad_trend/ens_{}_ssta.nc'.format(ens_memb_index))

# Calculating mean # of MHW enevts across all ensembles and radii (used in nkjnksjdc.ipynb)

def process_blobs_ssta_features_onlyquad(CESMLENS_SST_NEP_ds_no_nan, ens_memb_index):

    ########### Remove quadratic trend and seasonal cycle and mean 
    print('=================== 5 ===================')
    # could be funcs.calc_anomalies_features_removequadratictrend if loaded correctly
    features_noquadratic, ssta_noquadratic = calc_anomalies_features_removequadratictrend(
        ds=CESMLENS_SST_NEP_ds_no_nan)
    mean = np.isfinite(CESMLENS_SST_NEP_ds_no_nan)
    # -------------
    blobs_1 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 1)
    print()
    blobs_2 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 2)
    print()
    blobs_3 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 3)
    print()
    blobs_4 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 4)
    print()
    blobs_5 = apply_ocetrac(features_noquadratic, mean, RAD_VAL = 5)
    print()

    combined_ensemble = xr.concat([blobs_1, blobs_2, blobs_3, blobs_4, blobs_5], dim='radius')
    combined_ensemble.to_netcdf(
        '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj.nc'.format(ens_memb_index))
    
    # # -------------
    # blobs_1.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad1.nc'.format(ens_memb_index))
    # blobs_2.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad2.nc'.format(ens_memb_index))
    # blobs_3.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad3.nc'.format(ens_memb_index))
    # blobs_5.to_netcdf(
    #     '/glade/work/cmendiola/data_quad_trend/ens_{}_mhwobj_rad5.nc'.format(ens_memb_index))
    
    # -------------
    ssta_noquadratic.to_netcdf(
        '/glade/work/cmendiola/data_quad_trend/ens_{}_ssta.nc'.format(ens_memb_index))


def calculate_mean_objcount_all_ens_all_rad(ds):
    for RADIUS_IND in range(0,5):
        mhw_rad_ds = ds[:,RADIUS_IND,:,:]
        objct_all = []
        for ensemble_member_id in range(100):
            objct_ens = []
            blobs = mhw_rad_ds.isel(ensemble_member = ensemble_member_id).compute()
            objct_ens.append(np.unique(blobs.data)[:-1][-1])
            objct_all.append(objct_ens)
        print(f"\nResults for Radius {RADIUS_IND + 1}:")
        print(f"  STD: {np.std(objct_all)}")
        print(f"  Mean: {np.mean(objct_all)}")
        eventmean = np.mean(objct_all)
        eventstd = np.std(objct_all)
    return eventmean, eventstd

# Calculate time of occurence of MHWs across all ensembles (month and year) (used in nkjnksjdc.ipynb)

def get_time_of_events_all_rad(ds):
    frame = inspect.currentframe().f_back
    ds_name = [name for name, val in frame.f_locals.items() if val is ds]
    ds_name = ds_name[0] if ds_name else "dataset"
    radius_years = {}
    radius_months = {}
    
    for RADIUS_IND in range(0, 5):
        mhw_rad_ds = ds[:, RADIUS_IND, :, :]
        
        years = []
        months = []
        
        for ensemble_member_id in range(100):
            blobs = mhw_rad_ds.isel(ensemble_member=ensemble_member_id).compute()
            for object_id in np.unique(blobs.data):
                if object_id == 0:
                    continue  # Skip background
                
                object_count_per_time = (blobs == object_id).sum(dim=['lat', 'lon'])
                true_time_steps = object_count_per_time.time.where(object_count_per_time > 0, drop=True)
    
                if true_time_steps.shape[0] == 0:
                    continue
    
                dt = true_time_steps[0].item()
                years.append(dt.year)
                months.append(dt.month)
        
        # Save results under a label for this radius
        radius_years[f"radius_{RADIUS_IND + 1}"] = years
        radius_months[f"radius_{RADIUS_IND + 1}"] = months
    
        print(f"Finished Radius {RADIUS_IND + 1}: Found {len(years)} objects.")
    with open(f"{ds_name}_time_years.json", "w") as f:
        json.dump(radius_years, f, indent=2)
    with open(f"{ds_name}_time_months.json", "w") as f:
        json.dump(radius_months, f, indent=2)
    return radius_years, radius_months

# Calculate durations of MHW across all ensembles (used in nkjnksjdc.ipynb)

def calculate_durations_across_ens(ds, RADIUS_IND):
    mhw_rad_ds = ds[:,RADIUS_IND,:,:]
    duration_all = []
    for ensemble_member_id in range(100):
        duration_ens = []
        blobs = mhw_rad_ds.isel(ensemble_member = ensemble_member_id).compute()
        for object_id in np.unique(blobs.data)[:-1]:
            object_count_per_time = (blobs == object_id).sum(dim=['lat', 'lon'])  
            object_count_per_time_nan = xr.where(object_count_per_time == 0, np.nan, object_id).dropna(dim='time')
            duration_ens.append(object_count_per_time_nan.shape[0])
        duration_all.append(duration_ens)
    return duration_all